"""Enricher — Pipeline-based, column-oriented DataFrame enrichment.

Primary public interface for Lattice v0.3.  Accepts a Pipeline +
FieldManager, validates field routing, manages per-step checkpoints,
and provides sync/async DataFrame APIs.
"""

import asyncio
import warnings
from typing import Any, Optional

import pandas as pd

from .checkpoint import CheckpointManager
from .config import EnrichmentConfig
from .exceptions import EnrichmentError, FieldValidationError
from ..data import FieldManager
from ..pipeline import Pipeline
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Enricher:
    """Orchestrates column-oriented pipeline execution over a DataFrame.

    Usage::

        enricher = Enricher(pipeline, field_manager)
        df = enricher.run(df, "company_info")
    """

    def __init__(
        self,
        pipeline: Pipeline,
        field_manager: FieldManager,
        config: Optional[EnrichmentConfig] = None,
    ) -> None:
        self.pipeline = pipeline
        self.field_manager = field_manager
        self.config = config or EnrichmentConfig()
        self._checkpoint = CheckpointManager(self.config)

    # -- sync entry point ------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        category: str,
        overwrite_fields: Optional[bool] = None,
        data_identifier: Optional[str] = None,
    ) -> pd.DataFrame:
        """Synchronous wrapper around :meth:`run_async`.

        Raises ``RuntimeError`` if called from inside a running event loop
        (use ``run_async`` directly in that case).
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Enricher.run() cannot be called from inside an async context. "
                "Use 'await enricher.run_async(...)' instead."
            )
        except RuntimeError as exc:
            # Re-raise our own RuntimeError; swallow the "no running event loop" one
            if "run_async" in str(exc):
                raise
        return asyncio.run(
            self.run_async(df, category, overwrite_fields, data_identifier)
        )

    # -- async entry point -----------------------------------------------

    async def run_async(
        self,
        df: pd.DataFrame,
        category: str,
        overwrite_fields: Optional[bool] = None,
        data_identifier: Optional[str] = None,
    ) -> pd.DataFrame:
        """Execute the pipeline for *category* across all rows of *df*.

        Steps:
            1. Validate category exists in FieldManager.
            2. Validate field routing (every category field covered by exactly one step).
            3. Load checkpoint (if any) and skip already-completed steps.
            4. Execute pipeline.
            5. Write results back to DataFrame (filtering ``__`` internal fields).
            6. Clean up checkpoint on success.
        """
        if overwrite_fields is None:
            overwrite_fields = self.config.overwrite_fields

        if data_identifier is None:
            data_identifier = f"df_{hash(str(df.columns.tolist()) + str(len(df)))}"

        # 1. Validate category
        if not self.field_manager.validate_category(category):
            available = ", ".join(self.field_manager.get_categories())
            raise FieldValidationError(
                f"Category '{category}' not found. Available: {available}"
            )

        # 2. Get field specs & validate routing
        fields_dict = self.field_manager.get_category_fields(category)
        self._validate_field_routing(fields_dict)

        # 3. Checkpoint resume
        prior_step_results: dict[str, list[dict[str, Any]]] | None = None
        completed_steps: list[str] = []
        checkpoint_results: dict[str, list[dict[str, Any]]] = {}

        cp = self._checkpoint.load(data_identifier, category)
        if cp is not None:
            # Validate checkpoint compatibility
            if cp.total_rows != len(df):
                logger.warning(
                    f"Checkpoint row count mismatch ({cp.total_rows} vs {len(df)}), "
                    "starting fresh"
                )
            elif cp.fields_dict != fields_dict:
                logger.warning("Checkpoint fields_dict mismatch, starting fresh")
            else:
                prior_step_results = cp.step_results
                completed_steps = list(cp.completed_steps)
                checkpoint_results = dict(cp.step_results)
                logger.info(
                    f"Resuming from checkpoint: skipping {completed_steps}"
                )

        # 4. Convert DataFrame to rows
        rows = df.to_dict(orient="records")

        # 5. Build on_step_complete callback for checkpointing
        # Track completed steps and results across callback invocations
        cb_completed = list(completed_steps)
        cb_results = dict(checkpoint_results)

        def on_step_complete(step_name: str, step_row_results: list[dict[str, Any]]) -> None:
            cb_completed.append(step_name)
            cb_results[step_name] = step_row_results
            self._checkpoint.save_step(
                data_identifier=data_identifier,
                category=category,
                step_name=step_name,
                step_row_results=step_row_results,
                total_rows=len(rows),
                fields_dict=fields_dict,
                existing_completed=cb_completed[:-1],  # before this step
                existing_results={k: v for k, v in cb_results.items() if k != step_name},
            )

        # 6. Execute pipeline
        accumulated = await self.pipeline.execute(
            rows=rows,
            all_fields=fields_dict,
            config=self.config,
            prior_step_results=prior_step_results,
            on_step_complete=on_step_complete,
        )

        # 7. Write results back to DataFrame
        df_out = df.copy()
        for idx in range(len(df_out)):
            for key, value in accumulated[idx].items():
                # Filter __ internal fields
                if key.startswith("__"):
                    continue
                # Respect overwrite_fields
                if not overwrite_fields and key in df_out.columns:
                    existing = df_out.at[df_out.index[idx], key]
                    if pd.notna(existing) and existing != "":
                        continue
                df_out.at[df_out.index[idx], key] = value

        # 8. Cleanup checkpoint on success
        self._checkpoint.cleanup(data_identifier, category)

        logger.info(
            f"Enrichment complete: {len(df_out)} rows, category '{category}'"
        )
        return df_out

    # -- validation ------------------------------------------------------

    def _validate_field_routing(self, fields_dict: dict[str, dict[str, Any]]) -> None:
        """Ensure every category field is produced by exactly one step.

        Raises :class:`FieldValidationError` for missing or duplicate coverage.
        Warns (but does not error) if a step produces fields not in the category
        and not ``__``-prefixed.
        """
        category_fields = set(fields_dict.keys())
        step_field_map: dict[str, list[str]] = {}  # field -> list of step names

        all_step_fields: set[str] = set()
        for step_name in self.pipeline.step_names:
            step = self.pipeline.get_step(step_name)
            for field_name in step.fields:
                all_step_fields.add(field_name)
                if field_name.startswith("__"):
                    continue
                step_field_map.setdefault(field_name, []).append(step_name)

        # Missing fields: in category but not produced by any step
        missing = category_fields - all_step_fields
        if missing:
            raise FieldValidationError(
                f"Category fields not covered by any pipeline step: {sorted(missing)}"
            )

        # Duplicate fields: category field produced by >1 step
        duplicates = {
            f: steps for f, steps in step_field_map.items()
            if f in category_fields and len(steps) > 1
        }
        if duplicates:
            raise FieldValidationError(
                f"Category fields produced by multiple steps: {duplicates}"
            )

        # Extra fields: produced by steps but not in category (and not __)
        extra = set(step_field_map.keys()) - category_fields
        if extra:
            warnings.warn(
                f"Pipeline steps produce fields not in category: {sorted(extra)}",
                stacklevel=2,
            )

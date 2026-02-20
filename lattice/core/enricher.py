"""Enricher — Pipeline-based, column-oriented DataFrame enrichment.

Internal runner for Lattice. Created via ``Pipeline.runner(config)``
for repeated execution with checkpointing, or used directly for
backward compatibility with FieldManager-based workflows.
"""

import asyncio
import warnings
from typing import Any, Optional

import pandas as pd

from .checkpoint import CheckpointManager
from .config import EnrichmentConfig
from .exceptions import FieldValidationError
from ..pipeline import Pipeline
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Enricher:
    """Orchestrates column-oriented pipeline execution over a DataFrame.

    Two modes:
      1. **With FieldManager** (backward compat): ``Enricher(pipeline, field_manager=fm)``
         requires category on run().
      2. **Without FieldManager** (new API): ``Enricher(pipeline)``
         field specs come from steps.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        field_manager: Any = None,
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
        category: Optional[str] = None,
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
        category: Optional[str] = None,
        overwrite_fields: Optional[bool] = None,
        data_identifier: Optional[str] = None,
    ) -> pd.DataFrame:
        """Execute the pipeline across all rows of *df*.

        If a FieldManager is set, ``category`` is required and field routing
        is validated against the category's field definitions.

        If no FieldManager, field specs come from the steps themselves
        (inline dict fields on LLMStep).
        """
        if overwrite_fields is None:
            overwrite_fields = self.config.overwrite_fields

        if data_identifier is None:
            data_identifier = f"df_{hash(str(df.columns.tolist()) + str(len(df)))}"

        # Determine field specs
        if self.field_manager is not None:
            # Legacy path: FieldManager + category required
            if category is None:
                raise FieldValidationError(
                    "category is required when using FieldManager"
                )
            if not self.field_manager.validate_category(category):
                available = ", ".join(self.field_manager.get_categories())
                raise FieldValidationError(
                    f"Category '{category}' not found. Available: {available}"
                )
            fields_dict = self.field_manager.get_category_fields(category)
            self._validate_field_routing(fields_dict)
        else:
            # New path: field specs come from steps
            fields_dict = self.pipeline._collect_field_specs()
            category = category or "_default"

        # Checkpoint resume
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

        # Convert DataFrame to rows
        rows = df.to_dict(orient="records")

        # Build on_step_complete callback for checkpointing
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
                existing_completed=cb_completed[:-1],
                existing_results={k: v for k, v in cb_results.items() if k != step_name},
            )

        # Set up cache manager
        cache_manager = None
        if getattr(self.config, "enable_caching", False):
            from .cache import CacheManager

            cache_manager = CacheManager(
                cache_dir=getattr(self.config, "cache_dir", ".lattice"),
                ttl=getattr(self.config, "cache_ttl", 3600),
            )

        # Set up partial checkpoint callback
        checkpoint_interval = getattr(self.config, "checkpoint_interval", 0)
        on_partial_checkpoint = None
        if checkpoint_interval > 0 and self._checkpoint._enabled:
            def on_partial_checkpoint(step_name, partial_results, completed_count):
                self._checkpoint.save_step(
                    data_identifier=data_identifier,
                    category=category,
                    step_name=step_name,
                    step_row_results=partial_results,
                    total_rows=len(rows),
                    fields_dict=fields_dict,
                    existing_completed=list(cb_completed),
                    existing_results=dict(cb_results),
                )

        # Execute pipeline
        try:
            accumulated, errors, cost = await self.pipeline.execute(
                rows=rows,
                all_fields=fields_dict,
                config=self.config,
                prior_step_results=prior_step_results,
                on_step_complete=on_step_complete,
                cache_manager=cache_manager,
                on_partial_checkpoint=on_partial_checkpoint,
            )
        finally:
            if cache_manager is not None:
                cache_manager.close()

        # Log error summary if any
        if errors:
            error_summary = {}
            for err in errors:
                error_summary[err.step_name] = error_summary.get(err.step_name, 0) + 1
            logger.warning(
                "Pipeline completed with %d row errors: %s", len(errors), error_summary
            )

        # Write results back to DataFrame
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

        # Cleanup checkpoint on success
        self._checkpoint.cleanup(data_identifier, category)

        logger.info(
            f"Enrichment complete: {len(df_out)} rows"
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

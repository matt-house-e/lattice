"""Pipeline — DAG-based, column-oriented step execution engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd
from tqdm.auto import tqdm

from ..core.exceptions import PipelineError, RowError
from ..schemas.base import CostSummary, StepUsage, UsageInfo
from ..steps.base import Step, StepContext, StepResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Result from Pipeline.run() / Pipeline.run_async().

    Attributes:
        data: Enriched DataFrame with results merged in.
        cost: Aggregated token usage across all steps and rows.
        errors: Per-row errors (empty if all rows succeeded).
    """

    data: pd.DataFrame
    cost: CostSummary = field(default_factory=CostSummary)
    errors: list[RowError] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of rows that completed without error."""
        total = len(self.data)
        if total == 0:
            return 1.0
        return 1.0 - len(self.errors) / total

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


class Pipeline:
    """Orchestrates a DAG of steps with column-oriented execution.

    Construction validates: no duplicate step names, no missing
    dependencies, no cycles.  Execution runs steps level-by-level
    (topological order).  Steps within a level run in parallel;
    rows within a step run concurrently (bounded by semaphore).
    """

    def __init__(self, steps: list[Step]):
        self._steps = list(steps)
        self._step_map: dict[str, Step] = {}
        self._execution_levels: list[list[str]] = []
        self._validate_and_build()

    # -- public helpers --------------------------------------------------

    @property
    def step_names(self) -> list[str]:
        """All step names in execution order."""
        return [name for level in self._execution_levels for name in level]

    @property
    def execution_levels(self) -> list[list[str]]:
        """Topological execution levels (read-only copy)."""
        return [list(level) for level in self._execution_levels]

    def get_step(self, name: str) -> Step:
        return self._step_map[name]

    # -- primary API -----------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        config: Any = None,
    ) -> PipelineResult:
        """Synchronous entry point — the ONE way to use Lattice.

        Raises ``RuntimeError`` if called from inside a running event loop
        (use ``await pipeline.run_async(df)`` in that case).
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Pipeline.run() cannot be called from inside an async context. "
                "Use 'await pipeline.run_async(...)' instead."
            )
        except RuntimeError as exc:
            if "run_async" in str(exc):
                raise
        return asyncio.run(self.run_async(df, config))

    async def run_async(
        self,
        df: pd.DataFrame,
        config: Any = None,
    ) -> PipelineResult:
        """Async entry point."""
        from ..core.config import EnrichmentConfig

        config = config or EnrichmentConfig()

        # Collect field specs from steps
        all_fields = self._collect_field_specs()

        # Convert DataFrame to rows
        rows = df.to_dict(orient="records")

        # Execute
        accumulated, errors, cost = await self.execute(
            rows=rows,
            all_fields=all_fields,
            config=config,
        )

        # Build result DataFrame
        df_out = self._build_result_df(df, accumulated, config)

        return PipelineResult(data=df_out, cost=cost, errors=errors)

    def runner(self, config: Any = None) -> Any:
        """Power user: returns a reusable Enricher with config.

        Use for repeated execution, checkpointing, or server contexts.
        """
        from ..core.enricher import Enricher

        return Enricher(pipeline=self, config=config)

    def _collect_field_specs(self) -> dict[str, dict[str, Any]]:
        """Collect field specs from all steps.

        LLMSteps with inline field specs (FieldSpec objects) contribute their
        specs serialised as dicts.  FunctionSteps and LLMSteps with list fields
        contribute empty specs.
        """
        all_fields: dict[str, dict[str, Any]] = {}
        for step in self._steps:
            # Check if step has inline field specs (_field_specs from dict fields)
            field_specs = getattr(step, "_field_specs", {})
            for field_name in step.fields:
                if field_name.startswith("__"):
                    continue
                if field_name in field_specs:
                    spec = field_specs[field_name]
                    # FieldSpec objects → dict; plain dicts pass through
                    if hasattr(spec, "model_dump"):
                        all_fields[field_name] = spec.model_dump(exclude_none=True)
                    else:
                        all_fields[field_name] = spec
                elif field_name not in all_fields:
                    all_fields[field_name] = {}
        return all_fields

    def _build_result_df(
        self,
        df: pd.DataFrame,
        accumulated: list[dict[str, Any]],
        config: Any = None,
    ) -> pd.DataFrame:
        """Build result DataFrame from accumulated step outputs."""
        overwrite_fields = True
        if config is not None:
            overwrite_fields = getattr(config, "overwrite_fields", overwrite_fields)

        df_out = df.copy()
        for idx in range(len(df_out)):
            for key, value in accumulated[idx].items():
                if key.startswith("__"):
                    continue
                if not overwrite_fields and key in df_out.columns:
                    import pandas as _pd
                    existing = df_out.at[df_out.index[idx], key]
                    if _pd.notna(existing) and existing != "":
                        continue
                df_out.at[df_out.index[idx], key] = value
        return df_out

    # -- validation & DAG build -----------------------------------------

    def _validate_and_build(self) -> None:
        names = [s.name for s in self._steps]

        # Duplicate names
        seen: set[str] = set()
        dupes: set[str] = set()
        for n in names:
            if n in seen:
                dupes.add(n)
            seen.add(n)
        if dupes:
            raise PipelineError(f"Duplicate step names: {dupes}")

        self._step_map = {s.name: s for s in self._steps}

        # Missing dependencies
        for step in self._steps:
            for dep in step.depends_on:
                if dep not in self._step_map:
                    raise PipelineError(
                        f"Step '{step.name}' depends on unknown step '{dep}'"
                    )

        self._execution_levels = self._topological_sort()

    def _topological_sort(self) -> list[list[str]]:
        """Kahn's algorithm returning grouped execution levels."""
        in_degree: dict[str, int] = {s.name: len(s.depends_on) for s in self._steps}

        # Reverse adjacency: step -> list of steps that depend on it
        dependents: dict[str, list[str]] = {s.name: [] for s in self._steps}
        for step in self._steps:
            for dep in step.depends_on:
                dependents[dep].append(step.name)

        current_level = [name for name, deg in in_degree.items() if deg == 0]
        if not current_level:
            raise PipelineError("Cycle detected: no steps without dependencies")

        levels: list[list[str]] = []
        processed: set[str] = set()

        while current_level:
            levels.append(sorted(current_level))
            next_level: list[str] = []
            for name in current_level:
                processed.add(name)
                for dependent in dependents[name]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_level.append(dependent)
            current_level = next_level

        if len(processed) != len(self._steps):
            remaining = {s.name for s in self._steps} - processed
            raise PipelineError(f"Cycle detected involving steps: {remaining}")

        return levels

    # -- execution -------------------------------------------------------

    async def execute(
        self,
        rows: list[dict[str, Any]],
        all_fields: dict[str, dict[str, Any]],
        config: Any = None,
        prior_step_results: Optional[dict[str, list[dict[str, Any]]]] = None,
        on_step_complete: Optional[Callable[[str, list[dict[str, Any]]], None]] = None,
    ) -> tuple[list[dict[str, Any]], list[RowError], CostSummary]:
        """Execute the pipeline across all rows (column-oriented).

        Args:
            rows: Row dicts (converted from DataFrame at the Enricher boundary).
            all_fields: field_name -> field_spec dict.
            config: Optional EnrichmentConfig.
            prior_step_results: Pre-populated results for checkpoint resume.
            on_step_complete: Sync callback fired after each step completes.

        Returns:
            Tuple of (accumulated results, row errors, cost summary).
        """
        max_workers = 3
        if config is not None:
            max_workers = getattr(config, "max_workers", max_workers)

        on_error = "continue"
        if config is not None:
            on_error = getattr(config, "on_error", on_error)

        show_progress = True
        if config is not None:
            show_progress = getattr(config, "enable_progress_bar", show_progress)

        semaphore = asyncio.Semaphore(max_workers)
        num_rows = len(rows)

        # Pre-populate step_values from checkpoint data
        step_values: dict[str, list[dict[str, Any]]] = {}
        if prior_step_results:
            step_values.update(prior_step_results)

        all_errors: list[RowError] = []
        step_usage_map: dict[str, StepUsage] = {}

        total_steps = sum(len(level) for level in self._execution_levels)
        step_bar = tqdm(
            total=total_steps,
            desc="Pipeline",
            unit="step",
            disable=not show_progress,
        )

        for level in self._execution_levels:
            # Only execute steps not already in step_values (i.e. not resumed)
            steps_to_run = [name for name in level if name not in step_values]
            skipped = [name for name in level if name in step_values]

            # Advance bar for checkpointed/skipped steps
            if skipped:
                step_bar.update(len(skipped))

            if steps_to_run:
                step_bar.set_postfix(step=", ".join(steps_to_run))

                level_coros = [
                    self._execute_step(
                        self._step_map[step_name],
                        rows,
                        all_fields,
                        config,
                        step_values,
                        semaphore,
                        num_rows,
                        on_error,
                    )
                    for step_name in steps_to_run
                ]
                step_results_list = await asyncio.gather(*level_coros)

                for step_name, (step_errors, usage) in zip(steps_to_run, step_results_list):
                    all_errors.extend(step_errors)
                    if usage:
                        step_usage_map[step_name] = usage

                step_bar.update(len(steps_to_run))

                # Fire callback for each newly-executed step
                if on_step_complete is not None:
                    for step_name in steps_to_run:
                        on_step_complete(step_name, step_values[step_name])

        step_bar.close()

        # Merge all step results in execution order
        accumulated: list[dict[str, Any]] = [{} for _ in range(num_rows)]
        for level in self._execution_levels:
            for step_name in level:
                for idx in range(num_rows):
                    accumulated[idx].update(step_values[step_name][idx])

        # Build cost summary
        cost = CostSummary(
            total_prompt_tokens=sum(s.prompt_tokens for s in step_usage_map.values()),
            total_completion_tokens=sum(s.completion_tokens for s in step_usage_map.values()),
            total_tokens=sum(s.total_tokens for s in step_usage_map.values()),
            steps=step_usage_map,
        )

        return accumulated, all_errors, cost

    async def _execute_step(
        self,
        step: Step,
        rows: list[dict[str, Any]],
        all_fields: dict[str, dict[str, Any]],
        config: Any,
        step_values: dict[str, list[dict[str, Any]]],
        semaphore: asyncio.Semaphore,
        num_rows: int,
        on_error: str = "continue",
    ) -> tuple[list[RowError], StepUsage | None]:
        """Execute a single step across all rows concurrently.

        Returns tuple of (row errors, aggregated step usage).
        """
        # Slice fields for this step (internal __ fields won't be in all_fields — that's fine)
        step_fields = {f: all_fields[f] for f in step.fields if f in all_fields}

        results: list[dict[str, Any]] = [{} for _ in range(num_rows)]
        errors: list[RowError] = []
        usage_list: list[UsageInfo] = []

        async def process_row(idx: int) -> StepResult | BaseException:
            async with semaphore:
                # Gather prior results from dependency steps
                prior: dict[str, Any] = {}
                for dep_name in step.depends_on:
                    if dep_name in step_values:
                        prior.update(step_values[dep_name][idx])

                ctx = StepContext(
                    row=rows[idx],
                    fields=step_fields,
                    prior_results=prior,
                    config=config,
                )

                return await step.run(ctx)

        row_coros = [process_row(idx) for idx in range(num_rows)]
        raw_results = await asyncio.gather(*row_coros, return_exceptions=True)

        for idx, result_or_exc in enumerate(raw_results):
            if isinstance(result_or_exc, BaseException):
                row_error = RowError(
                    row_index=idx,
                    step_name=step.name,
                    error=result_or_exc,
                )
                errors.append(row_error)
                # Sentinel values for failed rows
                results[idx] = {f: None for f in step.fields}
                logger.warning(
                    "Row %d failed in step '%s': %s",
                    idx, step.name, result_or_exc,
                )
                if on_error == "raise":
                    # Store partial results before raising
                    step_values[step.name] = results
                    raise result_or_exc
            else:
                results[idx] = result_or_exc.values
                if result_or_exc.usage:
                    usage_list.append(result_or_exc.usage)

        step_values[step.name] = results

        # Aggregate usage for this step
        step_usage: StepUsage | None = None
        if usage_list:
            step_usage = StepUsage(
                prompt_tokens=sum(u.prompt_tokens for u in usage_list),
                completion_tokens=sum(u.completion_tokens for u in usage_list),
                total_tokens=sum(u.total_tokens for u in usage_list),
                rows_processed=len(usage_list),
                model=usage_list[0].model,
            )

        return errors, step_usage

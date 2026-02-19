"""Pipeline — DAG-based, column-oriented step execution engine."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from ..core.exceptions import PipelineError
from ..steps.base import Step, StepContext, StepResult


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
    ) -> list[dict[str, Any]]:
        """Execute the pipeline across all rows (column-oriented).

        Args:
            rows: Row dicts (converted from DataFrame at the Enricher boundary).
            all_fields: field_name -> field_spec dict from FieldManager.
            config: Optional EnrichmentConfig.
            prior_step_results: Pre-populated results for checkpoint resume.
                Steps in this dict are skipped; their results feed dependency routing.
            on_step_complete: Sync callback fired after each newly-executed step
                finishes for all rows.  Signature: (step_name, row_results).

        Returns:
            List of dicts, one per row, with merged outputs from all steps.
        """
        max_workers = 3
        if config is not None:
            max_workers = getattr(config, "max_workers", max_workers)

        semaphore = asyncio.Semaphore(max_workers)
        num_rows = len(rows)

        # Pre-populate step_values from checkpoint data
        step_values: dict[str, list[dict[str, Any]]] = {}
        if prior_step_results:
            step_values.update(prior_step_results)

        for level in self._execution_levels:
            # Only execute steps not already in step_values (i.e. not resumed)
            steps_to_run = [name for name in level if name not in step_values]

            if steps_to_run:
                level_coros = [
                    self._execute_step(
                        self._step_map[step_name],
                        rows,
                        all_fields,
                        config,
                        step_values,
                        semaphore,
                        num_rows,
                    )
                    for step_name in steps_to_run
                ]
                await asyncio.gather(*level_coros)

                # Fire callback for each newly-executed step
                if on_step_complete is not None:
                    for step_name in steps_to_run:
                        on_step_complete(step_name, step_values[step_name])

        # Merge all step results in execution order
        accumulated: list[dict[str, Any]] = [{} for _ in range(num_rows)]
        for level in self._execution_levels:
            for step_name in level:
                for idx in range(num_rows):
                    accumulated[idx].update(step_values[step_name][idx])

        return accumulated

    async def _execute_step(
        self,
        step: Step,
        rows: list[dict[str, Any]],
        all_fields: dict[str, dict[str, Any]],
        config: Any,
        step_values: dict[str, list[dict[str, Any]]],
        semaphore: asyncio.Semaphore,
        num_rows: int,
    ) -> None:
        """Execute a single step across all rows concurrently."""
        # Slice fields for this step (internal __ fields won't be in all_fields — that's fine)
        step_fields = {f: all_fields[f] for f in step.fields if f in all_fields}

        results: list[dict[str, Any]] = [{} for _ in range(num_rows)]

        async def process_row(idx: int) -> None:
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

                result: StepResult = await step.run(ctx)
                results[idx] = result.values

        row_coros = [process_row(idx) for idx in range(num_rows)]
        await asyncio.gather(*row_coros)

        step_values[step.name] = results

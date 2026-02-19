"""FunctionStep — wraps any sync or async callable as a pipeline step."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from .base import StepContext, StepResult


class FunctionStep:
    """Wraps a user-supplied callable as a pipeline step.

    The callable receives a ``StepContext`` and must return
    ``dict[str, Any]`` mapping field names to values.

    Sync functions are executed via ``run_in_executor`` so they
    never block the event loop.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any],
        fields: list[str],
        depends_on: list[str] | None = None,
    ):
        self.name = name
        self.fn = fn
        self.fields = fields
        self.depends_on = depends_on or []
        self._is_async = asyncio.iscoroutinefunction(fn)

    async def run(self, ctx: StepContext) -> StepResult:
        if self._is_async:
            raw = await self.fn(ctx)
        else:
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(None, self.fn, ctx)

        # Filter to declared fields only
        values = {k: v for k, v in raw.items() if k in self.fields}
        return StepResult(values=values)

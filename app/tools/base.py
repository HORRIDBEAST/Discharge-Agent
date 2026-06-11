"""
Base tool interface.

All agent tools inherit from BaseTool. This provides:
  - Consistent retry behaviour
  - Uniform error handling (never silent failure)
  - Execution timing for observability
  - Tool registration for dynamic dispatch
"""
from __future__ import annotations
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global tool registry — populated by @register_tool decorator
_TOOL_REGISTRY: dict[str, type["BaseTool"]] = {}


def register_tool(cls: type["BaseTool"]) -> type["BaseTool"]:
    """Class decorator to register a tool for dynamic dispatch."""
    _TOOL_REGISTRY[cls.name] = cls
    return cls


def get_tool_registry() -> dict[str, type["BaseTool"]]:
    return dict(_TOOL_REGISTRY)


class ToolResult:
    """Standardised tool result wrapper."""
    def __init__(
        self,
        tool_name: str,
        success: bool,
        data: Any = None,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
        retry_count: int = 0,
    ):
        self.tool_name = tool_name
        self.success = success
        self.data = data
        self.error = error
        self.duration_ms = duration_ms
        self.retry_count = retry_count

    def __repr__(self) -> str:
        status = "OK" if self.success else f"FAILED({self.error})"
        return f"ToolResult({self.tool_name}, {status}, {self.duration_ms:.0f}ms)"


class BaseTool(ABC):
    name: str = "base_tool"
    description: str = "Base tool"
    max_retries: int = 3
    retry_delay: float = 1.0

    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """Tool-specific logic. Raise exceptions on failure."""
        ...

    async def run(self, **kwargs) -> ToolResult:
        """
        Execute the tool with retry logic.
        NEVER raises — always returns a ToolResult with success=True/False.
        This is the single interface the agent uses.
        """
        last_error: Optional[str] = None
        start = time.monotonic()

        for attempt in range(self.max_retries):
            try:
                data = await self._execute(**kwargs)
                duration = (time.monotonic() - start) * 1000
                logger.info(
                    "tool_success",
                    tool=self.name,
                    attempt=attempt,
                    duration_ms=round(duration, 1),
                )
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    data=data,
                    duration_ms=duration,
                    retry_count=attempt,
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "tool_attempt_failed",
                    tool=self.name,
                    attempt=attempt,
                    error=last_error,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        duration = (time.monotonic() - start) * 1000
        logger.error(
            "tool_all_retries_failed",
            tool=self.name,
            retries=self.max_retries,
            error=last_error,
        )
        return ToolResult(
            tool_name=self.name,
            success=False,
            error=last_error,
            duration_ms=duration,
            retry_count=self.max_retries,
        )

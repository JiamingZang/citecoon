"""Tool assembly：MCP 纯工具面的全部工具注册。"""

from __future__ import annotations

from ..context import RunContext
from .batch_read import build_batch_read_tools
from .cards import build_card_tools
from .codebase import build_codebase_tools
from .config import build_config_tools
from .emit_result import build_emit_result_tool
from .graph import build_graph_tools
from .health import build_health_tools
from .notes import build_note_tools
from .read import build_read_tools
from .report import build_report_tools
from .roadmap import build_roadmap_tools
from .seeds import build_seed_tools
from .writing import build_writing_tools

__all__ = ["build_tools", "build_emit_result_tool"]


def build_tools(ctx: RunContext) -> list:
    """All non-finish tools available for one run."""
    tools: list = []
    tools.extend(build_seed_tools(ctx))
    tools.extend(build_graph_tools(ctx))
    tools.extend(build_read_tools(ctx))
    tools.extend(build_config_tools(ctx))
    tools.extend(build_note_tools(ctx))
    tools.extend(build_card_tools(ctx))
    tools.extend(build_batch_read_tools(ctx))
    tools.extend(build_codebase_tools(ctx))
    tools.extend(build_roadmap_tools(ctx))
    tools.extend(build_writing_tools(ctx))
    tools.extend(build_report_tools(ctx))
    tools.extend(build_health_tools(ctx))
    return tools

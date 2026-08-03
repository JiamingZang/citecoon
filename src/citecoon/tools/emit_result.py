"""收口工具：组装 PipelineResult 并落盘产物（result.json + graphml + view.html）。"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from stirrup import Tool, ToolResult, ToolUseCountMetadata
from citecoon.core.models import PipelineResult

from ..context import RunContext


class EmitParams(BaseModel):
    reason: str = Field(
        description="Short explanation of why the research task is complete (or why it cannot proceed further)."
    )


def _ensure_graph(ctx: RunContext):
    """纯确定性图物化：to_graph → 去重 → 年份自愈 → 指标（PageRank/velocity）。
    带指纹缓存：MCP 长驻会话里图持续生长，不能把早期物化的旧图当成品落盘。"""
    from ..core.graphprep import clean_years, compute_metrics, dedupe_nodes

    fp = (
        len(ctx.ws.papers),
        len(ctx.ws.seeds),
        sum(len(p.referenced_works) for p in ctx.ws.papers.values()),
    )
    if ctx.graph is not None and getattr(ctx, "_graph_fp", None) == fp:
        return ctx.graph
    ctx.graph = None
    ctx._graph_fp = fp
    g = ctx.ws.to_graph()
    dedupe_nodes(g)
    clean_years(g)
    compute_metrics(g)
    ctx.graph = g
    return g


def build_emit_result_tool(ctx: RunContext) -> Tool:
    def emit_result(p: EmitParams) -> ToolResult[ToolUseCountMetadata]:
        graph = _ensure_graph(ctx)
        if ctx.founding:
            fset = set(ctx.founding)
            for node in graph.nodes:
                if node.paper_id in fset:
                    node.role = "founding"
        ctx.ws.trace.add("agent", "finish", p.reason)

        result = PipelineResult(
            query=ctx.ws.query,
            seeds=list(ctx.ws.seeds),
            graph=graph,
            founding=list(ctx.founding),
            roadmap=ctx.roadmap,
            report=ctx.report,
            trace=ctx.ws.trace.dump(),
        )

        ctx.result = result
        gdir = ctx.graph_dir()
        out_path = gdir / "result.json"
        out_path.write_text(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        ctx.result_path = str(out_path)

        # 顺手做确定性导出（graphml/view.html/trace.log，零 LLM）。
        # 逐项各自兜底：report 为空（MCP 场景报告由外层 agent 自己写）时跳过
        # report.md，不能让它连累后面的 view.html
        from citecoon.render import export as _ex

        for job in (
            lambda: _ex.to_graphml(result, str(gdir / "citation_network.graphml")),
            lambda: _ex.roadmap_to_graphml(result, str(gdir / "roadmap.graphml")),
            lambda: any(result.report.model_dump(exclude={"query"}).values()) and _ex.to_markdown(result, str(ctx.project_dir("reports").parent / "report.md")),
            lambda: result.trace and _ex.to_trace_log(result, str(ctx.runtime_dir() / "trace.log")),
        ):
            try:
                job()
            except Exception:  # noqa: BLE001 — 单项失败不影响其余交付
                pass
        try:
            from citecoon.render import viz
            viz.to_html(result, str(gdir / "view.html"))
        except Exception:  # noqa: BLE001
            pass

        n, m = len(graph.nodes), len(graph.edges)
        msg = (
            f"result.json written to {out_path} ({n} nodes, {m} edges; "
            f"founding={len(ctx.founding)}, roadmap={len(ctx.roadmap.nodes)} nodes). "
            f"Reason: {p.reason}"
        )
        return ToolResult(content=msg, metadata=ToolUseCountMetadata(), success=True)

    return Tool[EmitParams, ToolUseCountMetadata](
        name="emit_result",
        description=(
            "Finish the task: assemble the citation network + findings into result.json "
            "(the deliverable) and end the run. Call this ONLY when you have at least a "
            "seed-anchored citation graph, and any founding/roadmap/report you intend to produce."
        ),
        parameters=EmitParams,
        executor=emit_result,
    )

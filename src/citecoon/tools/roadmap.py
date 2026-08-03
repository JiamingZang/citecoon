"""演化路线图装配工具（零 LLM）：角色/边的判断归外层 agent，这里只做校验与组装。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from stirrup import Tool, ToolResult, ToolUseCountMetadata

from ..context import RunContext
from ..core.models import Roadmap, RoadmapEdge, RoadmapNode


def _ok(content: str) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content=content, metadata=ToolUseCountMetadata(), success=True)


def build_roadmap_tools(ctx: RunContext) -> list[Tool]:
    class RoadmapNodeIn(BaseModel):
        paper_id: str = Field(description="图谱内论文 id（支持别名，会解析到正本）。")
        role: str = Field(description="演化角色：founding（奠基）/ breakthrough（范式突破）/ improvement（关键改进）/ branch（分支开创）/ survey（集大成综述）/ normal。")
        stage: str | None = Field(default=None, description="所属发展阶段名（与 fill_domain_report 的 stages 对应，可省）。")
        contribution: str | None = Field(default=None, description="一句话贡献（中文，可省）。")

    class RoadmapEdgeIn(BaseModel):
        source: str = Field(description="在前/施影响的论文 id。")
        target: str = Field(description="在后/受影响的论文 id。")
        relation: str = Field(default="leads_to", description="关系：leads_to / inspires / improves / contrasts。")

    class FillRoadmapParams(BaseModel):
        nodes: list[RoadmapNodeIn] = Field(description="演化路线图关键节点（5-15 篇为宜，角色判断归你）。")
        edges: list[RoadmapEdgeIn] = Field(description="演化边（谁启发/改进了谁），端点必须都在 nodes 里。")

    def fill_roadmap(p: FillRoadmapParams) -> ToolResult[ToolUseCountMetadata]:
        ws = ctx.ws
        roles = {"founding", "breakthrough", "improvement", "branch", "survey", "normal"}
        bad_roles = sorted({n.role for n in p.nodes} - roles)
        if bad_roles:
            return _ok(f"fill_roadmap 拒绝：未知角色 {bad_roles}——可用：{sorted(roles)}。")
        rid = {n.paper_id: ws.resolve_id(n.paper_id) for n in p.nodes}
        unknown = sorted(k for k, v in rid.items() if v is None)
        if unknown:
            return _ok(
                "fill_roadmap 拒绝：以下 paper_id 不在图谱中——" + ", ".join(unknown[:8])
                + "。用 graph_summary/find_in_graph 核对后重试。"
            )
        node_ids = set(rid.values())
        loose = [f"{e.source}→{e.target}" for e in p.edges
                 if (ws.resolve_id(e.source) not in node_ids or ws.resolve_id(e.target) not in node_ids)]
        if loose:
            return _ok("fill_roadmap 拒绝：边的端点不在 nodes 列表里——" + "; ".join(loose[:6]))
        ctx.roadmap = Roadmap(
            nodes=[
                RoadmapNode(
                    paper_id=rid[n.paper_id],
                    title=ws.papers[rid[n.paper_id]].title,
                    year=ws.papers[rid[n.paper_id]].year,
                    role=n.role,
                    stage=n.stage,
                    contribution=n.contribution,
                )
                for n in p.nodes
            ],
            edges=[
                RoadmapEdge(source=ws.resolve_id(e.source), target=ws.resolve_id(e.target), relation=e.relation)
                for e in p.edges
            ],
        )
        ctx.ws.trace.add("agent", "roadmap", f"fill_roadmap: {len(p.nodes)} nodes / {len(p.edges)} edges")
        return _ok(
            f"演化路线图已装配：{len(p.nodes)} 节点 / {len(p.edges)} 边。"
            f"调 emit_result 落盘——roadmap.graphml 与 result.json 会一起更新。"
        )

    return [
        Tool[FillRoadmapParams, ToolUseCountMetadata](
            name="fill_roadmap",
            description=(
                "装配演化路线图（零 LLM）：你（调用方）判断关键节点的演化角色（奠基/突破/改进/分支/综述）"
                "与演化边，本工具校验 id 在图并组装 DAG。通常在 fill_domain_report 之后、emit_result 之前调，"
                "roadmap.graphml 会随落盘；不调则 roadmap 产物为空。"
            ),
            parameters=FillRoadmapParams,
            executor=fill_roadmap,
        ),
    ]

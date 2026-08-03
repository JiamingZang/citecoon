"""项目加载与零 LLM 运维操作。

MCP server 与 CLI 共用 `load_project`（读 result.json / 边车快照，回灌图谱、
种子与既有分析成果）；`op_pdf` 供 `citecoon pdf` 补导报告 PDF。

约定：project 目录须含 result.json（emit_result 产出，在 01_graph/）；
缺失时回落 _runtime/result.autosave.json 边车快照（兼容旧平铺布局）。
"""

from __future__ import annotations

import json
from pathlib import Path

from .context import RunContext, build_context
from .core.layout import dir_for, find_result
from .core.models import Paper


def load_project(project: str, cache_path: str | None = None) -> RunContext:
    """从项目目录重建 RunContext：读图谱拿 query，把论文回灌进 Workspace。

    result.json（emit 精装版）与 autosave（每次图变更的边车快照）都读，
    取论文数更大的——只信 result.json 会静默丢掉上一会话扩大的图
    （实测：152 篇会话图被 96 篇旧 emit 盖掉，"缺失才回落"分支永远走不到）。
    """
    root = Path(project)
    prev: dict = {}
    best_n = -1
    for rp in (
        root / "01_graph" / "result.json",
        root / "result.json",
        root / "_runtime" / "result.autosave.json",
        root / "result.autosave.json",
    ):
        if not rp.is_file():
            continue
        try:
            data = json.loads(rp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        n = len((data.get("graph") or {}).get("papers") or {})
        if n > best_n:
            best_n, prev = n, data
    graph = prev.get("graph") or {}
    query = graph.get("query") or root.name
    ctx = build_context(query, out_dir=root, cache_path=cache_path)
    for pd in (graph.get("papers") or {}).values():
        try:
            ctx.ws.add_paper(Paper.model_validate(pd), 1)
        except Exception:  # noqa: BLE001 — 单篇坏数据不挡整体回灌
            continue
    # 种子身份一并恢复：seeds 是相关性锚点的兜底
    for sid in graph.get("seeds") or []:
        if sid in ctx.ws.papers and sid not in ctx.ws.seeds:
            ctx.ws.seeds.append(sid)
    # 分析成果同样回灌：founding/roadmap/report 不恢复的话，新会话 emit 会用
    # 空值覆盖 result.json，把前一个会话装配的领域报告静默抹掉（实测事故）
    if prev:
        from .core.models import FieldReport, Roadmap

        try:
            ctx.founding = [f for f in (prev.get("founding") or []) if f in ctx.ws.papers]
            if prev.get("roadmap"):
                ctx.roadmap = Roadmap.model_validate(prev["roadmap"])
            if prev.get("report"):
                ctx.report = FieldReport.model_validate(prev["report"])
        except Exception:  # noqa: BLE001 — 旧文件结构不兼容时不挡加载，等重新装配
            pass
    return ctx


def op_pdf(project: str, report: str) -> str:
    """存量报告补导 PDF（无 LLM，本地渲染）。"""
    from .render.pdf import report_md_to_pdf
    from .tools.report import find_report

    reports = dir_for(Path(project), "reports")
    md, err = find_report(reports, report)
    if err:
        return err
    pdf = report_md_to_pdf(md)
    return f"PDF 已导出：{pdf}（{pdf.stat().st_size // 1024} KB）"

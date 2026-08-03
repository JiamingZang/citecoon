"""Deep-read tools: let the agent read a paper's abstract + full text, and connect
citation-disconnected frontier papers to the lineage via full-text understanding.

These ground the agent's judgments (founding / roles / relevance) in what papers
actually say, not just their metadata — and expose the project's headline capability
(frontier full-text link-back) as an agent-driven tool.
"""

from __future__ import annotations

import asyncio
import os

from pydantic import BaseModel, Field

from stirrup import Tool, ToolResult, ToolUseCountMetadata

from ..context import RunContext


def _ok(content: str) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content=content, metadata=ToolUseCountMetadata(), success=True)


def _parse_local_pdf(path: str, max_chars: int = 60000) -> str:
    """Extract text from a local PDF using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    parts = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(parts)[:max_chars].strip()


def _render_sections(sections: dict, full: str, section_filter: str | None, parts: list[str]) -> None:
    """Append section content to parts list based on filter."""
    if sections:
        if section_filter:
            body = sections.get(section_filter) or sections.get(section_filter.replace(" ", "_"))
            if body:
                parts.append(f"[{section_filter.replace('_', ' ').upper()}]\n{body[:18000]}")
            else:
                available = ", ".join(sections.keys())
                parts.append(f"[section '{section_filter}' not found; available: {available}]")
        else:
            order = ["abstract", "introduction", "related_work", "method", "experiments", "conclusion"]
            for name in order:
                body = sections.get(name)
                if body:
                    parts.append(f"[{name.replace('_', ' ').upper()}]\n{body[:9000]}")
    elif full:
        if section_filter:
            parts.append(f"[sections not parseable; returning full text]\n{full[:36000]}")
        else:
            parts.append(f"FULL TEXT:\n{full[:36000]}")


class PaperIdParams(BaseModel):
    paper_id: str = Field(description="A paper id shown in the graph/summary (e.g. 'W2626778328' or 'arxiv:2312.00752').")
    section: str | None = Field(
        default=None,
        description="Optional: request a single section for deep reading (e.g. 'method', 'introduction', "
        "'experiments', 'conclusion', 'abstract', 'related_work'). Omit to get all sections.",
    )
    deep: bool = Field(
        default=False,
        description="本会话全文精读超额（>8 篇）后默认只回摘要+落盘路径；用户点名深读这一篇时传 true 继续全文。",
    )


class LocalPdfParams(BaseModel):
    path: str = Field(description="Absolute or relative path to a local PDF file.")
    section: str | None = Field(
        default=None,
        description="Optional: request a single section (e.g. 'method', 'introduction'). Omit to get all sections.",
    )


class WirePredecessorsParams(BaseModel):
    paper_id: str = Field(description="图内论文 id（即将为它接上前驱引用边）")
    predecessor_titles: list[str] = Field(
        description="该论文所基于的前驱工作标题列表（你自己从 read_paper 全文的 related-work/references 里抽，一次传全部，≤12 个）"
    )


def build_read_tools(ctx: RunContext) -> list[Tool]:
    ws = ctx.ws
    session_reads = {"n": 0}  # 本会话全文精读计数：批量分流提醒的触发器

    async def _resolve(pid: str):
        cid = ws.resolve_id(pid)  # 含别名（emit 合并后旧 id 仍可用）
        paper = (ws.papers.get(cid) if cid else None) or getattr(ctx, "candidate_pool", {}).get(pid)
        if paper is None:
            paper = await ctx.source.get_paper(pid)
        if paper is None and "arxiv" in pid.lower():
            # 同 add_seed 的后备：超新论文 OpenAlex 无索引，id 直查 arXiv
            paper = await ctx.arxiv.get_by_id(pid)
        return paper

    async def read_paper(p: PaperIdParams) -> ToolResult[ToolUseCountMetadata]:
        paper = await _resolve(p.paper_id)
        if paper is None:
            return _ok(f"'{p.paper_id}' not found. Use an id from the graph summary or find_candidates.")
        head = (
            f"{paper.title} ({paper.year})\n"
            f"id={paper.id} | citations={paper.citation_count}"
            f"{' | venue=' + paper.venue if paper.venue else ''}\n"
        )
        parts = [head]
        aid = (paper.source_ids or {}).get("arxiv")
        sections: dict = {}
        if aid and ctx.arxiv is not None:
            try:
                data = await ctx.arxiv.fulltext_sections(aid)
                sections = data.get("sections") or {}
                full = data.get("full_text") or ""
            except Exception as e:  # noqa: BLE001
                full = ""
                pdf_url = f"https://arxiv.org/pdf/{aid.split(':')[-1]}"
                parts.append(
                    f"[PDF 下载失败: {e}]\n"
                    f"论文 PDF 链接: {pdf_url}\n"
                    f"请直接询问用户是否能帮忙下载此 PDF 到本地，"
                    f"然后用 read_local_pdf 工具读取本地文件路径。"
                )
            else:
                over_budget = session_reads["n"] >= 8 and not p.deep and not p.section
                if full:
                    # 顺手把原文落 papers/（与 batch_read 同一约定）：读过的论文
                    # 不留附件，下次复读/写 idea 还得重新拉网（实测 MCP 会话读了
                    # 9 篇原文但 papers/ 为空）
                    from .batch_read import _persist_fulltext, _title_key
                    _persist_fulltext(ctx, paper, full[:40000].strip(), "arxiv")
                if over_budget and (sections or full):
                    # 超额闸门：提醒无效（实测"全部精读"40+ 篇时 agent 无视 in-band
                    # 提醒继续逐篇灌主上下文）——超 8 篇后全文不再进上下文，只回
                    # 摘要+落盘路径；取文/存档照常干，深读单篇传 deep=true
                    session_reads["n"] += 1
                    brief = (sections.get("abstract") or paper.abstract or full[:1200] or "")[:1500]
                    from .batch_read import _title_key as _tk
                    return _ok(
                        head
                        + f"[本会话已全文精读 {session_reads['n']} 篇，超出主上下文预算——本篇只回摘要，"
                        f"全文已存 papers/{_tk(paper.title)}.md]\n"
                        f"ABSTRACT:\n{brief}\n\n"
                        f"[批量落卡走分工模式：起你的子 agent 读盘出卡片要点，你校对后 fill_idea_card；"
                        f"待办用 list_papers(only_uncarded=true)。确需在主上下文深读这一篇：read_paper(deep=true) 或指定 section]"
                    )
                _render_sections(sections, full, p.section, parts)
                if full:
                    from .batch_read import _title_key
                    parts.append(
                        f"[原文已存 papers/{_title_key(paper.title)}.md——批量落卡时可让你的子 agent 直接读盘，不必占用主上下文重复 read_paper]"
                    )
        # abstract fallback (OpenAlex) when no PDF sections were parsed
        if "abstract" not in sections and paper.abstract:
            parts.insert(1, f"ABSTRACT:\n{paper.abstract}")
        if sections or (len(parts) > 1 and "ABSTRACT" not in parts[-1]):
            session_reads["n"] += 1
            if session_reads["n"] > 8:
                # 分流提醒放回显尾部（in-band 反复出现）：开场纪律在长会话里会被
                # 遗忘——实测"全部精读"40+ 篇时 agent 逐篇灌主上下文直到见底
                parts.append(
                    f"[提醒：本会话已全文精读 {session_reads['n']} 篇。继续批量落卡请改分工模式——"
                    f"原文都在 papers/ 目录，起你的子 agent 分批读盘出卡片要点，你只校对并 fill_idea_card；"
                    f"待办清单用 list_papers(only_uncarded=true) 看]"
                )
        if len(parts) == 1:
            if aid:
                pdf_url = f"https://arxiv.org/pdf/{aid.split(':')[-1]}"
                parts.append(
                    f"[无法获取全文]\n"
                    f"论文 PDF 链接: {pdf_url}\n"
                    f"请直接询问用户是否能帮忙下载此 PDF 到本地，"
                    f"然后用 read_local_pdf 工具读取。"
                )
            else:
                parts.append("[no abstract or arXiv full text available for this paper]")
        ws.trace.add("agent", "read", f"{paper.id} | {paper.title[:60]}")
        return _ok("\n\n".join(parts))

    async def read_local_pdf(p: LocalPdfParams) -> ToolResult[ToolUseCountMetadata]:
        path = os.path.expanduser(p.path)
        if not os.path.isfile(path):
            return _ok(f"文件不存在: {path}")
        try:
            from ..sources.arxiv import split_sections
            text = await asyncio.to_thread(_parse_local_pdf, path)
        except Exception as e:  # noqa: BLE001
            return _ok(f"PDF 解析失败: {e}")
        if not text:
            return _ok(f"PDF 解析结果为空: {path}")
        sections = split_sections(text)
        parts = [f"[LOCAL PDF] {os.path.basename(path)}\n"]
        _render_sections(sections, text, p.section, parts)
        if len(parts) == 1:
            parts.append(f"FULL TEXT:\n{text[:36000]}")
        return _ok("\n\n".join(parts))

    async def wire_predecessors(p: WirePredecessorsParams) -> ToolResult[ToolUseCountMetadata]:
        # link_frontier 的无 LLM 版：标题抽取交给调用方（MCP 纯工具面里就是外层 agent 自己读全文抽），
        # 这里只做确定性的标题→id 解析与 referenced_works 接线
        paper = ws.papers.get(ws.resolve_id(p.paper_id) or p.paper_id)
        if paper is None:
            return _ok(f"'{p.paper_id}' 不在图内——先 find_candidates + add_seed。")
        titles = [t.strip() for t in p.predecessor_titles if t and t.strip()][:12]
        if not titles:
            return _ok("predecessor_titles 为空。从 read_paper 的 related-work/references 里抽标题后再调。")
        from citecoon.sources.resolve import resolve_titles

        found, ref_ids, misses = await resolve_titles(
            titles, ctx.source, ctx.arxiv, multi=ctx.multi,
            local=ws.papers,  # 前驱常已在图内（互为前驱），本地命中零请求且不受限流影响
        )
        if ref_ids:
            paper.referenced_works = list(dict.fromkeys(list(paper.referenced_works) + ref_ids))
        depth = ws.depth.get(paper.id, 0) + 1
        added = sum(ws.add_paper(pp, depth) for pp in found.values())
        ws.trace.add("agent", "wire_predecessors", f"{paper.id}: +{len(ref_ids)} refs, +{added} new papers")
        lines = [f"wire_predecessors: {paper.id} 接上 {len(ref_ids)}/{len(titles)} 个前驱（新入图 {added} 篇；边在下次建图时生效）。"]
        for pid, pp in found.items():
            lines.append(f"  {pid} | {pp.year} | {pp.title[:70]}")
        if misses:
            lines.append("未解析成功（可能未索引/标题不准，可改短标题重试或放弃）：" + "; ".join(m[:60] for m in misses))
        return _ok("\n".join(lines))

    return [
        Tool[PaperIdParams, ToolUseCountMetadata](
            name="read_paper",
            description=(
                "Read a paper's abstract and (if on arXiv) its full text. Use this to GROUND "
                "your judgments — before confirming the founding paper, assigning roadmap roles, "
                "or writing the report — instead of guessing from title + citation count. "
                "For deep reading, pass section='method' (or introduction/experiments/conclusion) "
                "to get a single section with more text (18k chars vs 9k)."
            ),
            parameters=PaperIdParams,
            executor=read_paper,
        ),
        Tool[LocalPdfParams, ToolUseCountMetadata](
            name="read_local_pdf",
            description=(
                "Read a local PDF file for deep reading. Use when: (1) the user provides a local "
                "file path, or (2) arXiv PDF download failed and the user downloaded the file manually. "
                "Supports the same section parameter as read_paper."
            ),
            parameters=LocalPdfParams,
            executor=read_local_pdf,
        ),
        Tool[WirePredecessorsParams, ToolUseCountMetadata](
            name="wire_predecessors",
            description=(
                "把你自己从论文全文里抽出的前驱工作标题，解析成结构化 id 并接成引用边（无 LLM，"
                "纯确定性）。适用：断连论文的谱系接线——先 read_paper 读它的 related-work/references，"
                "你抽出 5-12 个前驱标题，一次性传进来；未解析成功的标题会列出，可改短重试。"
            ),
            parameters=WirePredecessorsParams,
            executor=wire_predecessors,
        ),
    ]

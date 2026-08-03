"""原文库工具：read_fulltext 从项目 papers/（或全局库）读已落盘全文。
全文获取链（arXiv → OA PDF）在 _fulltext，供 read_paper 复用；落盘约定 _persist_fulltext。
"""

from __future__ import annotations

import asyncio
import json
import os

from pydantic import BaseModel, Field

from stirrup import Tool, ToolResult, ToolUseCountMetadata

from ..context import RunContext
from ..core.utils import extract_json
from .cards import CARDS_DIR, _substance, _title_key

PAPERS_DIR = "papers"

# concurrent sub-calls; keep modest — each spawns a qodercli subprocess and the
# gateway throttles bursts (TOO_MANY_REQUESTS shows up well below 10 rps)
_CONCURRENCY = 3



async def _fetch_pdf_and_parse(ctx: RunContext, paper, url: str) -> str | None:
    import fitz
    import httpx

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
    # keep the original PDF as the source of truth (figures/formulas intact)
    try:
        pdf_path = ctx.project_dir(PAPERS_DIR) / f"{_title_key(paper.title)}.pdf"
        pdf_path.write_bytes(r.content)
    except OSError:
        pass

    def parse(data: bytes) -> str:
        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text

    text = (await asyncio.to_thread(parse, r.content))[:40000].strip()
    if text:
        _persist_fulltext(ctx, paper, text, "pdf_url")
    return text or None




def _persist_fulltext(ctx: RunContext, paper, text: str, source: str) -> None:
    """Keep the raw text — cards are lossy distillates; plans/reports need to
    consult original wording, derivations and eval numbers later."""
    try:
        key = _title_key(paper.title)
        path = ctx.project_dir(PAPERS_DIR) / f"{key}.md"
        if path.exists():
            return
        aid = (paper.source_ids or {}).get("arxiv")
        header = (
            f"# {paper.title}\n\n"
            f"> {paper.year} · id: {paper.id}"
            + (f" · arXiv: {aid}" if aid else "")
            + (f" · pdf: {paper.pdf_url}" if paper.pdf_url else "")
            + f" · 来源: {source}\n"
            f"> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。\n\n"
        )
        path.write_text(header + text, encoding="utf-8")
    except OSError:
        pass


async def _fulltext(ctx: RunContext, paper) -> str | None:
    """Three-tier full-text fetch: known arXiv id → title-resolved arXiv id → raw pdf_url.

    Graph nodes sourced from OpenAlex search often lack an arXiv id even when the
    paper IS on arXiv — tier 2 resolves it by exact-title match so those papers
    stop silently falling out of the deep-reading pipeline."""
    aid = (paper.source_ids or {}).get("arxiv")
    if not aid:
        # tier 2: resolve arXiv id by title (strip colons — they break arXiv query syntax)
        try:
            q = paper.title.replace(":", " ").strip()
            hits = await ctx.arxiv.search(q, max_results=3, sort="relevance")
            for h in hits:
                if _title_key(h.title) == _title_key(paper.title) and (h.source_ids or {}).get("arxiv"):
                    aid = h.source_ids["arxiv"]
                    paper.source_ids = {**(paper.source_ids or {}), "arxiv": aid}
                    break
        except Exception:  # noqa: BLE001
            pass
    if aid:
        try:
            data = await ctx.arxiv.fulltext_sections(aid)
        except Exception:  # noqa: BLE001
            data = None
        if data:
            sections = data.get("sections") or {}
            if sections:
                order = ["abstract", "introduction", "method", "experiments", "related_work", "conclusion"]
                text = "\n\n".join(f"## {k}\n{sections[k][:9000]}" for k in order if sections.get(k))
                if text:
                    _persist_fulltext(ctx, paper, text, "arxiv")
                    return text
            full = (data.get("full_text") or "")[:40000]
            if full:
                _persist_fulltext(ctx, paper, full, "arxiv-full")
                return full
    # tier 3: open-access PDF URL from metadata
    if paper.pdf_url:
        try:
            text = await _fetch_pdf_and_parse(ctx, paper, paper.pdf_url)
            if text:
                return text
        except Exception:  # noqa: BLE001
            pass
    return None


class ReadFulltextParams(BaseModel):
    title_query: str = Field(description="论文标题关键词（子串匹配本地原文库）。")
    section: str | None = Field(
        default=None,
        description="可选：只要某个章节（abstract/introduction/method/experiments/related_work/conclusion）。",
    )


def build_batch_read_tools(ctx: RunContext) -> list[Tool]:
    ws = ctx.ws

    def read_fulltext(p: ReadFulltextParams) -> ToolResult[ToolUseCountMetadata]:
        d = ctx.project_dir(PAPERS_DIR)
        q = _title_key(p.title_query)
        hits = [f for f in sorted(d.glob("*.md")) if q in f.stem]
        if not hits:
            names = [f.stem[:50] for f in sorted(d.glob("*.md"))]
            return ToolResult(
                content=f"本地原文库没有匹配 '{p.title_query}' 的论文。现有 {len(names)} 篇：{names[:20]}\n"
                f"（原文在 read_paper 读取时自动落库；没落库的先 read_paper 获取。）",
                metadata=ToolUseCountMetadata(), success=True,
            )
        text = hits[0].read_text(encoding="utf-8")
        if p.section:
            import re as _re
            m = _re.search(rf"## {p.section}\n(.*?)(?=\n## |\Z)", text, _re.DOTALL)
            body = m.group(1).strip() if m else f"（未找到章节 {p.section}，返回开头）\n{text[:6000]}"
            return ToolResult(content=f"{hits[0].stem}\n\n{body[:20000]}", metadata=ToolUseCountMetadata(), success=True)
        pdf = hits[0].with_suffix(".pdf")
        tail = f"\n\n（原始 PDF: {pdf}）" if pdf.exists() else ""
        return ToolResult(content=text[:30000] + tail, metadata=ToolUseCountMetadata(), success=True)

    return [
        Tool[ReadFulltextParams, ToolUseCountMetadata](
            name="read_fulltext",
            description=(
                "从本地原文库（papers/ 目录，批读时自动落盘的全文）按标题读原始内容。"
                "写方案/报告/草稿时需要核对原文推导、实验数字、原话措辞时用这个——卡片是有损压缩，"
                "关键细节必须回原文。本地无网络零成本；没落库的论文用 read_paper 拉取。"
            ),
            parameters=ReadFulltextParams,
            executor=read_fulltext,
        ),
    ]

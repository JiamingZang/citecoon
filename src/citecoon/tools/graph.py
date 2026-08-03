"""Graph-building tools: resolve seeds, search, expand, inspect.

Thin Stirrup wrappers over the superacademic engine + a shared Workspace. Mirrors
the logic of superacademic.agents.tools.scout_tools, but as typed Stirrup Tools so
the agent-loop drives exploration and decides when it has enough.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from stirrup import Tool, ToolResult, ToolUseCountMetadata
from stirrup.core.models import EmptyParams

from ..context import RunContext


def _ok(content: str) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content=content, metadata=ToolUseCountMetadata(), success=True)


class SearchParams(BaseModel):
    query: str = Field(description="Free-text search string to find more papers in this field.")


class RecentSearchParams(BaseModel):
    query: str = Field(
        description="Free-text search string. Use a SHORT tight phrase — a specific named "
        "sub-technique/acronym when you're chasing a hot narrow wave, or the general topic otherwise."
    )
    sort: str = Field(
        default="submittedDate",
        description="arXiv sort order: 'submittedDate' surfaces the newest follow-ups (best for "
        "catching this week's/month's papers); 'relevance' surfaces the paper most central to the "
        "phrase, which may be slightly older (often the one that started a sub-wave). If a narrow "
        "phrase matters, call search_recent twice — once with each sort — to get both angles.",
    )


class ExpandParams(BaseModel):
    paper_id: str = Field(description="A paper id shown in the graph summary (e.g. 'W2626778328').")


class FindInGraphParams(BaseModel):
    query: str = Field(
        description="标题关键词、方法名缩写（如 'MegaPose'）或 arxiv id 片段，大小写不敏感。"
    )
    newest_first: bool = Field(
        default=False,
        description="默认按引用数排序；找近年新工作（低引/0 引）时传 true 按年份降序——"
        "引用序会把它们截出视野。",
    )


def build_graph_tools(ctx: RunContext) -> list[Tool]:
    ws = ctx.ws
    s = ctx.settings
    frontier_done: set[str] = set()  # paper_ids already frontier-expanded this run

    async def graph_search(p: SearchParams) -> ToolResult[ToolUseCountMetadata]:
        hits = await ctx.source.search(p.query, limit=8)
        added = sum(ws.add_paper(h, 1) for h in hits)
        return _ok(f"search('{p.query}'): {len(hits)} hits, {added} new added.\n" + ws.summary())

    class ListPapersParams(BaseModel):
        only_uncarded: bool = Field(default=False, description="只列还没有论文卡的（批量落卡时用）。")

    async def list_papers(p: ListPapersParams) -> ToolResult[ToolUseCountMetadata]:
        """全图论文清单 + 落卡/原文状态（零 LLM）。graph_summary 只显 top 8，
        实测 agent 为枚举全图被迫用 find_in_graph 子串翻页 hack，浪费好几轮。"""
        from .cards import _title_key, load_cards

        carded = {_title_key(c.get("paper_title", "")) for c in load_cards(ctx)}
        papers_dir = ctx.out_dir / "papers"
        on_disk = {f.stem for f in papers_dir.glob("*.md")} if papers_dir.is_dir() else set()
        rows = []
        for pp in sorted(ws.papers.values(), key=lambda x: x.citation_count, reverse=True):
            key = _title_key(pp.title)
            has_card = key in carded
            if p.only_uncarded and has_card:
                continue
            marks = ("卡✓" if has_card else "卡✗") + (" 文✓" if key in on_disk else "")
            rows.append(f"{pp.id} | {pp.year or '?'} | c={pp.citation_count} | {marks} | {pp.title[:64]}")
        head = f"图谱 {len(ws.papers)} 篇，已落卡 {len(ws.papers) - len(rows) if p.only_uncarded else sum(1 for r in rows if '卡✓' in r)} 张"
        tip = "（文✓ = 原文已在 papers/，批量落卡可起子 agent 直接读盘，不必重复 read_paper）"
        return _ok(head + tip + "\n" + "\n".join(rows[:80]))

    async def expand_forward(p: ExpandParams) -> ToolResult[ToolUseCountMetadata]:
        pid = ws.resolve_id(p.paper_id) or p.paper_id
        if pid not in ws.papers:
            return _ok(f"'{pid}' not in graph. Pick a paper_id shown in the summary.")
        if pid in ws.expanded_fwd:
            return _ok(f"'{pid}' already expanded forward.")
        depth = ws.depth.get(pid, 0) + 1
        # most-cited citing papers (established descendants)
        citing = await ctx.multi.citing(pid, s.per_node_citations, "citations")
        via = ctx.multi.route_note()
        added = sum(ws.add_paper(c, depth) for c in citing)
        # ALSO the recent frontier: newest papers rank ~0 by raw citations, so the
        # most-cited sort systematically misses 2022+ work — pull it explicitly.
        recent = []
        if s.per_node_recent > 0:
            recent = await ctx.multi.citing(pid, s.per_node_recent, "recent_cited")
            added += sum(ws.add_paper(c, depth) for c in recent)
        # 拉取即记边：这批论文引用 pid 是拉取条件本身，不登记就等于扔掉——
        # arXiv/S2 形态的节点没有 refs 字段可事后匹配（实测 48 篇 0 边）
        for c in citing + recent:
            ws.note_edge(c.id, pid)
        ws.expanded_fwd.add(pid)
        return _ok(
            f"expand_forward({pid}): {len(citing)} most-cited + {len(recent)} recent citing papers{via}, "
            f"{added} new added.\n" + ws.summary()
        )

    async def expand_frontier(p: ExpandParams) -> ToolResult[ToolUseCountMetadata]:
        pid = ws.resolve_id(p.paper_id) or p.paper_id
        if pid not in ws.papers:
            return _ok(f"'{pid}' not in graph. Pick a paper_id shown in the summary.")
        if pid in frontier_done:
            return _ok(f"'{pid}' already frontier-expanded.")
        depth = ws.depth.get(pid, 0) + 1
        # IMPORTANT frontier first: recent papers ranked by citations (recent_cited) —
        # this is where Mamba/RWKV-tier new architectures live. Pure newest-by-date
        # ("recent") is dominated by ~0-citation drive-by application papers (noise), so
        # we take only a small bleeding-edge slice of it.
        important = await ctx.multi.citing(pid, max(s.per_node_recent, 15), "recent_cited")
        via = ctx.multi.route_note()
        added = sum(ws.add_paper(c, depth) for c in important)
        newest = await ctx.multi.citing(pid, 5, "recent")
        added += sum(ws.add_paper(c, depth) for c in newest)
        for c in important + newest:  # 拉取即记边（同 expand_forward）
            ws.note_edge(c.id, pid)
        frontier_done.add(pid)
        got = important + newest
        years = sorted({c.year for c in got if c.year}, reverse=True)[:5]
        return _ok(
            f"expand_frontier({pid}): {len(important)} important-recent + {len(newest)} newest "
            f"citing papers{via} (years: {years}), {added} new added.\n" + ws.summary()
        )

    async def expand_backward(p: ExpandParams) -> ToolResult[ToolUseCountMetadata]:
        pid = ws.resolve_id(p.paper_id) or p.paper_id
        if pid not in ws.papers:
            return _ok(f"'{pid}' not in graph.")
        if pid in ws.expanded_bwd:
            return _ok(f"'{pid}' already expanded backward.")
        paper = ws.papers[pid]
        papers = await ctx.multi.references_of(paper, s.per_node_references)
        via = ctx.multi.route_note()
        added = sum(ws.add_paper(pp, ws.depth.get(pid, 0) + 1) for pp in papers)
        for pp in papers:  # S2 后备取的 refs 不在 paper.referenced_works 里，补登记
            ws.note_edge(pid, pp.id)
        ws.expanded_bwd.add(pid)
        return _ok(f"expand_backward({pid}): {len(papers)} references{via}, {added} new added.\n" + ws.summary())

    async def search_recent(p: RecentSearchParams) -> ToolResult[ToolUseCountMetadata]:
        # newest papers on the topic, by DATE (or by relevance if requested) — reaches
        # brand-new follow-ups that have ~0 citations and unindexed citation edges
        # (missed by graph_search/expansion). OpenAlex indexing lags arXiv by months, so
        # for a hot sub-topic that just took off it can return literally 0 — arXiv is
        # the primary source here, OpenAlex is a bonus when it has caught up. `sort` is
        # the agent's call: pick submittedDate for the newest wave, relevance to find
        # what's most central to the phrase (which may be slightly older).
        oa = await ctx.source.search_recent(p.query, limit=15)
        ax = await ctx.arxiv.search(p.query, max_results=15, sort=p.sort) if ctx.arxiv else []
        hits = oa + ax
        added = sum(ws.add_paper(h, 1) for h in hits)
        yrs = sorted({h.year for h in hits if h.year}, reverse=True)[:4]
        return _ok(
            f"search_recent('{p.query}', sort={p.sort}): {len(hits)} recent papers "
            f"(years: {yrs}), {added} new added — use this to catch the latest follow-ups.\n" + ws.summary()
        )

    class FindSurveysParams(BaseModel):
        query: str = Field(default="", description="主题词（缺省用项目查询）。")
        limit: int = Field(default=8, description="返回条数上限。")

    async def find_surveys(p: FindSurveysParams) -> ToolResult[ToolUseCountMetadata]:
        # mine_surveys 的纯工具版：老版内嵌 LLM 精选被白名单挡掉后，综述挖掘这条
        # 高信号检索路径断了——这里只做机械检索+标题过滤，哪些值得吸收归外层判断
        import re as _re
        from itertools import zip_longest

        q = (p.query or ws.query).strip()
        batches = await ctx.multi.broad_search(f"{q} survey")
        pat = _re.compile(r"survey|review|overview|tutorial|advances in", _re.I)
        seen: dict[str, object] = {}
        for paper in (x for tup in zip_longest(*batches) for x in tup if x is not None):
            key = paper.title.strip().lower()
            if pat.search(paper.title) and key not in seen:
                seen[key] = paper
        hits = sorted(seen.values(), key=lambda x: (x.year or 0, x.citation_count), reverse=True)[: p.limit]
        if not hits:
            return _ok(f"find_surveys('{q}'): 没找到疑似综述。换更宽的主题词重试，或直接 graph_search。")
        lines = [f"  {pp.id} | {pp.year} | cites={pp.citation_count} | {pp.title[:80]}" for pp in hits]
        return _ok(
            f"find_surveys('{q}'): {len(hits)} 篇疑似综述（标题含 survey/review 等，按年份+引用排序）。\n"
            + "\n".join(lines)
            + "\n吸收路径：挑权威综述 add_seed 入图，再 expand_backward 拉它的精选引文——"
            "这是够到『综述作者认为值得引』的重要近期工作的高信号通道；是否权威归你判断。"
        )

    async def graph_summary(_p: EmptyParams) -> ToolResult[ToolUseCountMetadata]:
        return _ok(ws.summary(newest_first=True))

    def find_in_graph(p: FindInGraphParams) -> ToolResult[ToolUseCountMetadata]:
        q = p.query.strip().lower()
        hits = [
            pp for pp in ws.papers.values()
            if q in pp.title.lower()
            or q in pp.id.lower()
            or q in ((pp.source_ids or {}).get("arxiv") or "").lower()
        ]
        if p.newest_first:
            hits.sort(key=lambda pp: (pp.year or 0, pp.citation_count), reverse=True)
        else:
            hits.sort(key=lambda pp: pp.citation_count, reverse=True)
        if not hits:
            return _ok(f"图内没有匹配 '{p.query}' 的论文。确实不在图里时才用 graph_search 去外部检索。")
        lines = [f"图内匹配 {len(hits)} 篇："]
        for pp in hits[:10]:
            lines.append(f"  {pp.id} | {pp.year} | cites={pp.citation_count} | {pp.title[:70]}")
        if len(hits) > 10:
            order = "年份" if p.newest_first else "引用数"
            alt = "引用序" if p.newest_first else "newest_first=true 的年份序"
            # 实测宽词命中上百条只显引用序前 10，近年低引新论文被静默截掉
            lines.append(f"  …还有 {len(hits) - 10} 篇未显示（当前按{order}）。要看新论文换{alt}，或收窄关键词。")
        return _ok("\n".join(lines))

    return [
        Tool[SearchParams, ToolUseCountMetadata](
            name="graph_search",
            description="Free-text search for more papers in this field and add hits to the graph.",
            parameters=SearchParams,
            executor=graph_search,
        ),
        Tool[ListPapersParams, ToolUseCountMetadata](
            name="list_papers",
            description=(
                "列出图谱全部论文及其状态（是否已落卡、原文是否已在 papers/），按引用数排序。"
                "批量落卡/盘点进度用这个，不要用 find_in_graph 子串翻页凑清单。"
                "only_uncarded=true 只看待办。"
            ),
            parameters=ListPapersParams,
            executor=list_papers,
        ),
        Tool[RecentSearchParams, ToolUseCountMetadata](
            name="search_recent",
            description="Find recent papers on a topic (OpenAlex + arXiv). Use this to catch brand-new follow-up / frontier papers that have ~0 citations and unindexed citation edges — they won't show up via graph_search (relevance-ranked) or citation expansion. Pass a tight topic phrase. Choose `sort`: 'submittedDate' for the newest by-date wave, 'relevance' for what's most central to the phrase — call it twice with both sorts when a narrow hot phrase matters.",
            parameters=RecentSearchParams,
            executor=search_recent,
        ),
        Tool[ExpandParams, ToolUseCountMetadata](
            name="expand_forward",
            description="Fetch papers that CITE a node (newer work): both the most-cited descendants AND the recent frontier. Use on key nodes to see how the field developed.",
            parameters=ExpandParams,
            executor=expand_forward,
        ),
        Tool[ExpandParams, ToolUseCountMetadata](
            name="expand_frontier",
            description="Fetch the NEWEST papers citing a node (this year / last year), even if they have ~0 citations yet. Use on the founding/central node to reach the bleeding edge (2024-2026) that citation-ranked expansion misses.",
            parameters=ExpandParams,
            executor=expand_frontier,
        ),
        Tool[ExpandParams, ToolUseCountMetadata](
            name="expand_backward",
            description="Fetch a node's REFERENCES (older work) to trace where its ideas came from.",
            parameters=ExpandParams,
            executor=expand_backward,
        ),
        Tool[FindSurveysParams, ToolUseCountMetadata](
            name="find_surveys",
            description=(
                "找领域综述（纯检索，零 LLM）：多源检索后按标题过滤 survey/review，返回候选清单不自动入图。"
                "建图期用它补引文扩边够不到的重要工作：挑权威综述 add_seed，再 expand_backward 拉其精选引文。"
            ),
            parameters=FindSurveysParams,
            executor=find_surveys,
        ),
        Tool[EmptyParams, ToolUseCountMetadata](
            name="graph_summary",
            description="Show the current citation graph: node count, top papers by citations, and the newest additions by year（子题盘点看后者——引用序 Top 全是领域基石）。",
            parameters=EmptyParams,
            executor=graph_summary,
        ),
        Tool[FindInGraphParams, ToolUseCountMetadata](
            name="find_in_graph",
            description=(
                "在当前图谱内按标题/方法名/arxiv id 查论文 id（本地子串匹配，零成本零网络）。"
                "要定位图里已有的论文（比如精读前找 id）先用这个，不要用 graph_search——"
                "那会去外部检索并往图里加新节点。默认按引用数排序只显前 10；"
                "找近年低引新工作传 newest_first=true。"
            ),
            parameters=FindInGraphParams,
            executor=find_in_graph,
        ),
    ]

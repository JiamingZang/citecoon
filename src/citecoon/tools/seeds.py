"""Seed selection tools — the autonomous replacement for resolve_seeds.

Instead of a black box that recalls candidates AND picks the seed for the model,
we split it: `find_candidates` only recalls (dual-source, no LLM gate) and returns
the raw list; the agent inspects, asks the user when the candidates span different
fields, then commits its choice with `add_seed`. This gives the model full control
over disambiguation (and naturally triggers asking the user on genuine ambiguity).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from stirrup import Tool, ToolResult, ToolUseCountMetadata
from citecoon.core.utils import looks_like_id
from citecoon.core.models import Paper
from citecoon.util.text import normalize_title

from ..context import RunContext


def _ok(content: str) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content=content, metadata=ToolUseCountMetadata(), success=True)


class FindCandidatesParams(BaseModel):
    query: str = Field(
        default="",
        description="Term / paper title / fuzzy description / identifier to look up. Empty = use the run's "
        "original query. 短查询（1-3 词）命中率最高；长短语会先整串短语匹配、零命中时自动降级逐词 AND 合取，"
        "但仍建议优先短词。已知 arXiv id/DOI 时直接传 id（如 'arxiv:2502.09992'）最稳；认不出的新名词/查不到的具体论文，"
        "先用你环境里的联网搜索工具（WebSearch 等）查出它的 arXiv id 再传进来——不要凭记忆猜 id。",
    )


class AddSeedParams(BaseModel):
    paper_id: str = Field(description="id of a candidate (from find_candidates) to commit as a seed/anchor of the graph.")
    match_rationale: str = Field(
        description="Required self-check before committing: state the paper's actual technique/"
        "contribution (what it specifically does), then compare that against the exact wording "
        "of the current research topic — not just whether they share vocabulary. If the "
        "comparison reveals the paper is actually a different technique/field than the topic "
        "asks about, do not add_seed it — ask the user instead.",
    )
    set_topic: str = Field(
        default="",
        description="Optional: only pass this if the ORIGINAL query was too vague to search/frame "
        "with (e.g. an ambiguous acronym) and this seed resolves what the field actually is — then "
        "give a short topic phrase in your own words (NOT necessarily the seed's raw title, which "
        "may be too broad, e.g. a survey title) to anchor downstream relevance/founding/report on. "
        "Leave empty to keep the user's original query as the anchor — this is the common case.",
    )


def _fmt(p: Paper) -> str:
    srcs = []
    if p.id.startswith("W") or (p.source_ids or {}).get("openalex"):
        srcs.append("openalex")
    if (p.source_ids or {}).get("arxiv"):
        srcs.append("arXiv")
    src = "+".join(srcs) or "openalex"
    venue = f" | {p.venue}" if p.venue else ""
    return f"  {p.id} | {p.year} | cites={p.citation_count} | {src}{venue} | {p.title[:80]}"


def _reconcile(a: Paper, b: Paper) -> Paper:
    """Merge two same-title records (cross-source) into one clean candidate.

    Why: the same paper often appears as both an OpenAlex work and an arXiv
    preprint, and OpenAlex records can carry a wrong (re-indexed) year — e.g.
    "Attention Is All You Need" comes back as 2025 from OpenAlex while arXiv
    (1706.03762) correctly says 2017. Dropping either record (the old behaviour)
    hid the correct signal from the model. Instead we keep ONE candidate and
    reconcile: prefer the OpenAlex id as primary (it carries citation edges for
    graph expansion), take the EARLIEST plausible year (a paper's canonical year
    is its first appearance), union the source ids (so the arXiv id rides along
    for later full-text link-back), and keep the richest of the other fields.
    """
    primary, other = (a, b) if a.id.startswith("W") else ((b, a) if b.id.startswith("W") else (a, b))
    sources = {**(other.source_ids or {}), **(primary.source_ids or {})}
    for m in (a, b):  # make sure an arXiv id from either member is captured
        if m.id.startswith("arxiv:"):
            sources.setdefault("arxiv", m.id.split(":", 1)[1])
    years = [y for y in (a.year, b.year) if y and y > 1500]
    return primary.model_copy(update={
        "year": min(years) if years else primary.year,
        "citation_count": max(a.citation_count or 0, b.citation_count or 0),
        "source_ids": sources,
        "arxiv_url": primary.arxiv_url or other.arxiv_url,
        "pdf_url": primary.pdf_url or other.pdf_url,
        "doi": primary.doi or other.doi,
        "abstract": primary.abstract or other.abstract,
        "venue": primary.venue or other.venue,
    })


def build_seed_tools(ctx: RunContext) -> list[Tool]:
    candidates: dict[str, Paper] = {}  # shared cache: id -> Paper, populated by find_candidates
    # 挂到 ctx 上让读工具也能解析候选 id：实测 agent find_candidates 命中后直接
    # read_paper 却报 not found（候选池与图两个世界），被迫绕道交叉印证填卡
    ctx.candidate_pool = candidates
    explained = {"done": False}  # 教育文案只发一次：每轮重复 600+ 字符白吃上下文（实测拖慢思考）

    async def find_candidates(p: FindCandidatesParams) -> ToolResult[ToolUseCountMetadata]:
        q = (p.query or ctx.ws.query).strip()
        groups: dict[str, Paper] = {}  # normalized title -> merged Paper (dict preserves insertion order)

        def _add(papers):
            for pp in papers:
                key = normalize_title(pp.title)
                if not pp.id or not key:
                    continue
                # merge same-title records across sources instead of dropping duplicates
                groups[key] = _reconcile(groups[key], pp) if key in groups else pp

        def _hits_now() -> int:
            return ctx.multi.throttle_hits

        throttle_before = _hits_now()
        if looks_like_id(q):
            # 超新论文单源常未索引——id 直查走聚合层的多源链，任一源挂掉不致失败
            paper = await ctx.multi.get_paper(q)
            if paper:
                _add([paper])
        if not groups and not looks_like_id(q):
            # 全源广召回并发（聚合层编排；此前串行 await 壁钟是各源之和，实测单次 45s）。
            # id 查询不降级：拿 "arxiv:xxxx" 当短语全文检索毫无意义，限流时还白烧请求
            for rs in await ctx.multi.broad_search(q):
                _add(rs)
        # “名字 + 版本号”查询（如 "LLaDA 2"）：arXiv 把标题里的 "LLaDA2.0" 整个当一个词，
        # 分写形式和裸连写 "LLaDA2" 都匹不上，必须 ti:"LLaDA2.0" 全等（实测）——
        # 整数版本号顺带扫小版本，一条 OR 链一次 HTTP 全带上
        import re as _re
        m_ver = _re.fullmatch(r"(\D\S*)\s+(\d[\w.]*)", q.strip())
        if m_ver:
            name, ver = m_ver.group(1), m_ver.group(2)
            alts = [f'ti:"{name}{ver}"', f'ti:"{name} {ver}"']
            if ver.isdigit():
                alts += [f'ti:"{name}{ver}.{i}"' for i in range(6)]
            _add(await ctx.arxiv.search(" OR ".join(alts), max_results=8, sort="relevance"))
        if not groups and " " in q:
            # 短语整串匹配对长查询太严（标题词序稍有出入即零命中，实测烧掉 agent 8 轮试错）——
            # 降级成 all:"词" AND 逐词合取，同 review 查新的修法；降级在工具内部一次 HTTP 完成，
            # 不把重试成本推给推理层
            ws = q.split()
            fallback = f'all:"{ws[0]}"' + "".join(f' AND all:"{w}"' for w in ws[1:])
            _add(await ctx.arxiv.search(fallback, max_results=6, sort="relevance"))

        found = list(groups.values())
        for pp in found:
            candidates[pp.id] = pp

        if not found:
            if _hits_now() > throttle_before:
                cd = ctx.arxiv.cooldown_remaining()
                is_id = looks_like_id(q)
                budget_out = getattr(ctx.source, "budget_exhausted", lambda: False)()
                if is_id:
                    # 查询本身就是 id：id 直查刚刚失败，绝不能再建议"拿 id 来直查"——
                    # 实测 agent 会拿同一 id 连打三次原地打转
                    wait = max(cd, 30)
                    return _ok(
                        f"find_candidates('{q}'): id 直查已执行但两个数据源都失败了"
                        f"（OpenAlex {'额度耗尽' if budget_out else '限流'}，arXiv 限流{f'，约 {cd}s 后恢复' if cd else ''}）。"
                        f"这个 id 不必换姿势：等 {wait}s 后重试同一查询即可；连续重打只会延长限流。"
                        f"等待期间可先处理图内已有论文（read_paper/填卡）。"
                    )
                if budget_out:
                    return _ok(
                        f"find_candidates('{q}'): 空结果，但 OpenAlex 今日免费额度已耗尽（UTC 午夜重置）"
                        f"{f'，且 arXiv 也在限流（约 {cd}s 恢复）' if cd else ''}——这很可能是假零命中。"
                        f"缓存过的查询不受影响。可先用你环境里的联网搜索工具（WebSearch 等）查证论文是否存在、拿到准确 arXiv id；"
                        f"arXiv 恢复后用 find_candidates('arxiv:xxxx.xxxxx') 直查入图。"
                    )
                wait_hint = f"熔断中，约 {cd}s 后自动恢复" if cd else "稍等片刻"
                return _ok(
                    f"find_candidates('{q}'): 空结果，但本轮 arXiv/OpenAlex 出现限流/网络异常——这很可能是假零命中。"
                    f"不要换词（{wait_hint}后重试同一查询）；等待期间可先用你环境里的联网搜索工具（WebSearch 等）查证。"
                )
            return _ok(
                f"find_candidates('{q}'): no papers found（含逐词 AND 降级）. The term may be misspelled or "
                f"too obscure — 别猜 id：用你环境里的联网搜索工具（WebSearch 等）查出准确 arXiv id 再来，"
                f"或向用户要全名/作者/领域。"
            )
        head = (
            f"find_candidates('{q}'): {len(found)} candidate(s) (same-title records merged across OpenAlex+arXiv; "
            f"year = earliest appearance)."
        )
        if not explained["done"]:
            explained["done"] = True
            head += (
                " List order is just call order (OpenAlex then arXiv), NOT a "
                "relevance ranking — judge each by its year/cites, don't assume an earlier position means a "
                "better match. A candidate with a recent year and near-zero citations may be a brand-new paper "
                "that isn't well-indexed yet; if the query looks like it's pointing at a specific recent paper, "
                "don't discount such a candidate just because it's outranked on citations or listed later. If "
                "candidates span different fields/topics, ask the user which to focus on (offer them as choices), "
                "then add_seed the chosen one. For a well-known paper prefer the candidate whose title exactly "
                "matches and that appears on both sources. If they clearly agree on one topic, add_seed the best "
                "match."
            )
        lines = [head]
        lines += [_fmt(pp) for pp in found]
        return _ok("\n".join(lines))

    async def add_seed(p: AddSeedParams) -> ToolResult[ToolUseCountMetadata]:
        pid = p.paper_id
        paper = candidates.get(pid) or ctx.ws.papers.get(pid)
        if paper is None:
            paper = await ctx.source.get_paper(pid)
        if paper is None and "arxiv" in pid.lower():
            paper = await ctx.arxiv.get_by_id(pid)  # 同 find_candidates：超新论文 OpenAlex 无索引
        if paper is None:
            return _ok(f"'{pid}' not found. Pick a paper_id from find_candidates output.")
        # topic anchor is the user's original query by default; only change it if the
        # agent explicitly asks to (set_topic) — never silently overwritten with the
        # seed's raw title, which can be too broad (e.g. a survey) or off-focus.
        if p.set_topic.strip():
            ctx.ws.query = p.set_topic.strip()
        ctx.ws.add_seed(paper)
        ctx.ws.trace.add("agent", "seed", f"{paper.id} | {paper.title[:60]} | rationale: {p.match_rationale[:200]}")
        return _ok(
            f"add_seed: '{paper.title[:70]}' ({paper.year}) anchored as seed. Topic now = '{ctx.ws.query[:60]}'.\n"
            + ctx.ws.summary()
        )

    return [
        Tool[FindCandidatesParams, ToolUseCountMetadata](
            name="find_candidates",
            description="Recall candidate papers for a term from OpenAlex + arXiv (no auto-pick). Returns titles/years/citations/venue so YOU can judge which topic the user means. First step; inspect before committing a seed.",
            parameters=FindCandidatesParams,
            executor=find_candidates,
        ),
        Tool[AddSeedParams, ToolUseCountMetadata](
            name="add_seed",
            description="Commit a candidate (by paper_id from find_candidates) as a seed/anchor of the citation graph. Requires match_rationale: a genuine self-check of the paper's actual technique against the topic's exact wording, not a rubber stamp. The topic anchor used for relevance/founding/report stays the user's original query unless you pass `set_topic`. Add 1-3 seeds, then expand.",
            parameters=AddSeedParams,
            executor=add_seed,
        ),
    ]

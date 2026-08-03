"""多源聚合层：工具层的唯一数据入口，后备链全部收敛在这里。

设计约定：
- 工具层只跟 MultiSource 打交道，不感知任何具体数据源（OpenAlex/arXiv/S2/Crossref）；
  新增或调整源、修改降级顺序，只改本文件。
- 每个方法调用后 `last_route` 记录实际走的源（如 "openalex" / "s2(后备)"），
  工具可把它附进返回消息，agent 能看到数据来自哪里。
- throttle_hits 聚合所有子源——零命中时"疑似限流 vs 真无此文"的判定入口。
- 空结果不缓存的纪律由各子源自己保证。
"""
from __future__ import annotations

from ..core.models import Paper
from ..util.text import normalize_title


def _overlap(a: str, b: str) -> int:
    sa = {w for w in normalize_title(a).split() if len(w) > 3}
    sb = set(normalize_title(b).split())
    return len(sa & sb)


class MultiSource:
    def __init__(self, openalex=None, arxiv=None, s2=None, crossref=None):
        self.openalex = openalex
        self.arxiv = arxiv
        self.s2 = s2
        self.crossref = crossref
        self.last_route = ""

    # ---- 可观测性 --------------------------------------------------------
    @property
    def throttle_hits(self) -> int:
        return sum(
            getattr(src, "throttle_hits", 0)
            for src in (self.openalex, self.arxiv, self.s2, self.crossref)
            if src is not None
        )

    def route_note(self) -> str:
        """非主源命中时的来源注记（附进工具消息，agent 可见降级路径）。"""
        return f"（经 {self.last_route}）" if "后备" in self.last_route else ""

    # ---- id 直查 ----------------------------------------------------------
    async def get_paper(self, pid: str) -> Paper | None:
        """id 直查链：OpenAlex → arXiv → S2 → Crossref(DOI)。任一源挂掉不致直查失败。"""
        low = pid.lower()
        if self.openalex is not None:
            p = await self.openalex.get_paper(pid)
            if p:
                self.last_route = "openalex"
                return p
        if self.arxiv is not None and "arxiv" in low:
            p = await self.arxiv.get_by_id(pid)
            if p:
                self.last_route = "arxiv 后备"
                return p
        if self.s2 is not None:
            p = await self.s2.get_by_id(pid)
            if p:
                self.last_route = "s2 后备"
                return p
        if self.crossref is not None and (low.startswith("doi:") or pid.startswith("10.")):
            p = await self.crossref.get_by_doi(pid)
            if p:
                self.last_route = "crossref 后备"
                return p
        self.last_route = "全源未命中"
        return None

    # ---- 广召回（find_candidates 用）：并发打全部源，合并去重交给调用方 ----
    async def broad_search(self, q: str) -> list[list[Paper]]:
        import asyncio

        jobs = []
        if self.openalex is not None:
            jobs += [self.openalex.search(q, limit=8), self.openalex.search_recent(q, limit=4)]
        if self.arxiv is not None:
            jobs += [
                self.arxiv.search(f'"{q}"', max_results=6, sort="relevance"),
                self.arxiv.search(q, max_results=4, sort="submittedDate"),
            ]
        if self.s2 is not None:
            jobs.append(self.s2.search(q, limit=6))
        if self.crossref is not None:
            jobs.append(self.crossref.search(q, limit=5))
        return await asyncio.gather(*jobs) if jobs else []

    # ---- 标题解析（谱系接线用）：逐源试，词重叠护栏防错配 -------------------
    async def resolve_title(self, title: str) -> Paper | None:
        if self.openalex is not None:
            hits = await self.openalex.search(title, limit=1)
            if hits and _overlap(title, hits[0].title) >= 2:
                self.last_route = "openalex"
                return hits[0]
        if self.arxiv is not None:
            hits = await self.arxiv.search(f'"{title}"', max_results=1, sort="relevance")
            if hits and _overlap(title, hits[0].title) >= 2:
                self.last_route = "arxiv 后备"
                return hits[0]
        if self.s2 is not None:
            hits = await self.s2.search(title, limit=1)
            if hits and _overlap(title, hits[0].title) >= 2:
                self.last_route = "s2 后备"
                return hits[0]
        if self.crossref is not None:
            hits = await self.crossref.search(title, limit=1)
            if hits and _overlap(title, hits[0].title) >= 2:
                self.last_route = "crossref 后备"
                return hits[0]
        self.last_route = "全源未命中"
        return None

    # ---- 引文扩边 ----------------------------------------------------------
    async def citing(self, pid: str, limit: int, mode: str = "citations") -> list[Paper]:
        """谁引用了它。mode: citations（按引用数）/ recent_cited（新作里的重要者）/ recent（最新）。"""
        if self.openalex is not None:
            papers = await self.openalex.get_citing(pid, limit, mode)
            if papers:
                self.last_route = "openalex"
                return papers
        if self.s2 is not None:
            papers = await self.s2.citations(pid, max(limit, 50))
            if papers:
                if mode in ("recent_cited", "recent"):
                    papers.sort(key=lambda c: (c.year or 0, c.citation_count), reverse=True)
                self.last_route = "s2 后备"
                return papers[:limit]
        self.last_route = "全源未命中"
        return []

    async def references_of(self, paper: Paper, limit: int) -> list[Paper]:
        """它引用了谁。两种死法都接住：①有 refs 但 OpenAlex 批取被限流；
        ②节点根本没 refs（arXiv-only 记录）——S2 按论文 id 直取参考文献。"""
        refs = list(paper.referenced_works)[:limit]
        if refs and self.openalex is not None:
            papers = await self.openalex.get_many(refs)
            if papers:
                self.last_route = "openalex"
                return papers
        if self.s2 is not None:
            papers = await self.s2.references(paper.id, limit)
            if papers:
                self.last_route = "s2 后备"
                return papers
        self.last_route = "全源未命中"
        return []

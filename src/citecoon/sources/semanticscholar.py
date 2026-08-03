"""Semantic Scholar citation-count enrichment.

OpenAlex citation counts are wrong for some papers — e.g. "Attention Is All You
Need" comes back as 6576 (a broken/re-indexed record) when the real count is
~182k. Semantic Scholar, looked up by arXiv id or DOI, returns the accurate
count, so we use it to *correct* citation_count on graph nodes before metrics
(PageRank / velocity) and founding are computed. OpenAlex stays the primary
source; S2 only overwrites the citation number when it has a better one.

Best-effort by design: on rate limits (HTTP 429, common without an API key) or
any network error we back off a few times and then give up silently — the graph
still works with OpenAlex counts. Results are cached (per id) so repeat runs and
re-analysis never re-hit the API. Set S2_API_KEY to avoid the rate limit.
"""
from __future__ import annotations

import time

import httpx

from ..core.cache import Cache

_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
_GRAPH_BASE = "https://api.semanticscholar.org/graph/v1"
_BACKOFF = (2, 5, 10, 20)  # seconds between retries on 429 / transient errors
_MAX_IDS = 500  # S2 batch cap per request

_FIELDS = "title,year,abstract,citationCount,externalIds,venue,authors"


class S2Source:
    """Semantic Scholar 作为一等检索源：标题搜索 + id 直查。

    与 OpenAlex/arXiv 平行的第三源——单源限流/预算耗尽时检索链路不再致命
    （OpenAlex 每日预算制 + arXiv 严限流曾同时失效，实测检索全瞎）。
    匿名 ~100 req/5min；配 S2_API_KEY 提额。空结果不缓存。
    """

    name = "s2"

    def __init__(self, settings=None, cache: Cache | None = None):
        from ..util.ratelimit import RateLimiter

        self.cache = cache
        self._limiter = RateLimiter(rate_per_sec=0.3)
        self.throttle_hits = 0
        headers = {"User-Agent": "SuperAcademicAISearch/0.1"}
        key = getattr(settings, "s2_api_key", None) if settings else None
        if key:
            headers["x-api-key"] = key
        self._client = httpx.AsyncClient(base_url=_GRAPH_BASE, timeout=25.0, headers=headers)

    async def _get(self, path: str, params: dict) -> dict | None:
        await self._limiter.acquire()
        for attempt in range(2):
            try:
                r = await self._client.get(path, params=params)
                if r.status_code == 404:
                    return None
                if r.status_code == 429:
                    if attempt == 0:
                        import asyncio as _aio

                        await _aio.sleep(3)
                        continue
                    self.throttle_hits += 1
                    return None
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPError, ValueError):
                if attempt == 0:
                    continue
                self.throttle_hits += 1
                return None
        return None

    def _parse(self, d: dict):
        from ..core.models import Paper

        ext = d.get("externalIds") or {}
        arxiv = ext.get("ArXiv")
        doi = ext.get("DOI")
        pid = f"arxiv:{arxiv}" if arxiv else (f"s2:{d.get('paperId')}" if d.get("paperId") else "")
        if not pid or not d.get("title"):
            return None
        source_ids = {"s2": d.get("paperId")}
        if arxiv:
            source_ids["arxiv"] = f"arxiv:{arxiv}"
        return Paper(
            id=pid,
            title=d["title"],
            year=d.get("year"),
            abstract=(d.get("abstract") or "")[:4000] or None,
            citation_count=d.get("citationCount") or 0,
            venue=d.get("venue") or None,
            doi=doi,
            authors=[a.get("name", "") for a in (d.get("authors") or [])][:12],
            source_ids=source_ids,
            url=f"https://arxiv.org/abs/{arxiv}" if arxiv else None,
            pdf_url=f"https://arxiv.org/pdf/{arxiv}" if arxiv else None,
        )

    async def search(self, query: str, limit: int = 8) -> list:
        ck = f"s2:search:{query}:{limit}"
        if self.cache and (c := self.cache.get(ck)) is not None:
            from ..core.models import Paper

            return [Paper(**p) for p in c]
        data = await self._get("/paper/search", {"query": query, "limit": limit, "fields": _FIELDS})
        papers = [p for d in (data or {}).get("data", []) if (p := self._parse(d))]
        if self.cache and papers:  # 空结果不缓存：防限流期假零命中被固化
            self.cache.set(ck, [p.model_dump() for p in papers])
        return papers

    async def get_by_id(self, raw_id: str):
        """id 直查：支持 arxiv:xxxx / DOI:xxx / s2:<paperId>。"""
        rid = raw_id.strip()
        low = rid.lower()
        if low.startswith("arxiv:"):
            s2id = f"ARXIV:{rid.split(':', 1)[1]}"
        elif low.startswith("doi:") or rid.startswith("10."):
            s2id = f"DOI:{rid.split(':', 1)[-1]}"
        elif low.startswith("s2:"):
            s2id = rid.split(":", 1)[1]
        else:
            return None
        ck = f"s2:id:{s2id}"
        if self.cache and (c := self.cache.get(ck)) is not None:
            from ..core.models import Paper

            return Paper(**c)
        data = await self._get(f"/paper/{s2id}", {"fields": _FIELDS})
        paper = self._parse(data) if data else None
        if self.cache and paper:
            self.cache.set(ck, paper.model_dump())
        return paper

    def _to_s2id(self, raw_id: str) -> str | None:
        rid = raw_id.strip()
        low = rid.lower()
        if low.startswith("arxiv:"):
            return f"ARXIV:{rid.split(':', 1)[1]}"
        if low.startswith("doi:") or rid.startswith("10."):
            return f"DOI:{rid.split(':', 1)[-1]}"
        if low.startswith("s2:"):
            return rid.split(":", 1)[1]
        return None

    async def references(self, raw_id: str, limit: int = 20) -> list:
        """一篇论文引用了谁（S2 引文数据独立于 OpenAlex——后者预算耗尽时扩边的后备）。"""
        s2id = self._to_s2id(raw_id)
        if not s2id:
            return []
        ck = f"s2:refs:{s2id}:{limit}"
        if self.cache and (c := self.cache.get(ck)) is not None:
            from ..core.models import Paper

            return [Paper(**p) for p in c]
        data = await self._get(f"/paper/{s2id}/references", {"fields": _FIELDS, "limit": limit})
        papers = [p for d in (data or {}).get("data", []) if (p := self._parse(d.get("citedPaper") or {}))]
        if self.cache and papers:
            self.cache.set(ck, [p.model_dump() for p in papers])
        return papers

    async def citations(self, raw_id: str, limit: int = 20) -> list:
        """谁引用了这篇论文（客户端按引用数排序）。"""
        s2id = self._to_s2id(raw_id)
        if not s2id:
            return []
        ck = f"s2:cites:{s2id}:{limit}"
        if self.cache and (c := self.cache.get(ck)) is not None:
            from ..core.models import Paper

            return [Paper(**p) for p in c]
        data = await self._get(f"/paper/{s2id}/citations", {"fields": _FIELDS, "limit": max(limit, 50)})
        papers = [p for d in (data or {}).get("data", []) if (p := self._parse(d.get("citingPaper") or {}))]
        papers.sort(key=lambda p: p.citation_count, reverse=True)
        papers = papers[:limit]
        if self.cache and papers:
            self.cache.set(ck, [p.model_dump() for p in papers])
        return papers

    async def aclose(self) -> None:
        await self._client.aclose()


def _s2_id(doi: str | None, arxiv: str | None) -> str | None:
    """Map our ids to a Semantic Scholar id string (prefer arXiv, then DOI)."""
    if arxiv:
        return f"ARXIV:{arxiv}"
    if doi:
        d = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
        if d:
            return f"DOI:{d}"
    return None


def _fetch_counts(s2ids: list[str], api_key: str | None) -> dict[str, int]:
    """POST the S2 batch endpoint for citationCount. Returns {s2id: count} (found only)."""
    headers = {"x-api-key": api_key} if api_key else {}
    for attempt, wait in enumerate((*_BACKOFF, None)):
        try:
            with httpx.Client(timeout=25.0) as c:
                r = c.post(
                    _BATCH_URL,
                    params={"fields": "citationCount,externalIds"},
                    json={"ids": s2ids},
                    headers=headers,
                )
            if r.status_code == 429:
                if wait is None:
                    return {}
                time.sleep(wait)
                continue
            r.raise_for_status()
            out: dict[str, int] = {}
            # response list is positionally aligned with the request ids (null = not found)
            for sent, got in zip(s2ids, r.json()):
                if got and isinstance(got.get("citationCount"), int):
                    out[sent] = got["citationCount"]
            return out
        except Exception:  # noqa: BLE001 — network hiccup; back off and retry
            if wait is None:
                return {}
            time.sleep(wait)
    return {}


def enrich_citations(graph, settings, cache: Cache | None = None, trace=None) -> int:
    """Correct citation_count on graph nodes using Semantic Scholar.

    Walks nodes that carry an arXiv id or DOI, batch-fetches accurate counts
    (cache-first), and overwrites node/paper citation_count when S2's count is
    higher (keeps the original in metrics['cite_raw'] for audit). Returns the
    number of papers corrected. Never raises — enrichment is best-effort.
    """
    if not getattr(settings, "s2_enrich", True):
        return 0

    # collect (paper_id, s2id) for nodes we can resolve
    pid_to_s2: dict[str, str] = {}
    for n in graph.nodes:
        p = graph.papers.get(n.paper_id)
        if not p:
            continue
        sid = _s2_id(p.doi, (p.source_ids or {}).get("arxiv"))
        if sid:
            pid_to_s2[n.paper_id] = sid
    if not pid_to_s2:
        return 0

    # cache-first: only hit the network for ids we haven't seen
    counts: dict[str, int] = {}
    missing: list[str] = []
    for sid in set(pid_to_s2.values()):
        cached = cache.get(f"s2cite:{sid}") if cache else None
        if cached is not None:
            counts[sid] = cached
        else:
            missing.append(sid)

    for i in range(0, len(missing), _MAX_IDS):
        chunk = missing[i : i + _MAX_IDS]
        fetched = _fetch_counts(chunk, getattr(settings, "s2_api_key", None))
        for sid in chunk:
            val = fetched.get(sid, -1)  # -1 = known-missing, cached to avoid refetch
            counts[sid] = val
            if cache:
                cache.set(f"s2cite:{sid}", val)

    corrected = 0
    for pid, sid in pid_to_s2.items():
        s2c = counts.get(sid, -1)
        if s2c is None or s2c < 0:
            continue
        n = graph.node(pid)
        p = graph.papers.get(pid)
        old = (n.citation_count if n else 0) or 0
        if s2c > old:  # S2 is authoritative when it knows more citations
            if n:
                n.metrics = {**(n.metrics or {}), "cite_raw": old, "cite_source": "s2"}
                n.citation_count = s2c
            if p:
                p.citation_count = s2c
            corrected += 1

    if trace is not None and corrected:
        trace.add("agent", "note", f"S2 citation enrichment: corrected {corrected} paper(s)")
    return corrected

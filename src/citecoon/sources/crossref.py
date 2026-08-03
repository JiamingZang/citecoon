"""Crossref 检索源：DOI 元数据兜底（第四源）。

免注册；带 mailto 进礼貌池即可获得稳定服务。覆盖面最广（所有出版商 DOI），
但没有 arXiv 预印本；作为 OpenAlex / arXiv / Semantic Scholar 全部失效时的
最后一路检索兜底。空结果不缓存。
"""
from __future__ import annotations

import re

import httpx

from ..core.cache import Cache
from ..util.ratelimit import RateLimiter

_BASE = "https://api.crossref.org"
_TAG_RE = re.compile(r"<[^>]+>")


class CrossrefSource:
    name = "crossref"

    def __init__(self, settings=None, cache: Cache | None = None):
        self.cache = cache
        self.throttle_hits = 0
        self._limiter = RateLimiter(rate_per_sec=0.5)
        mailto = getattr(settings, "mailto", None) if settings else None
        self._mailto = mailto
        ua = f"SuperAcademicAISearch/0.1 (mailto:{mailto})" if mailto else "SuperAcademicAISearch/0.1"
        self._client = httpx.AsyncClient(base_url=_BASE, timeout=25.0, headers={"User-Agent": ua})

    async def _get(self, path: str, params: dict) -> dict | None:
        if self._mailto:
            params = {**params, "mailto": self._mailto}
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

    def _parse(self, w: dict):
        from ..core.models import Paper

        title = " ".join((w.get("title") or [""])[0].split())
        doi = w.get("DOI")
        if not title or not doi:
            return None
        year = None
        for key in ("published-print", "published-online", "issued", "created"):
            parts = ((w.get(key) or {}).get("date-parts") or [[None]])[0]
            if parts and parts[0]:
                year = parts[0]
                break
        abstract = w.get("abstract")
        if abstract:
            abstract = _TAG_RE.sub(" ", abstract)
            abstract = " ".join(abstract.split())[:4000]
        authors = [
            " ".join(x for x in (a.get("given"), a.get("family")) if x)
            for a in (w.get("author") or [])
        ][:12]
        # 参考文献里带 DOI 的先记下来（doi: 前缀 id），谱系接线时可解析
        refs = [f"doi:{r['DOI']}" for r in (w.get("reference") or []) if r.get("DOI")][:100]
        return Paper(
            id=f"doi:{doi}",
            title=title,
            doi=doi,
            year=year,
            abstract=abstract or None,
            authors=[a for a in authors if a],
            venue=(w.get("container-title") or [None])[0],
            citation_count=w.get("is-referenced-by-count") or 0,
            referenced_works=refs,
            url=w.get("URL"),
            source_ids={"crossref": doi},
        )

    async def search(self, query: str, limit: int = 6) -> list:
        ck = f"cr:search:{query}:{limit}"
        if self.cache and (c := self.cache.get(ck)) is not None:
            from ..core.models import Paper

            return [Paper(**p) for p in c]
        data = await self._get(
            "/works",
            {"query.bibliographic": query, "rows": limit, "select": "DOI,title,author,issued,published-print,published-online,created,abstract,container-title,is-referenced-by-count,URL"},
        )
        items = ((data or {}).get("message") or {}).get("items") or []
        papers = [p for w in items if (p := self._parse(w))]
        if self.cache and papers:  # 空结果不缓存：防限流期假零命中被固化
            self.cache.set(ck, [p.model_dump() for p in papers])
        return papers

    async def get_by_doi(self, doi: str):
        d = doi.strip().removeprefix("doi:").removeprefix("https://doi.org/")
        ck = f"cr:doi:{d}"
        if self.cache and (c := self.cache.get(ck)) is not None:
            from ..core.models import Paper

            return Paper(**c)
        data = await self._get(f"/works/{d}", {})
        paper = self._parse((data or {}).get("message") or {}) if data else None
        if self.cache and paper:
            self.cache.set(ck, paper.model_dump())
        return paper

    async def aclose(self) -> None:
        await self._client.aclose()

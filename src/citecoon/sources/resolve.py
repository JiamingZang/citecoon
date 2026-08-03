"""标题 → Paper 确定性解析（零 LLM）：wire_predecessors 的接线核心。"""
from __future__ import annotations

import asyncio

from ..core.models import Paper
from ..util.text import normalize_title

_STOP = {"the", "a", "an", "of", "for", "and", "in", "to", "with", "via", "using"}


def overlap(a: str, b: str) -> int:
    q = {w for w in normalize_title(a).split() if w not in _STOP}
    t = set(normalize_title(b).split())
    return len(q & t)


async def resolve_titles(
    titles: list[str], source, arxiv, s2=None, crossref=None, local: dict | None = None, multi=None
) -> tuple[dict[str, Paper], list[str], list[str]]:
    """标题 → Paper：本地图内先匹配（零请求），未命中走多源聚合层。
    返回 (found: id→Paper, ref_ids, misses: 未解析成功的标题)。"""
    from .multi import MultiSource

    ms = multi or MultiSource(openalex=source, arxiv=arxiv, s2=s2, crossref=crossref)

    # 前驱常常已经在图里，直接命中零请求——外部源全部限流时也照样接得上线
    local_index: list[tuple[str, Paper]] = []
    for p in (local or {}).values():
        if p.title:
            local_index.append((normalize_title(p.title), p))

    def _local_hit(t: str):
        key = normalize_title(t)
        if not key:
            return None
        for nk, p in local_index:
            if nk == key or (len(key) > 24 and (key in nk or nk in key)):
                return p
        for nk, p in local_index:  # 词重叠退一档（标题被截断/副标题差异）
            if overlap(t, p.title) >= 3:
                return p
        return None

    async def _resolve(t: str):
        hit = _local_hit(t)
        if hit is not None:
            return [hit]
        p = await ms.resolve_title(t)
        return [p] if p else []

    hit_lists = await asyncio.gather(*(_resolve(t) for t in titles))
    found: dict[str, Paper] = {}
    ref_ids: list[str] = []
    misses: list[str] = []
    for t, hits in zip(titles, hit_lists):
        if hits and overlap(t, hits[0].title) >= 2:  # guard against wrong matches
            found.setdefault(hits[0].id, hits[0])
            ref_ids.append(hits[0].id)
        else:
            misses.append(t)
    return found, ref_ids, misses

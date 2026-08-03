"""零 LLM 的通用小工具：从旧 harness 的 llm/resolver 层抢救出来的纯函数。"""
from __future__ import annotations

import json
import re

_ID_PATTERNS = [
    re.compile(r"^[Ww]\d+$"),  # OpenAlex
    re.compile(r"^10\.\d{4,9}/"),  # DOI
    re.compile(r"doi\.org/", re.I),
    re.compile(r"arxiv", re.I),
    re.compile(r"openalex\.org/", re.I),
]


def looks_like_id(text: str) -> bool:
    """查询串是否本身就是一个论文标识符（OpenAlex id / DOI / arXiv / URL）。"""
    t = text.strip()
    if " " in t and not any(p.search(t) for p in _ID_PATTERNS[2:]):
        return False
    return any(p.search(t) for p in _ID_PATTERNS)


def extract_json(text: str | None):
    """尽力从带杂质的文本里抠出 JSON（围栏/前后废话/坏格式都兜住）。"""
    if not text:
        return None
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = t.find(open_c), t.rfind(close_c)
        if i != -1 and j != -1 and j > i:
            block = t[i : j + 1]
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass
            try:
                from json_repair import repair_json
                repaired = repair_json(block, return_objects=True)
                if isinstance(repaired, (dict, list)):
                    return repaired
            except Exception:
                pass
    return None

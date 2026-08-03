"""Idea-card tools: structured per-paper contribution cards, the raw material for ideation.

A card upgrades a free-text reading note into six structured fields (problem /
method / core assumption / limitation / eval setup / resources). Cards are written
to <out_dir>/cards/<paper_id>.json the moment they're filled — they are project
memory, not conversation state, so any later session over the same out_dir can
consume them without export ceremonies.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from stirrup import Tool, ToolResult, ToolUseCountMetadata
from stirrup.core.models import EmptyParams

from ..context import RunContext

CARDS_DIR = "cards"


def _ok(content: str) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content=content, metadata=ToolUseCountMetadata(), success=True)


def _safe_id(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", paper_id)


def _title_key(title: str) -> str:
    """Normalized title as the card's identity — the same paper often exists in the
    graph twice (OpenAlex id + arXiv id); keying by title merges those duplicates."""
    t = title.lower().replace("&amp;", " and ").replace("&", " and ")
    t = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", t).strip("_")
    return t[:80] or "untitled"


def _substance(card: dict) -> int:
    """Rough information mass of a card — used to keep a thin re-fill (e.g. “未能读取
    原文” placeholders) from clobbering an earlier full-text card."""
    return sum(
        len(str(card.get(k) or ""))
        for k in ("problem", "method", "core_assumption", "limitation", "eval_setup", "resources")
    )


def load_cards(ctx: RunContext) -> list[dict]:
    """All cards in the project, newest last (shared with the ideation tools).
    Files starting with '_' are project metadata (e.g. _themes.json), not cards."""
    cards = []
    for f in sorted(ctx.project_dir(CARDS_DIR).glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            cards.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return cards


class FillCardParams(BaseModel):
    paper_id: str = Field(description="The paper id this card is about (graph id or 'arxiv:...').")
    paper_title: str = Field(description="Human-readable title of the paper.")
    problem: str = Field(description="要解决的问题：这篇论文针对什么具体问题（1-3 句，中文）。")
    method: str = Field(description="核心方法：它怎么解决的，关键机制是什么（2-4 句，中文）。")
    core_assumption: str = Field(
        description="核心假设：方法成立所依赖的关键假设/前提——这是组合新想法时最重要的字段（1-3 句，中文）。"
    )
    limitation: str = Field(
        description="局限：论文自己承认的 + 你读出来的局限，尤其是后续可能无人跟进的开放问题（中文）。"
    )
    eval_setup: str = Field(
        description="实验设置：数据集/基线/指标/规模，供之后设计对照实验参考（中文，紧凑）。"
    )
    resources: str = Field(
        default="",
        description="复现资源：代码/数据/算力需求，未知则留空（中文）。",
    )


class ListCardsParams(BaseModel):
    keyword: str | None = Field(
        default=None,
        description="Optional substring filter over title + all fields; omit to list everything.",
    )


def build_card_tools(ctx: RunContext) -> list[Tool]:

    def fill_idea_card(p: FillCardParams) -> ToolResult[ToolUseCountMetadata]:
        card = p.model_dump()
        paper = ctx.ws.papers.get(p.paper_id)
        if paper is not None:
            card["year"] = paper.year
            card["citation_count"] = paper.citation_count
        cards_dir = ctx.project_dir(CARDS_DIR)
        path = cards_dir / f"{_title_key(p.paper_title)}.json"
        # migrate: an older id-keyed card for the same title counts as the existing card
        existing = None
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
        else:
            for f in cards_dir.glob("*.json"):
                if f.name.startswith("_"):
                    continue  # _themes.json 等元文件是 JSON 数组，当卡解析会直接炸（实测）
                try:
                    c = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(c, dict):
                    continue
                if _title_key(c.get("paper_title", "")) == _title_key(p.paper_title):
                    existing = c
                    f.unlink(missing_ok=True)
                    break
        if existing is not None:
            # same paper under another graph id — remember all known ids
            ids = set(existing.get("paper_ids") or [existing.get("paper_id")]) | {p.paper_id}
            card["paper_ids"] = sorted(i for i in ids if i)
            if _substance(card) < _substance(existing) * 0.6:
                path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
                return _ok(
                    f"已有同名卡片且内容更实（很可能是同一论文的另一个图谱 id），本次内容明显更空，"
                    f"未覆盖：{path.name}。若确实要重写，请先读到全文再填。"
                )
        else:
            card["paper_ids"] = [p.paper_id]
        path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.ws.trace.add("agent", "fill_idea_card", f"{p.paper_id} | {p.paper_title[:60]}")
        n = len([f for f in cards_dir.glob("*.json") if not f.name.startswith("_")])
        verb = "更新" if existing is not None else "新建"
        # 回显带项目名：实测 agent 在错误项目里落卡（会话重启回退初始项目），
        # "现有 56 张卡片"的异常数字它看不懂，带上项目名一眼即穿
        return _ok(f"卡片已{verb}: {path.name}。项目 {ctx.out_dir.name} 现有 {n} 张卡片。")

    class ThemeIn(BaseModel):
        theme: str = Field(description="母题陈述：跨多篇论文的共性假设/路线分叉/结构空白（中文，一句话）。")
        type: str = Field(description="类型：假设趋同 / 路线分叉 / 结构空白 / 方法惯性 等。")
        cards: list[str] = Field(description="支撑该母题的论文标题列表（须与已有卡片的 paper_title 对得上）。")
        evidence: str = Field(description="证据：点名各卡片的哪个字段支撑该母题（中文）。")
        tension: str = Field(description="张力：该母题内未被解决的矛盾——idea 的直接原料（中文）。")

    class SaveThemesParams(BaseModel):
        themes: list[ThemeIn] = Field(description="全量母题列表（重算式覆盖 _themes.json，不是追加）。")
        force: bool = Field(default=False, description="卡片覆盖率不足时强制落盘（仅当用户明确要快速版时用）。")

    def save_themes(p: SaveThemesParams) -> ToolResult[ToolUseCountMetadata]:
        """母题落盘（零 LLM）：格式内置对齐老项目 {theme,type,cards,evidence,tension}，
        卡片引用逐条校验——外层 agent 不必再去老项目考古格式（实测每轮都在 ls 旧目录）。"""
        all_cards = load_cards(ctx)
        # 覆盖闸门：母题是跨论文归纳，卡太少时只能是提纲不是归纳（实测 51 篇图
        # 谱 4 张卡就起了 4 条母题，每条平均一张卡支撑）。"攒够"必须有操作定义。
        n_graph = len(ctx.ws.papers)
        need = min(8, max(3, n_graph // 3))
        if len(all_cards) < need and not p.force:
            carded = {_title_key(c.get("paper_title", "")) for c in all_cards}
            todo = sorted(
                (pp for pp in ctx.ws.papers.values() if _title_key(pp.title) not in carded),
                key=lambda x: x.citation_count, reverse=True,
            )[:5]
            hint = "；".join(f"{pp.title[:44]}({pp.citation_count})" for pp in todo)
            return _ok(
                f"save_themes 暂缓：当前 {len(all_cards)} 张卡 / 图谱 {n_graph} 篇，低于归纳门槛 {need} 张——"
                f"母题是跨论文归纳，卡太少会变成提纲。建议先精读落卡（高引未落卡：{hint}）。"
                f"用户明确要快速版时可带 force=true 强制落盘。"
            )
        known = {_title_key(c.get("paper_title", "")): c.get("paper_title") for c in all_cards}
        missing = []
        for t in p.themes:
            for ref in t.cards:
                if not _theme_ref_known(ref, known):
                    missing.append(ref)
        if missing:
            return _ok(
                "save_themes 拒绝：以下 cards 引用对不上任何已有卡片的 paper_title——"
                + "; ".join(sorted(set(missing))[:6])
                + f"。现有卡片：{', '.join(v for v in known.values() if v)[:400]}"
            )
        path = ctx.project_dir(CARDS_DIR) / "_themes.json"
        path.write_text(
            json.dumps([t.model_dump() for t in p.themes], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ctx.ws.trace.add("agent", "themes", f"{len(p.themes)} themes saved")
        return _ok(f"_themes.json 已写入 {len(p.themes)} 条母题（全量覆盖）。它是 idea 的证据中枢，写 idea 时点名『母题N 的 tension』。")

    def list_idea_cards(p: ListCardsParams) -> ToolResult[ToolUseCountMetadata]:
        cards = load_cards(ctx)
        if p.keyword:
            kw = p.keyword.lower()
            cards = [c for c in cards if kw in json.dumps(c, ensure_ascii=False).lower()]
        if not cards:
            return _ok("没有匹配的卡片。精读论文后用 fill_idea_card 填卡。")
        lines = [f"共 {len(cards)} 张卡片：\n"]
        for c in cards:
            lines.append(
                f"### {c.get('paper_title', '?')} ({c.get('year', '?')}) [{c.get('paper_id')}]\n"
                f"- 问题: {c.get('problem', '')}\n"
                f"- 方法: {c.get('method', '')}\n"
                f"- 核心假设: {c.get('core_assumption', '')}\n"
                f"- 局限: {c.get('limitation', '')}\n"
                f"- 实验设置: {c.get('eval_setup', '')}"
                + (f"\n- 资源: {c['resources']}" if c.get("resources") else "")
            )
        return _ok("\n\n".join(lines))

    def _theme_ref_known(ref: str, known: dict[str, str]) -> bool:
        """save_themes 同款匹配逻辑抽出来共用：全等或互为子串算对得上。"""
        k = _title_key(ref)
        return k in known or any(k in kk or kk in k for kk in known)

    def list_themes(_p: object) -> ToolResult[ToolUseCountMetadata]:
        """读现有母题（零 LLM）。实测母题重算是全量覆盖式，但没有任何读工具——
        外层只能裸读磁盘找 _themes.json；且历史短名别名（'NOCS'/'DUSt3R'）与卡片
        paper_title 脱节后一直静默失效，直到下一次 save_themes 才炸。这里顺带做
        漂移检查，把失效引用暴露在重算之前。"""
        path = ctx.project_dir(CARDS_DIR) / "_themes.json"
        if not path.exists():
            return _ok("还没有母题（cards/_themes.json 不存在）。卡片攒够后用 save_themes 归纳。")
        try:
            themes = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _ok("_themes.json 损坏，无法解析。用 save_themes 全量重写覆盖。")
        known = {_title_key(c.get("paper_title", "")): c.get("paper_title") for c in load_cards(ctx)}
        lines = [f"共 {len(themes)} 条母题（save_themes 为全量覆盖式，重算前先抄走全文）："]
        drift = []
        for i, t in enumerate(themes, 1):
            refs = t.get("cards", [])
            stale = [r for r in refs if not _theme_ref_known(r, known)]
            lines.append(f"{i}. [{t.get('type', '?')}] {t.get('theme', '?')}（{len(refs)} 卡）")
            if stale:
                drift.append((i, stale))
                lines.append(f"   ⚠ 失效引用 {len(stale)} 条: {', '.join(stale)}")
        if drift:
            lines.append(
                "漂移提示：失效引用多为卡片标题改名/短名别名所致；重算时逐条换成"
                "真实 paper_title（list_papers/list_idea_cards 可查），否则 save_themes 会拒绝。"
            )
        return _ok("\n".join(lines))

    return [
        Tool[FillCardParams, ToolUseCountMetadata](
            name="fill_idea_card",
            description=(
                "在精读一篇论文后，把它的贡献沉淀为结构化 idea card（问题/方法/核心假设/局限/"
                "实验设置/资源），立即持久化到项目的 cards/ 目录。这是后续 find_gaps / "
                "propose_idea 的原料——精读过的重要论文都应该填卡。同一篇论文（按标题归一）只存一张，"
                "重复填会合并更新。【没读到全文就不要填卡】——占位卡没有信息量，也覆盖不了已有的好卡。"
            ),
            parameters=FillCardParams,
            executor=fill_idea_card,
        ),
        Tool[SaveThemesParams, ToolUseCountMetadata](
            name="save_themes",
            description=(
                "把你归纳的母题落盘到 cards/_themes.json（零 LLM，格式内置，无需参考老项目）。"
                "每条母题 {theme,type,cards,evidence,tension}；cards 引用会逐条校验必须对上已有卡片标题。"
                "全量覆盖式写入。tension 字段是 idea 的直接原料，认真写。"
            ),
            parameters=SaveThemesParams,
            executor=save_themes,
        ),
        Tool[EmptyParams, ToolUseCountMetadata](
            name="list_themes",
            description=(
                "读当前项目的母题清单（cards/_themes.json），并检查每条母题的卡片引用"
                "是否仍对得上现有卡片标题（失效引用会标出）。母题重算（save_themes 是全量"
                "覆盖式）之前先调它拿到现有母题与漂移情况，不要裸读磁盘。"
            ),
            parameters=EmptyParams,
            executor=list_themes,
        ),
        Tool[ListCardsParams, ToolUseCountMetadata](
            name="list_idea_cards",
            description=(
                "列出项目里已沉淀的所有 idea card（可按关键词过滤）。在提出新想法、设计实验、"
                "或续接历史会话时先看这个，避免重读论文。"
            ),
            parameters=ListCardsParams,
            executor=list_idea_cards,
        ),
    ]

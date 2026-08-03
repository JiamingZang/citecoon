"""Codebase study: autonomous engineering-knowledge acquisition.

Papers get distilled into idea cards; repositories get distilled into repo cards.
When a plan or tech report needs engineering facts (hardcoded constants, entry
points, reproduction commands), the agent must NOT write them from LLM memory —
it spawns a sandboxed Qoder coding sub-agent that clones the repo, greps the
actual source, and writes a repo card with file:line evidence. Downstream tools
(write_tech_report) consume these verified cards as system context.
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime

from pydantic import BaseModel, Field

from stirrup import Tool, ToolResult, ToolUseCountMetadata

from ..context import RunContext

CODEBASES_DIR = "codebases"

_MAX_TURNS = 60
_WALL_SECONDS = 1200


def _ok(content: str) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content=content, metadata=ToolUseCountMetadata(), success=True)


def _repo_slug(repo: str) -> str:
    name = repo.rstrip("/").split("/")[-1]
    name = name.removesuffix(".git")
    return re.sub(r"[^\w.-]+", "_", name) or "repo"


def load_repo_cards(ctx: RunContext) -> list[tuple[str, str]]:
    """(name, markdown) of all studied repos — consumed by the tech-report bundle."""
    out = []
    for f in sorted(ctx.project_dir(CODEBASES_DIR).glob("*.md")):
        try:
            out.append((f.stem, f.read_text(encoding="utf-8")))
        except OSError:
            continue
    return out


class ListRepoCardsParams(BaseModel):
    name: str | None = Field(default=None, description="仓库名；省略则列出全部 repo 卡的摘要。")


def build_codebase_tools(ctx: RunContext) -> list[Tool]:

    def list_repo_cards(p: ListRepoCardsParams) -> ToolResult[ToolUseCountMetadata]:
        cards = load_repo_cards(ctx)
        if not cards:
            return _ok("还没有 repo 卡。用你的 Bash/Read 克隆并精读目标仓库，然后 fill_repo_card 落卡。")
        if p.name:
            for name, md in cards:
                if p.name.lower() in name.lower():
                    return _ok(md[:8000])
            return _ok(f"没有匹配 '{p.name}' 的 repo 卡。现有：{[n for n, _ in cards]}")
        lines = [f"共 {len(cards)} 张 repo 卡："]
        for name, md in cards:
            first = next((l for l in md.splitlines() if l.strip() and not l.startswith("#")), "")
            lines.append(f"- {name}: {first[:80]}")
        return _ok("\n".join(lines))

    class FillRepoCardParams(BaseModel):
        repo: str = Field(description="仓库地址：完整 git URL 或 GitHub 'org/name' 简写。卡片文件名由它生成，同仓库就地覆盖。")
        focus: str = Field(description="本次查证的关注点（来自当前 idea/报告），如'VRPO 的 ELBO 估计在哪个文件、采样预算怎么配、改成 TCSM score 要动哪几处'。")
        findings_markdown: str = Field(description="你亲自克隆/精读后的查证结果：架构要点、关键事实、硬编码参数、复现命令、改造接入点。代码级论断必须带 文件:行号 证据（如 `train/vrpo.py:142`），少于 3 处会被拒——凭模型记忆写的不要交。")

    def fill_repo_card(p: FillRepoCardParams) -> ToolResult[ToolUseCountMetadata]:
        """repo 卡落盘（零 LLM）：clone 与读码是外层 agent 的原生能力（Bash/Read），
        本工具只做证据格式校验与存储——对称 fill_idea_card：论文蒸馏成论文卡，
        仓库蒸馏成 repo 卡，报告的代码级细节只能引用后者。"""
        body = p.findings_markdown.strip()
        # 硬闸门：文件:行号 证据——这是 repo 卡与“凭记忆写的伪工程事实”的唯一区分线
        locs = re.findall(r"[\w./-]+\.\w{1,4}:\d+", body)
        if len(locs) < 3:
            return _ok(
                f"fill_repo_card 拒绝：文件:行号 证据仅 {len(locs)} 处（需 ≥3）——repo 卡的价值就在可回查的代码事实。"
                "先用你的 Bash 浅克隆（git clone --depth 1）、Read/Grep 精读，把论断落到 文件:行号 再交。"
            )
        slug = _repo_slug(p.repo)
        base = ctx.project_dir(CODEBASES_DIR)
        card_path = base / f"{slug}.md"
        existed = card_path.exists()
        url = p.repo if "://" in p.repo else f"https://github.com/{p.repo}"
        card_path.write_text(
            f"# repo 卡：{slug}\n\n> {url} · {datetime.now().strftime('%Y-%m-%d')} · 外层实读落卡（fill_repo_card）\n"
            f"> 关注点：{p.focus}\n\n{body}\n",
            encoding="utf-8",
        )
        ctx.ws.trace.add("agent", "repo_card", f"{slug} | {len(body)} chars | {len(locs)} locs")
        return _ok(
            f"repo 卡已{'覆盖' if existed else '新建'}: codebases/{slug}.md（{len(locs)} 处 文件:行号 证据）。"
            f"报告里的代码级论断用 evidence_refs 引用它；可行性评估章的实现复杂度/依赖风险应基于它写。"
        )

    return [
        Tool[FillRepoCardParams, ToolUseCountMetadata](
            name="fill_repo_card",
            description=(
                "把你亲自克隆、精读目标仓库后的查证结果落盘为 repo 卡（零 LLM，存 codebases/）。"
                "硬闸门：代码级论断必须带 文件:行号 证据（≥3 处），凭模型记忆写的会被拒。"
                "技术可行性报告的代码级细节（实现复杂度、依赖风险、改造接入点）只能引用 repo 卡；"
                "写可行性评估章前，先用你的 Bash 浅克隆（git clone --depth 1）+ Read/Grep 实读，再来落卡。"
            ),
            parameters=FillRepoCardParams,
            executor=fill_repo_card,
        ),
        Tool[ListRepoCardsParams, ToolUseCountMetadata](
            name="list_repo_cards",
            description="查看已探索仓库的 repo 卡（省略 name 列摘要，给 name 看全文）。",
            parameters=ListRepoCardsParams,
            executor=list_repo_cards,
        ),
    ]

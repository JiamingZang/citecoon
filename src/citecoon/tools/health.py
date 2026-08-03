"""项目状态驾驶舱 + 环境体检：纯文件/进程扫描，零 LLM。

status 同时以两种形态暴露（用户终态是单一对话 agent，CLI 只是脚手架）：
- agent 工具 project_status：chat 里一句"现在到哪了"即可全览
- CLI `citecoon status -p <project>`：脚本化入口

doctor 只做 CLI（进程/环境是系统级操作，不给 agent）。
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from stirrup import Tool, ToolResult, ToolUseCountMetadata

from ..context import RunContext
from ..core.layout import dir_for, find_result


def _count(d: Path, pattern: str) -> int:
    return len(list(d.glob(pattern))) if d.is_dir() else 0


def render_project_status(root: Path) -> str:
    """一屏看全：知识资产计数 + 每个 idea/报告/实验的推进度。"""
    if find_result(root) is None:
        return f"{root} 不是项目目录（缺 result.json）。"

    lines = [f"# 项目状态: {root.name}"]
    cards_dir = dir_for(root, "cards")
    n_cards = _count(cards_dir, "*.json") - (1 if (cards_dir / "_themes.json").is_file() else 0)
    lines.append(
        f"资产：论文卡 {n_cards} · repo 卡 {_count(dir_for(root, 'codebases'), '*.md')} · "
        f"原文 {_count(dir_for(root, 'papers'), '*.md')} · 母题 {'有' if (cards_dir / '_themes.json').is_file() else '无'}"
    )

    ideas_dir = dir_for(root, "ideas")
    ideas = sorted(ideas_dir.glob("*.md")) if ideas_dir.is_dir() else []
    lines.append(f"\n## ideas（{len(ideas)}）")
    for f in ideas:
        text = f.read_text(encoding="utf-8", errors="ignore")
        critiqued = "已评审" if ("评审" in text and ("通过" in text or "票" in text)) else "未见评审记录"
        lines.append(f"- {f.name} — {critiqued}")

    reports_dir = dir_for(root, "reports")
    reports = sorted(d for d in reports_dir.iterdir() if d.is_dir()) if reports_dir.is_dir() else []
    lines.append(f"\n## reports（{len(reports)}）")
    for d in reports:
        md = d / "report.md"
        if not md.is_file():
            lines.append(f"- {d.name} — 无 report.md")
            continue
        n_pending = md.read_text(encoding="utf-8", errors="ignore").count("待验证")
        review = d / "review.md"
        verdict = "未审校"
        if review.is_file():
            rt = review.read_text(encoding="utf-8", errors="ignore")
            # 长词优先：“需修订后发布”含“可发布”子串，顺序反了会误判
            for v in ("不可信需重写", "需修订后发布", "可发布"):
                if v in rt:
                    verdict = f"审校判定[{v}]"
                    break
        pdf = "PDF✓" if list(d.glob("*.pdf")) else "无PDF"
        lines.append(f"- {d.name} — {md.stat().st_size // 1024}KB · {verdict} · 待验证×{n_pending} · {pdf}")

    exps_dir = dir_for(root, "experiments")
    exps = sorted(d for d in exps_dir.iterdir() if d.is_dir()) if exps_dir.is_dir() else []
    lines.append(f"\n## experiments（{len(exps)}）")
    for d in exps:
        state = "无 PLAN"
        if (d / "PLAN.md").is_file():
            state = "有方案"
            if (d / ".approved").is_file():
                state = "已批准"
            rf = d / "result.json"
            if rf.is_file():
                try:
                    verdict = json.loads(rf.read_text(encoding="utf-8")).get("verdict", "?")
                    state = f"已实测（verdict: {verdict}）"
                except (OSError, json.JSONDecodeError):
                    state = "result.json 损坏"
        lines.append(f"- {d.name} — {state}")

    changelog = root / "CHANGELOG.md"
    if changelog.is_file():
        tail = [ln for ln in changelog.read_text(encoding="utf-8").splitlines() if ln.strip()][-3:]
        lines.append("\n## 最近变更")
        lines.extend(tail)
    return "\n".join(lines)


def render_doctor() -> str:
    """环境体检：孤儿子 agent 进程、env 配置完整性、缓存体积。只诊断不动手。"""
    lines = ["# citecoon 环境体检"]

    # 1. 残留 citecoon mcp 进程：客户端异常退出可能留下僵尸 server，
    #    只列出并给清理命令，不自动杀
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,etime,command"], capture_output=True, text=True, timeout=10,
        ).stdout
        orphans = []
        for ln in out.splitlines():
            if "citecoon mcp" in ln and "grep" not in ln:
                pid, etime = ln.split(None, 2)[:2]
                # etime 含 '-' 表示超过一天（dd-hh:mm:ss）
                if "-" in etime:
                    orphans.append(f"  pid {pid}（已运行 {etime}）")
        if orphans:
            lines.append(f"⚠ 疑似残留 citecoon mcp 进程 {len(orphans)} 个（运行超一天）：")
            lines.extend(orphans)
            lines.append(f"  清理：kill {' '.join(o.split()[1] for o in orphans)}")
        else:
            lines.append("✓ 无残留 citecoon mcp 进程")
    except (OSError, subprocess.SubprocessError):
        lines.append("? 进程检查失败（ps 不可用）")

    # 2. env 配置：当前 HOME 的副本 vs 真实用户 home 的真身
    cur = Path.home() / ".citecoon.env"
    try:
        import pwd
        real = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".citecoon.env"
    except (KeyError, OSError):
        real = cur

    def _n_vars(p: Path) -> int:
        if not p.is_file():
            return -1
        return sum(
            1 for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and "=" in ln and not ln.strip().startswith("#")
        )

    n_cur, n_real = _n_vars(cur), _n_vars(real)
    if n_cur >= 1:
        lines.append(f"✓ env: {cur}（{n_cur} 个变量）")
    elif n_real >= 1:
        lines.append(f"⚠ env: {cur} 缺失/为空（/tmp 会被 macOS 定期清理），已可回退真身 {real}（{n_real} 个变量）")
    else:
        lines.append(f"✗ env: {cur} 与 {real} 均无有效配置，LLM 功能不可用")
    if not os.getenv("OPENALEX_API_KEY") and not os.getenv("SAAS_MAILTO") and not os.getenv("OPENALEX_MAILTO"):
        lines.append("⚠ OPENALEX_API_KEY / OPENALEX_MAILTO 均未配置：匿名模式建图搜索可能被限流")

    # 3. 缓存体积
    cache = os.getenv("SAAS_CACHE_PATH") or str(Path.home() / ".citecoon" / "cache" / "superacademic.sqlite")
    cp = Path(cache)
    if cp.is_file():
        lines.append(f"✓ 缓存: {cp}（{cp.stat().st_size // 1024 // 1024} MB）")
    else:
        lines.append(f"- 缓存: {cp} 尚未建立")
    return "\n".join(lines)


def build_health_tools(ctx: RunContext) -> list[Tool]:
    from pydantic import BaseModel

    class EmptyParams(BaseModel):
        pass

    def project_status(_p: EmptyParams) -> ToolResult[ToolUseCountMetadata]:
        return ToolResult(content=render_project_status(ctx.out_dir), metadata=ToolUseCountMetadata())

    return [
        Tool[EmptyParams, ToolUseCountMetadata](
            name="project_status",
            description="项目推进度一屏全览（零成本纯扫描）：资产计数、每个 idea 的评审状态、每份报告的审校判定/待验证数/PDF、每个实验的方案/批准/实测状态、最近变更。用户问'现在到哪了/进展如何'时先调这个。",
            parameters=EmptyParams,
            executor=project_status,
        ),
    ]

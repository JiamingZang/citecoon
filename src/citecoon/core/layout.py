"""产物分层布局中枢：所有"项目目录里找东西"的代码都走这里。

布局（按研究流程编号，机器产物收进 _runtime）：

    projects/<课题>/
    ├── 01_graph/     result.json、graphml、view.html
    ├── 02_reading/   papers/ 原文 · cards/ 卡片+母题
    ├── 03_thinking/  ideas/
    ├── 04_writing/   reports/ · drafts/ · report.md
    ├── 05_code/      codebases/ · experiments/
    └── _runtime/     trace、autosave、changelog

带旧平铺布局回退：存量项目迁移前仍可读（新路径不存在时落回根下同名目录）。
"""
from __future__ import annotations

from pathlib import Path

GRAPH_DIR = "01_graph"
RUNTIME_DIR = "_runtime"

# project_dir(name) 的映射表（RunContext 用）
LAYOUT = {
    "cards": f"02_reading/cards",
    "papers": f"02_reading/papers",
    "ideas": f"03_thinking/ideas",
    "reports": f"04_writing/reports",
    "drafts": f"04_writing/drafts",
    "codebases": f"05_code/codebases",
    "experiments": f"05_code/experiments",
}


def sub(root: Path, name: str) -> Path:
    """新布局路径（不判断存在性，写路径用）。"""
    return root / LAYOUT.get(name, name)


def dir_for(root: Path, name: str) -> Path:
    """读路径：新布局存在用新布局，否则回退旧平铺（存量项目迁移期兼容）。"""
    new = root / LAYOUT.get(name, name)
    if new.is_dir():
        return new
    legacy = root / name
    return legacy if legacy.is_dir() else new


def find_result(root: Path) -> Path | None:
    """result.json / autosave 的定位（新布局 01_graph + _runtime，兼容旧根平铺）。"""
    for c in (
        root / GRAPH_DIR / "result.json",
        root / "result.json",
        root / RUNTIME_DIR / "result.autosave.json",
        root / "result.autosave.json",
    ):
        if c.is_file():
            return c
    return None


def is_project_dir(root: Path) -> bool:
    return find_result(root) is not None

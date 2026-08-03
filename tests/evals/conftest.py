"""Eval 夹具：造一个最小但结构完整的项目目录（新分层布局），供行为级回归测试用。

与 test_core_logic.py 的分工：那边测纯函数（输入→输出），这里测行为
（造真项目目录→跑真实调用路径→断言）。status verdict 子串误判等
只在行为层暴露，纯函数测试抓不到。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

THEMES_FILE = "_themes.json"


class ProjectBuilder:
    """链式造夹具：按 01_graph/02_reading/03_thinking/04_writing/05_code 布局。"""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        # 最小 result.json：load_project 靠它拿 query + graph
        g = root / "01_graph"
        g.mkdir(exist_ok=True)
        (g / "result.json").write_text(
            json.dumps({"query": "test-project", "graph": {"query": "test-project", "papers": {}}}),
            encoding="utf-8",
        )

    def add_cards(self, *names: str) -> "ProjectBuilder":
        d = self.root / "02_reading" / "cards"
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            fn = n if n.endswith(".json") else f"{n}.json"
            (d / fn).write_text(
                json.dumps({"paper_title": n, "summary": "x"}), encoding="utf-8"
            )
        return self

    def set_themes(self) -> "ProjectBuilder":
        d = self.root / "02_reading" / "cards"
        d.mkdir(parents=True, exist_ok=True)
        (d / THEMES_FILE).write_text(json.dumps([]), encoding="utf-8")
        return self

    def add_idea(self, name: str, *, critiqued: bool = False) -> "ProjectBuilder":
        d = self.root / "03_thinking" / "ideas"
        d.mkdir(parents=True, exist_ok=True)
        body = f"# {name}\n> 状态: draft\n"
        if critiqued:
            body += "\n## 评审记录\n评审：3/3 通过。\n"
        (d / f"{name}.md").write_text(body, encoding="utf-8")
        return self

    def add_report(self, name: str, *, verdict: str | None = None,
                   pending: int = 0, pdf: bool = False) -> "ProjectBuilder":
        d = self.root / "04_writing" / "reports" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.md").write_text(
            f"# {name}\n" + ("待验证\n" * pending), encoding="utf-8"
        )
        if verdict:
            (d / "review.md").write_text(f"## 判定\n总体判定：{verdict}。\n", encoding="utf-8")
        if pdf:
            (d / f"{name}.pdf").write_bytes(b"%PDF-1.4 stub")
        return self

    def add_experiment(self, name: str, *, plan=True, approved=False,
                       verdict: str | None = None) -> "ProjectBuilder":
        d = self.root / "05_code" / "experiments" / name
        d.mkdir(parents=True, exist_ok=True)
        if plan:
            (d / "PLAN.md").write_text(f"# 实验方案: {name}\n", encoding="utf-8")
        if approved:
            (d / ".approved").write_text("2026-01-01", encoding="utf-8")
        if verdict:
            (d / "result.json").write_text(
                json.dumps({"verdict": verdict}), encoding="utf-8"
            )
        return self


@pytest.fixture
def make_project(tmp_path):
    def _make(name: str = "proj") -> ProjectBuilder:
        return ProjectBuilder(tmp_path / name)
    return _make

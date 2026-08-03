"""单次 MCP 会话的共享状态。

所有工具闭包共享一个 RunContext：引文图谱（Workspace）、数据源、
结果槽位（emit_result 最终组装成 PipelineResult）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from citecoon.core.workspace import Workspace
from citecoon.core.config import Settings
from citecoon.core.models import CitationGraph, FieldReport, PipelineResult, Roadmap
from citecoon.sources.arxiv import ArxivSource
from citecoon.sources.openalex import OpenAlexSource
from citecoon.core.cache import Cache

# 产物分层布局：项目目录按研究流程编号，机器产物收进 _runtime
GRAPH_DIR = "01_graph"
RUNTIME_DIR = "_runtime"
_LAYOUT = {
    "cards": "02_reading/cards",
    "papers": "02_reading/papers",
    "ideas": "03_thinking/ideas",
    "reports": "04_writing/reports",
    "drafts": "04_writing/drafts",
    "codebases": "05_code/codebases",
    "experiments": "05_code/experiments",
}


@dataclass
class ReadingNote:
    """A single deep-reading insight recorded by the agent via take_note."""

    paper_id: str
    paper_title: str
    section: str  # e.g. "method", "abstract", "overall"
    content: str
    note_type: str  # "insight" / "confusion_resolved" / "key_finding" / "comparison"


@dataclass
class RunContext:
    """Everything the tools need for one research run."""

    settings: Settings
    cache: Cache
    source: OpenAlexSource
    arxiv: ArxivSource
    ws: Workspace
    out_dir: Path

    # 多源检索：工具层只认 multi（聚合层），后备链/降级路由全部收敛在 sources/multi.py
    s2: object | None = None  # S2Source（Semantic Scholar，第三源）
    crossref: object | None = None  # CrossrefSource（DOI 作后备，第四源）
    multi: object | None = None  # MultiSource 聚合入口

    # materialized citation graph (built once from ws, then reused)
    graph: CitationGraph | None = None
    # result slots — populated by tools during the loop, read by emit_result
    founding: list[str] = field(default_factory=list)
    roadmap: Roadmap = field(default_factory=Roadmap)
    report: FieldReport = field(default_factory=FieldReport)
    result_path: str | None = None
    result: "PipelineResult | None" = None  # final PipelineResult, set by emit_result
    # deep-reading notes — accumulated by take_note, exported by export_notes
    reading_notes: list[ReadingNote] = field(default_factory=list)

    def project_dir(self, name: str) -> Path:
        """File-backed research-project subdirectory, created on demand.

        These artifacts ARE the project memory: tools read/write the files
        directly, so a resumed or brand-new session over the same out_dir
        sees everything without extra session plumbing.
        """
        d = self.out_dir / _LAYOUT.get(name, name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def runtime_dir(self) -> Path:
        """机器产物目录（trace/autosave/changelog），人不用看。"""
        d = self.out_dir / RUNTIME_DIR
        d.mkdir(parents=True, exist_ok=True)
        return d

    def graph_dir(self) -> Path:
        """图谱产物目录（result.json / graphml / view.html）。"""
        d = self.out_dir / GRAPH_DIR
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def aclose(self) -> None:
        try:
            await self.source.aclose()
        except Exception:
            pass
        try:
            await self.arxiv.aclose()
        except Exception:
            pass
        for extra in (self.s2, self.crossref):
            if extra is not None:
                try:
                    await extra.aclose()
                except Exception:
                    pass
        try:
            self.cache.close()
        except Exception:
            pass


def build_context(
    query: str,
    *,
    out_dir: str | Path,
    max_nodes: int | None = None,
    max_depth: int | None = None,
    cache_path: str | None = None,
    on_event=None,
) -> RunContext:
    """Construct settings + sources + workspace for a run."""
    overrides: dict = {}
    if max_nodes is not None:
        overrides["max_nodes"] = max_nodes
    if max_depth is not None:
        overrides["max_depth"] = max_depth
    if cache_path is not None:
        overrides["cache_path"] = cache_path

    s = Settings.from_env(**overrides)
    cache = Cache(s.cache_path, enabled=s.use_cache)
    source = OpenAlexSource(s, cache)
    arxiv = ArxivSource(s, cache)
    from .sources.crossref import CrossrefSource
    from .sources.semanticscholar import S2Source

    s2 = S2Source(s, cache)
    crossref = CrossrefSource(s, cache)
    from .sources.multi import MultiSource

    multi = MultiSource(openalex=source, arxiv=arxiv, s2=s2, crossref=crossref)
    ws = Workspace(query, s, on_event=on_event)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return RunContext(
        settings=s, cache=cache, source=source, arxiv=arxiv, ws=ws, out_dir=out,
        s2=s2, crossref=crossref, multi=multi,
    )

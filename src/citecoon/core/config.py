"""Runtime configuration, populated from defaults + environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Load a gitignored `.env` (CWD or repo root) into os.environ.

    Zero-dependency. **`.env` values OVERRIDE existing env vars** — `.env` is the
    app's explicit local config, so it must win even when the surrounding session
    already exports ANTHROPIC_* (e.g. an internal gateway you want to override).
    This only affects the current (sub)process. `.env` is gitignored — never commit it.
    """
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]
    for f in candidates:
        if not f.is_file():
            continue
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ[key.strip()] = val.strip().strip('"').strip("'")
        except OSError:
            pass
        break


@dataclass
class Settings:
    # --- Data source (OpenAlex) ---
    openalex_base: str = "https://api.openalex.org"
    mailto: str | None = None  # OpenAlex "polite pool" contact email -> faster, nicer
    openalex_api_key: str | None = None  # free OpenAlex API key -> uninterrupted (anonymous search is rate-limited under load)
    request_timeout: float = 30.0
    max_concurrency: int = 5
    rate_per_sec: float = 8.0  # OpenAlex polite pool tolerates ~10 req/s

    # --- Graph construction defaults ---
    max_depth: int = 2
    max_nodes: int = 200
    # Working collection cap during expansion (defaults to max_nodes*4). Kept larger
    # than max_nodes so late-fetched frontier papers (expand_frontier) still get in
    # before the graph fills; the final graph is trimmed by relevance pruning, not
    # by this cap. Prevents old most-cited papers from squatting all the slots.
    max_collect: int | None = None
    per_node_citations: int = 25  # top-K most-cited citing papers per node
    per_node_recent: int = 10  # additional newest citing papers per node (surfaces frontier)
    per_node_references: int = 25  # top-K references to keep per node
    # broad arXiv topic recall: pull recent same-topic papers citation search misses
    # (off-topic ones are pruned later by relevance tagging)
    arxiv_recall: int = 20

    # --- Cache ---
    cache_path: str = ".cache/superacademic.sqlite"
    use_cache: bool = True

    # --- Semantic Scholar (citation-count enrichment) ---
    # OpenAlex citation counts can be wrong for some papers (e.g. "Attention Is All
    # You Need" = 6576 vs the real ~182k). S2 by arXiv id / DOI returns accurate
    # counts; we use it to correct citation_count before metrics. Optional API key
    # avoids the aggressive unauthenticated rate limit (HTTP 429).
    s2_enrich: bool = True
    s2_api_key: str | None = None

    # --- 报告语言（set_config 工具面保留；翻译由外层 agent 完成，此处仅记偏好） ---
    translate: bool = True

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        _load_dotenv()
        s = cls(
            mailto=os.getenv("SAAS_MAILTO") or os.getenv("OPENALEX_MAILTO"),
            openalex_api_key=os.getenv("OPENALEX_API_KEY") or os.getenv("SAAS_OPENALEX_API_KEY"),
            translate=os.getenv("SAAS_TRANSLATE", "1") not in ("0", "false", "False", "no"),
            s2_enrich=os.getenv("SAAS_S2_ENRICH", "1") not in ("0", "false", "False", "no"),
            s2_api_key=os.getenv("S2_API_KEY") or os.getenv("SEMANTICSCHOLAR_API_KEY"),
        )
        for k, v in overrides.items():
            if v is not None and hasattr(s, k):
                setattr(s, k, v)
        return s

"""CLI: citecoon 的运维入口——项目主体是 skill + MCP server。

    citecoon mcp -p <项目目录>     # MCP stdio server（纯确定性工具面）
    citecoon status/doctor/pdf/lib  # 零 LLM 运维子命令

调研/写作/审校等智能环节全部由外层 agent 通过 MCP 工具面完成
（见 skill/citecoon/SKILL.md）；本 CLI 不内置 agent 循环。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Cache == raw data (methodology): keep it in a stable, machine-portable location so
# re-runs never re-hit the API. Overridable via --cache-path or CITECOON_CACHE_PATH env.
_DEFAULT_CACHE = os.getenv("CITECOON_CACHE_PATH") or os.getenv("SAAS_CACHE_PATH") or str(
    Path.home() / ".citecoon" / "cache" / "superacademic.sqlite"
)

# Secrets (OpenAlex key 等) live OUTSIDE the repo so a scanning security/DLP
# agent doesn't quarantine an internal repo that contains plaintext credentials.
_EXTERNAL_ENV_FILE = os.getenv("CITECOON_ENV_FILE") or os.getenv("SAAS_ENV_FILE") or str(Path.home() / ".citecoon.env")


def _load_external_env(path: str = _EXTERNAL_ENV_FILE) -> None:
    """Load a gitignored, out-of-repo `.env` into os.environ (values win).

    当前 HOME 下的 env 缺失时（如 /tmp/home 被 macOS 清理）回退真实用户 home 的真身。
    """
    f = Path(path)
    if not f.is_file():
        try:
            import pwd

            real_home = pwd.getpwuid(os.getuid()).pw_dir
            fallback = Path(real_home) / ".citecoon.env"
            if fallback != f and fallback.is_file():
                f = fallback
            else:
                return
        except (KeyError, OSError):
            return
    try:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ[key.strip()] = val.strip().strip('"').strip("'")
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    _load_external_env()

    parser = argparse.ArgumentParser(prog="citecoon", description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    mcp_p = sub.add_parser("mcp", help="MCP stdio server（纯确定性工具面）")
    mcp_p.add_argument("-p", "--project", required=True, help="初始项目目录（use_project 可运行时切换）")
    mcp_p.add_argument("--cache-path", default=_DEFAULT_CACHE, help="SQLite 缓存路径")

    status = sub.add_parser("status", help="项目推进度一屏全览（零 LLM 纯扫描）")
    status.add_argument("-p", "--project", required=True, help="项目目录")

    sub.add_parser("doctor", help="环境体检（零 LLM）：孤儿进程、env 完整性、缓存体积")

    pdf = sub.add_parser("pdf", help="存量报告补导 PDF（本地渲染，无 LLM；多份严格串行）")
    pdf.add_argument("report", help="报告名（模糊匹配 reports/ 下唯一目录）")
    pdf.add_argument("-p", "--project", required=True, help="项目目录")

    args = parser.parse_args(argv)

    if args.cmd == "mcp":
        from .mcp.server import serve

        return serve(args.project, args.cache_path)
    if args.cmd == "status":
        from .tools.health import render_project_status

        print(render_project_status(Path(args.project)))
        return 0
    if args.cmd == "doctor":
        from .tools.health import render_doctor

        print(render_doctor())
        return 0
    if args.cmd == "pdf":
        from .ops import op_pdf

        print(op_pdf(args.project, args.report))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

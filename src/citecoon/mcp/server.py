"""citecoon MCP server —— 把研究工具面暴露给任意 MCP 客户端（Qoder/Claude 等外层 agent）。

架构定位：外层 agent 出循环质量与全流程编排，本进程只出【工具 + 会话状态】——
长驻进程持有内存图（RunContext.ws），研究会话跨多轮工具调用累积，emit_result 落盘持久化。

stdout 纪律：MCP stdio 协议要求 stdout 只有 JSON-RPC。依赖链会往 stdout 吐横幅
（qodercli 子进程继承 fd1 直接写，Python 层换 sys.stdout 挡不住），所以启动时把 fd1 整体
dup2 到 stderr，协议流走 dup 出来的真 stdout。

注册（Qoder / Claude Desktop mcpServers）：
    "citecoon": {
      "command": "<repo>/.venv/bin/python",
      "args": ["-m", "citecoon.mcp.server", "-p", "<项目目录>"]
    }
"""
from __future__ import annotations

import argparse
import os

# 纯工具面（默认）：只暴露确定性数据面工具，内嵌 LLM/子 agent 的一律不上——
# 智能全部留给外层 agent（codegraph 同款分层：daemon 只做索引与查询）。
# 外层自己能干的不重复提供：联网查证用它原生 WebSearch，卡片/母题/idea/报告用它原生文件工具写。
# emit_result 内含的翻译是后处理不是决策，保留。
_PURE_TOOLS = frozenset({
    # 检索与建图
    "find_candidates", "add_seed", "graph_search", "search_recent", "find_surveys",
    "expand_forward", "expand_backward", "expand_frontier",
    "graph_summary", "find_in_graph", "wire_predecessors", "list_papers",
    # 原文获取
    "read_paper", "read_local_pdf", "read_fulltext",
    # 状态与产出
    "emit_result", "project_status", "take_note", "export_notes",
    "export_report_pdf", "write_draft", "fill_domain_report", "fill_research_idea", "fill_roadmap",
    "save_report", "save_review", "save_report_section",
    "list_idea_cards", "fill_idea_card", "save_themes", "list_themes", "list_repo_cards", "fill_repo_card",
    "get_config", "set_config", "set_output_dir",
})

# server instructions：MCP 协议每会话必达（codegraph 同款）——SKILL.md 靠语义触发，
# 用户措辞不带"调研"时不激活，纪律整段丢失（实测三轮翻车：跳沉淀/跳查重/跳审校）。
# 这里只放最硬的顺序纪律与分流，打法细节仍在 SKILL.md。
_INSTRUCTIONS = """# citecoon —— 学术调研工具集（分析与写作由你完成）

工具只做机械操作：检索、引文图谱维护、原文获取、格式化落盘（带校验）、渲染导出。选题判断、谱系抽取、知识归纳、评审写作全部由你完成。

## 通用规矩（每会话生效）
- 找项目用 list_projects（不要去磁盘 find）；切换/新建用 use_project（'名字' / 'new:名字' / 绝对路径）。落盘类工具的回显带项目名，留意别在错误项目里干活。
- 精读即沉淀：每篇 read_paper 读完立刻 fill_idea_card 落卡再读下一篇；卡片攒够用 save_themes 归纳母题。
- 批量精读（超过 ~8 篇）别全走主上下文：read_paper 会把原文存进项目 papers/ 目录，起你自己的子 agent 分批读盘产出卡片要点，你只校对并 fill_idea_card 落盘。
- 研究想法只经 fill_research_idea 写（证据锚点会被校验），不要用文件工具裸写；更新时保持标题不变。
- 出 idea 后立即做一轮轻量查重：拆核心组合词，find_candidates/search_recent + 你的 WebSearch 三路撞车；撞上强相关就地改 idea 或在正文明写差异化，查重结论附进 idea。完整对抗评审留给技术可行性报告闭环，不在此重复。
- 增量刷新中被大改（结论/机制变了）的 idea，传导到报告层之前必须重跑上述轻量查重——新组合可能撞上新发表的工作。
- 调研收口固定顺序：fill_domain_report（领域导读报告，引用只能用图谱内论文 id）→ fill_roadmap（演化路线图：关键节点角色+演化边，不调则 roadmap 产物为空）→ emit_result（必调）。
- 检索结果为空但提示"疑似限流"时：等待后重试同一查询，别急着换词；"确实没有此文"再用你的 WebSearch 查出论文 id 直查。

## 子 agent 分工契约（起子 agent 时适用）
- 信息隔离：每个子 agent 只给它该看的素材，并明写不许看什么。审校子 agent 只拿报告正文 + 素材库（cards/papers/codebases/ideas），不拿写作过程的思路与辞护，也不拿其他子 agent 的结论——否则审校独立性归零。子 agent 不得再起子 agent。
- 候选与定级分离：子 agent 只产『候选问题 + 核查过程 + 证据边界』，不定 P0/P1/P2；最终分级由你逐条复核后统一给出。合并仅限同一位置+同一后果+同一修法的候选；主题相近但后果不同的必须各自保留。
- 委派上限：对抗评审一轮最多 3 个子 agent，审校只起 1 个；批量读盘落卡一批最多 3 个。不够就分批跑，不要越跑越多。

## 不得为省事而做（硬约束）
- 不得为缩短报告、凑整数或赶进度而砍掉已成立的发现/候选/论文；证据不够就标『待验证』并说明缺什么。
- 不得把『未能查证』写成『不存在』；检索为空、源不可用、没查到出处是三种不同状态，分开说。
- 不得用自己写的审校代替独立审校，不得跳过修订直奔出版。
- 工具回报拒绝时不得绕道（换措辞、自建编号、拆成多次绕过阈值）——拒绝消息说的就是缺口本身。

## 报告分流（写错层等于白写）
- 领域导读报告 → fill_domain_report + emit_result。
- 轻量综述/盘点 → save_report + export_report_pdf。
- 技术可行性报告（判断某想法值不值得投入的决策文档）→ 完整流程，环节一个不能少，一个想法一份：
  1) 查重：find_candidates/search_recent + 你的 WebSearch 三路撞车；检索为空≠无相关工作。
  2) 对抗评审：起子 agent 分别扮三个怀疑者（新颖性/技术成立性/实验设计）+ 可执行性五维定级（仓库/权重/GPU/数据/胶水代码）。三个怀疑者彼此不传结论，只产候选不定级。
  3) 逐段查证写作：只用 save_report_section（一轮一节，节标题带 N.M 编号，evidence_refs 须对上图谱/卡片/原文/repo 卡，否则拒）。骨架固定五章：背景与动机/方法/实验计划/可行性评估/结论——『要不要做/怎么做/代价/风险』是校验维度不是章名，缺必备小节出版会被拒。代码级论断（实现复杂度/依赖风险/接入点）先用你的 Bash 浅克隆目标仓库实读，fill_repo_card 落卡后引用——不许凭模型记忆写。查不到出处标『待验证』，不许编。
  4) 审校：另起对立视角子 agent，只给它报告正文+素材库（不给写作思路），它出候选问题清单与核查过程 → 你定 P0/P1/P2 后 save_review（可一并传四维评分；没有审校记录不给出 PDF）。
  5) 修订：核实来源后 save_report_section 同标题节就地替换。
  6) export_report_pdf（多份严格串行）。
"""


def serve(project: str, cache_path: str | None = None) -> int:
    # ---- fd 级 stdout 保护：必须在任何重型 import 之前 ----
    real_fd = os.dup(1)
    os.dup2(2, 1)
    real_stdout = os.fdopen(real_fd, "w", encoding="utf-8", buffering=1)

    import anyio
    from mcp.server import MCPServer
    from mcp.types import TextContent
    from mcp.types import Tool as McpTool

    from ..cli import _load_external_env

    _load_external_env()  # ~/.citecoon.env 的密钥在 build_context 读 os.environ 前就位

    from ..ops import load_project
    from ..tools import build_tools, build_emit_result_tool

    # 重启恢复最后活跃项目：-p 只是首次默认——实测两次事故：/quit 重启后 server
    # 回到初始项目，agent 带着"我在 jepa"的记忆继续干活，卡片误存错项目
    import sys as _sys
    from pathlib import Path as _PP

    _LAST_FILE = _PP.home() / ".citecoon" / "last_project"
    try:
        _last = _LAST_FILE.read_text(encoding="utf-8").strip()
        if _last and _PP(_last).is_dir() and (
            (_PP(_last) / "01_graph" / "result.json").is_file()
            or (_PP(_last) / "result.json").is_file()
            or (_PP(_last) / "_runtime" / "result.autosave.json").is_file()
        ):
            if str(_PP(project).resolve()) != str(_PP(_last).resolve()):
                print(f"[citecoon-mcp] 恢复最后活跃项目：{_last}（-p {project} 仅作首次默认）", file=_sys.stderr)
            project = _last
    except OSError:
        pass

    def _remember_project(path: str) -> None:
        try:
            _LAST_FILE.parent.mkdir(parents=True, exist_ok=True)
            _LAST_FILE.write_text(str(_PP(path).resolve()), encoding="utf-8")
        except OSError:
            pass

    ctx = load_project(project, cache_path=cache_path)
    _remember_project(project)

    def _autosave(c) -> None:
        """图谱变更后的边车快照（纯确定性，裸 to_graph，不跑 LLM 相关性标注）。

        实测事故：外层 agent 思考里说了 emit 却忘了调，95 篇图只在内存——纪律问题
        工程化：不赌记性，每次图变更自动落 result.autosave.json（不覆盖 emit_result
        的精装 result.json；load_project 在 result.json 缺失时回落边车）。"""
        try:
            g = c.ws.to_graph()
            payload = {
                "query": c.ws.query,
                "seeds": list(c.ws.seeds),
                "graph": g.model_dump(),
                "autosave": True,
            }
            p = c.out_dir / "_runtime" / "result.autosave.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            import json as _json
            p.write_text(_json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception as _e:  # noqa: BLE001 — 快照失败不能影响工具调用本身
            import sys, traceback
            print(f"[autosave] failed: {_e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def _build(c):
        ts = build_tools(c)
        ts.append(build_emit_result_tool(c))
        # 白名单仍保留：防止未来加回带内层推理的工具时悄悄暴露
        ts = [t for t in ts if t.name in _PURE_TOOLS]
        return {t.name: t for t in ts}

    # ask_user 不暴露：MCP 客户端侧的外层 agent 自己负责与用户对话
    # holder.pool 是 LRU 项目池（上限 3）：来回切项目只换指针不销毁，笔记/候选池等
    # 内存态得以保留；池满才快照+关闭最旧的。"切换前必须 emit 否则丢状态"的纪律作废。
    from collections import OrderedDict
    from pathlib import Path as _P

    holder = {
        "ctx": ctx,
        "tools": _build(ctx),
        "pool": OrderedDict({str(_P(project).resolve()): (ctx, None)}),  # path -> (ctx, tools|None)
    }
    _POOL_MAX = 3

    def _roots() -> list:
        base = _P(project).resolve().parent
        return [c for c in {base, base.parent / "results", base.parent / "projects"} if c.is_dir()]

    def _known_projects() -> list[dict]:
        import json as _json

        out, seen = [], set()
        for root in _roots():
            for d in sorted(root.iterdir()):
                cand = [d / "01_graph" / "result.json", d / "result.json", d / "_runtime" / "result.autosave.json"]
                f = next((c for c in cand if c.is_file()), None)
                if not d.is_dir() or f is None or str(d.resolve()) in seen:
                    continue
                seen.add(str(d.resolve()))
                try:
                    g = _json.loads(f.read_text(encoding="utf-8")).get("graph") or {}
                    n, q = len(g.get("papers") or {}), (g.get("query") or d.name)[:40]
                except Exception:  # noqa: BLE001
                    n, q = 0, d.name
                out.append({"path": str(d.resolve()), "name": d.name, "papers": n, "query": q,
                            "mtime": f.stat().st_mtime})
        out.sort(key=lambda x: x["mtime"], reverse=True)
        return out

    def _project_brief(c) -> str:
        """切换回显带资产概况：agent 落地即知家底，省一轮 project_status。"""
        def n_files(sub, pat):
            d = c.out_dir / sub
            return len([f for f in d.glob(pat) if not f.name.startswith("_")]) if d.is_dir() else 0

        return (
            f"图谱 {len(c.ws.papers)} 篇（种子 {len(c.ws.seeds)}）· 卡片 {n_files('02_reading/cards', '*.json')} · "
            f"idea {n_files('03_thinking/ideas', '*.md')} · 报告 {n_files('04_writing/reports', '*')} · 原文 {n_files('02_reading/papers', '*')}"
            + ("（含母题）" if (c.out_dir / "02_reading" / "cards" / "_themes.json").is_file() else "")
        )

    server = MCPServer("citecoon", instructions=_INSTRUCTIONS)

    _USE_PROJECT = McpTool(
        name="use_project",
        description=(
            "切换当前研究项目。传绝对路径，或已知项目名；传 'new:<名字>' 在惯例目录下新建项目。"
            "常用项目缓存在内存池里来回切换零成本；切走的项目图谱自动快照，不丢。"
        ),
        inputSchema={
            "type": "object",
            "properties": {"project": {"type": "string", "description": "项目绝对路径 / 已知项目名 / new:<新项目名>"}},
            "required": ["project"],
        },
    )
    _LIST_PROJECTS = McpTool(
        name="list_projects",
        description="列出全部已知研究项目（名字/主题/论文数/最后更新），并标注当前项目。找项目用这个，不要去磁盘 find。",
        inputSchema={"type": "object", "properties": {}},
    )

    async def _list_tools(ctx_req, params):
        # 覆盖 MCPServer 内置 tools/list：工具集随项目切换动态变化，静态注册行不通
        from mcp.types import ListToolsResult
        return ListToolsResult(tools=[_USE_PROJECT, _LIST_PROJECTS] + [
            McpTool(
                name=t.name,
                description=t.description or "",
                inputSchema=t.parameters.model_json_schema(),
            )
            for t in holder["tools"].values()
        ])

    async def _call_tool(ctx_req, params):
        from mcp.types import CallToolResult

        name, arguments = params.name, params.arguments

        def _txt(s: str) -> CallToolResult:
            return CallToolResult(content=[TextContent(type="text", text=s)])
        if name == "list_projects":
            cur = str(holder["ctx"].out_dir.resolve())
            lines = ["已知项目（按最后更新排序）："]
            for pr in _known_projects():
                mark = " ←当前" if pr["path"] == cur else ""
                lines.append(f"- {pr['name']}（{pr['papers']} 篇 · 主题「{pr['query']}」）{mark}")
            lines.append("切换：use_project('<名字>')；新建：use_project('new:<名字>')")
            return _txt("\n".join(lines))
        if name == "use_project":
            proj = str((arguments or {}).get("project") or "").strip()
            if not proj:
                return _txt("use_project 需要 project 参数（绝对路径 / 已知项目名 / new:<名字>）")
            if proj.startswith("new:"):
                # 显式新建：命名惯例内置（惯例根下蛇形目录），手滑错路径不再静默建库
                name_new = proj[4:].strip().replace(" ", "_")
                if not name_new:
                    return _txt("new: 后面要带项目名，如 new:jepa_world_models")
                proj = str(_roots()[0] / name_new)
            elif not _P(proj).is_absolute():
                cand = [r / proj for r in _roots()]
                hit = next((c for c in cand if (c / "01_graph" / "result.json").is_file()
                            or (c / "result.json").is_file()
                            or (c / "_runtime" / "result.autosave.json").is_file()), None)
                if hit is None:
                    known = ", ".join(p["name"] for p in _known_projects()) or "无"
                    return _txt(f"'{proj}' 不是已知项目。已知：{known}。新建请用 new:{proj}，或传绝对路径。")
                proj = str(hit)
            elif not _P(proj).exists():
                return _txt(f"路径不存在：{proj}。新建项目请用 use_project('new:<名字>')（将建在惯例目录下），避免手滑错路径静默建库。")
            key = str(_P(proj).resolve())
            old = holder["ctx"]
            if old.ws.papers:
                _autosave(old)  # 切走前快照（防进程挂掉）；旧 ctx 仍留在池里不销毁
            pool = holder["pool"]
            if key in pool:
                new_ctx, cached_tools = pool.pop(key)
                pool[key] = (new_ctx, cached_tools)  # 提到 LRU 队尾
                holder["ctx"] = new_ctx
                holder["tools"] = cached_tools or _build(new_ctx)
                pool[key] = (new_ctx, holder["tools"])
                _remember_project(key)
                return _txt(f"已切回项目 {new_ctx.out_dir.name}（内存态保留）：{_project_brief(new_ctx)}")
            new_ctx = load_project(proj, cache_path=cache_path)
            holder["ctx"] = new_ctx
            holder["tools"] = _build(new_ctx)
            pool[key] = (new_ctx, holder["tools"])
            _remember_project(key)
            while len(pool) > _POOL_MAX:  # 逐出最久未用的：快照+关闭
                _, (ev_ctx, _t) = pool.popitem(last=False)
                if ev_ctx.ws.papers:
                    _autosave(ev_ctx)
                try:
                    await ev_ctx.aclose()
                except Exception:  # noqa: BLE001 — 旧连接收尾失败不影响切换
                    pass
            return _txt(f"已切换到项目 {new_ctx.out_dir.name}（主题「{new_ctx.ws.query[:40]}」）：{_project_brief(new_ctx)}")
        t = holder["tools"].get(name)
        if t is None:
            return _txt(f"未知工具: {name}（用 tools/list 查看可用工具）")
        c = holder["ctx"]
        n_papers_before = len(c.ws.papers)
        try:
            params = t.parameters.model_validate(arguments or {})
            import asyncio, inspect
            raw = t.executor(params)
            result = await raw if inspect.isawaitable(raw) else raw
            text = result.content if isinstance(result.content, str) else str(result.content)
        except Exception as e:  # noqa: BLE001 — 单次调用失败不能带崩整个 server
            text = f"{name} 执行失败：{type(e).__name__}: {e}"
        if len(c.ws.papers) != n_papers_before:
            _autosave(c)  # 图变更即落盘，进程挂掉/忘 emit 都不丢图
        return _txt(text)

    # 覆盖底层 tools/list 与 tools/call（MCPServer 构造时注册的是静态工具管理版本）
    from mcp_types import CallToolRequestParams, PaginatedRequestParams
    ll = server._lowlevel_server
    ll.add_request_handler("tools/list", PaginatedRequestParams, _list_tools)
    ll.add_request_handler("tools/call", CallToolRequestParams, _call_tool)

    async def _run() -> None:
        # run_stdio_async 内部用默认 stdout（已被 dup2 指到 stderr），必须显式传真 stdout
        from mcp.server.stdio import stdio_server
        out = anyio.wrap_file(real_stdout)
        async with stdio_server(stdout=out) as (read, write):
            await ll.run(read, write, ll.create_initialization_options())

    try:
        anyio.run(_run)
    finally:
        import asyncio

        for _key, (_c, _t) in list(holder["pool"].items()):
            try:
                asyncio.run(_c.aclose())
            except Exception:  # noqa: BLE001 — 收尾失败不影响退出码
                pass
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="citecoon-mcp", description="citecoon 研究工具的 MCP stdio server")
    ap.add_argument("-p", "--project", required=True, help="项目目录（含 result.json；不存在则按目录名起新会话）")
    ap.add_argument("--cache-path", default=None, help="SQLite 缓存路径（默认同 CLI）")
    args = ap.parse_args(argv)
    return serve(args.project, args.cache_path)


if __name__ == "__main__":
    raise SystemExit(main())

# AGENTS.md —— Citecoon

学术调研工具集：一个 MCP 服务 + 一个 skill。**工具面零模型推理**（检索、引文图谱、原文获取、带校验的格式化落盘、渲染导出），选题判断、知识归纳、评审写作全部由调用它的外层 agent 完成。改代码时守住这条分层：不要往工具里塞 LLM 调用或子 agent。

## 改完必须验证

```bash
cd citecoon && .venv/bin/python -m pytest tests/ -q     # 必须全绿
```

- **不要假设 `python` / `python3` / `uv` 可用**——解释器只用项目内的 `.venv/bin/python`。
- 改了 `src/citecoon/` 下任何源码都要跑一次；失败先修再继续，不要带着红灯往下做。
- 改了工具的参数 schema 或 `_PURE_TOOLS` 白名单，还要 `.venv/bin/python -c "import citecoon.mcp.server"` 确认能导入，并提醒用户重启 MCP 连接才生效。

## 目录

| 路径 | 内容 |
|---|---|
| `src/citecoon/mcp/server.py` | MCP 入口。`_PURE_TOOLS` 白名单决定哪些工具对外可见；`_INSTRUCTIONS` 是每会话必达的编排纪律 |
| `src/citecoon/tools/` | 工具实现，按主题分文件（seeds/graph/read/cards/writing/report/roadmap/codebase/batch_read/notes/config/health） |
| `src/citecoon/core/` | 会话状态（workspace）、数据模型（models）、配置（config）、缓存（cache）、产物布局中枢（layout）、图预处理（graphprep） |
| `src/citecoon/sources/` | 数据源：openalex/arxiv/semanticscholar/crossref，经 multi.py 聚合降级；resolve.py 是标题→论文确定性解析 |
| `src/citecoon/render/` | 导出：graphml/markdown/trace（export）、交互可视化（viz）、PDF（pdf） |
| `skill/citecoon/` | skill 手册：`SKILL.md` 入口 + `references/`（building / knowledge / report-loop / operations） |
| `tests/` | 纯逻辑回归 + evals 行为级回归 |
| `projects/<课题>/` | 研究产物（见下），不是源码 |

### 产物分层布局（projects/<课题>/）

```
01_graph/     result.json、citation_network.graphml、roadmap.graphml、view.html
02_reading/   papers/ 原文 · cards/ 卡片+_themes.json 母题
03_thinking/  ideas/ 研究想法
04_writing/   reports/<报告名>/{report.md,review.md,*.pdf} · drafts/ · report.md
05_code/      codebases/ repo卡 · experiments/
_runtime/     trace.log、result.autosave.json、CHANGELOG.md（机器产物，人不用看）
```

布局常量与新旧兼容逻辑全部收敛在 `core/layout.py`——加目录/找文件都走它，不要在工具里硬编码路径。

### 用户级文件（家目录）

- `~/.citecoon.env` — 密钥（OPENALEX_API_KEY 等），仓库外存放
- `~/.citecoon/cache/` — 检索 API 的 SQLite 缓存
- `~/.citecoon/last_project` — MCP 重启后恢复最后活跃项目

## 约定

- **质量约束优先做成工具闸门**（校验+拒绝+可执行的修法提示），其次才是 instructions 文案。实测反复证明：只写在文档里的纪律会被无视，工具侧拒绝 100% 有效。
- 拒绝消息要说清"缺什么、怎么改"，不要只说不合格。
- 中文写 UI 文案、日志、错误消息和注释；注释解释**为什么**（尤其踩过的坑），不要复述代码在做什么。
- 新增工具后要同步三处：`tools/__init__.py` 注册、`mcp/server.py` 的 `_PURE_TOOLS` 白名单、必要时 `_INSTRUCTIONS` 与 skill 的 `references/`。漏白名单的工具外层看不见。

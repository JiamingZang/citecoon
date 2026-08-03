# Citecoon

学术调研工具集：**MCP 纯工具面 + skill 手册 + 纯文本产物库**。

工具只做机械操作（检索、引文图谱、原文获取、带校验落盘、渲染导出），零模型推理；
选题判断、知识归纳、评审写作全部由外层 agent（Qoder/Claude 等）完成。
工具可弃，语料不朽——所有产物都是纯文本，git 可管、可读、可迁移。

## 组成

| 部件 | 说明 |
|---|---|
| MCP server（`citecoon mcp`） | 长驻进程，持有内存引文图谱，暴露 ~40 个确定性工具 |
| skill（`skill/citecoon/`） | 外层 agent 的打法手册：建图/精读/母题/idea/报告闭环 |
| 产物库（`projects/<课题>/`） | 每个研究方向一个目录，分层布局，长期累积 |

## 安装

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/citecoon doctor        # 体检：env/缓存/残留进程
```

密钥放 `~/.citecoon.env`（仓库外）：`OPENALEX_API_KEY` 等。

MCP 注册（Qoder 为例）：

```json
"citecoon": {
  "command": "<repo>/.venv/bin/citecoon",
  "args": ["mcp", "-p", "<初始项目目录绝对路径>"]
}
```

`-p` 只是初始项目，会话里 `use_project` 随时切换/新建。

skill 安装：把 `skill/citecoon` 链接到 skills 目录（保持目录名 citecoon）。

## 产物布局

```
projects/<课题>/
├── 01_graph/     result.json、graphml、view.html
├── 02_reading/   papers/ 原文 · cards/ 卡片+母题
├── 03_thinking/  ideas/ 研究想法
├── 04_writing/   reports/（report.md + review.md + PDF）· drafts/
├── 05_code/      codebases/ repo卡 · experiments/
└── _runtime/     trace、autosave（机器产物）
```

## CLI

```bash
citecoon mcp -p <项目目录>      # MCP stdio server
citecoon status -p <项目目录>   # 项目推进度一屏全览
citecoon doctor                 # 环境体检
citecoon pdf <报告名> -p <项目> # 存量报告补导 PDF
```

## 开发

改代码先读 [AGENTS.md](AGENTS.md)；改完跑 `.venv/bin/python -m pytest tests/ -q`。

---
name: citecoon
description: Citecoon 科研调研全流程编排——用 citecoon MCP 工具（文献检索/引文图谱/原文获取/结构化落盘/产物导出）做领域调研；分析、判断与写作由你完成。当用户要调研某个学术方向、追论文谱系、建引文图谱、沉淀论文卡片与研究想法、写研究报告、跟踪领域新进展、或中断后恢复一个调研项目时使用。不适用：普通的单篇论文问答（直接 WebSearch/读 PDF 即可）、与文献调研无关的写作任务。
---

# Citecoon 科研调研手册（引文大亨：从一篇论文攒到一座图谱帝国）

Citecoon 是一套学术研究工具集：论文检索（OpenAlex + arXiv 双源）、引文图谱构建、原文获取、结构化知识落盘、报告导出。每个研究方向对应一个项目目录（`projects/<课题>/`），产物按研究流程分层长期累积：`01_graph/`（引文图谱）· `02_reading/`（原文 papers/ + 卡片 cards/）· `03_thinking/`（研究想法 ideas/）· `04_writing/`（报告 reports/）· `05_code/`（repo 卡 + 实验）· `_runtime/`（机器产物）。

**分工**：citecoon 工具只做机械操作（检索、图谱维护、格式化落盘、渲染导出），不做任何分析判断；**选题判断、谱系抽取、知识归纳、评审写作全部由你完成**，联网查证用你原生的 WebSearch，多视角评审可起你自己的子 agent。落盘工具自带格式模板与引用校验，不合规的写入会被拒绝并说明原因。

## 场景路由（按需读对应文档，不用全读）

| 用户要什么 | 打法 | 细节 |
|---|---|---|
| 调研新方向 / 建引文图谱 | 开项目 → 检索定种子 → 扩展引文网络 → 接谱系 | [references/building.md](references/building.md) |
| 精读论文 / 沉淀知识 / 提研究想法 | 读一篇沉一篇 → 归纳母题 → 写带证据锚点的 idea | [references/knowledge.md](references/knowledge.md) |
| 写报告（任何一种） | **先分流**：领域导读 / 轻量综述 / 技术可行性报告 | [references/report-loop.md](references/report-loop.md) |
| 跟踪新进展（增量刷新） | 补检索新论文 → 落卡 → 母题重算 → 更新受影响的 idea | [references/knowledge.md](references/knowledge.md) |
| 中断后恢复 / 新会话接手一个项目 | **先重建概览再动手**：状态盘点 → 产物对账 → 从最后落盘处续跑 | [references/operations.md](references/operations.md) |
| 查项目状态 / 环境问题 | `project_status` 工具；终端 `citecoon doctor` | — |

## 全局规矩（四条）

1. **落盘只走对应工具**（fill_idea_card / save_themes / fill_research_idea / fill_domain_report / save_report_section / save_review）：格式模板内置、引用会被校验——不要用文件工具裸写这些产物，也不需要参考旧项目的文件格式。
2. **调研收口顺序固定**：`fill_domain_report`（领域导读）→ `fill_roadmap`（演化路线图，不调则 roadmap 产物为空）→ `emit_result`（落盘渲染，必调）。
3. **状态先看工具再决策**：走到哪一步、还差什么，一律用 `project_status` / `list_idea_cards` / `list_themes` / `graph_summary` 取现成状态，不要翻对话历史自己数，也不要裸读项目目录。判断即时落盘——上下文随时会被压缩，落盘产物就是你的断点。
4. PDF 导出严格串行（渲染引擎不支持并发）；项目路径一律传绝对路径。

## 子 agent 委派（默认自己干，委派是例外）

委派判据只有一条：**这一步是否引入你拿不到的新信息或独立视角**（读整篇全文、独立对抗审校）。复述已有卡片、改写已有结论不委派。多 agent 协作的 token 开销约为单线流程的一个数量级以上，收益必须覆盖得起。委派契约与恢复协议见 [references/operations.md](references/operations.md)。

## MCP 注册

```json
"citecoon": {
  "command": "<citecoon 安装根目录>/.venv/bin/citecoon",
  "args": ["mcp", "-p", "<初始项目目录绝对路径>"]
}
```

`-p` 只是初始项目，会话里用 `use_project` 随时切换。工具面是纯确定性工具（零模型推理），全部默认可见；没有隐藏的"重型工具"档。

**连接失败自查三连**：`command` 路径是否存在（ls 验证）→ `-p` 目录是否存在 → 终端 `citecoon doctor` 体检。

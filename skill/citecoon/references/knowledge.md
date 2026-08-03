# 精读沉淀与研究想法

沉淀是精读的默认动作，不是可选项：读过不落卡等于白读，后续归纳和写作都会失去证据。

## 精读即沉淀

- `read_paper` / `read_fulltext` 拿原文。section 参数只用报错提示里列出的可用章节名，不要猜；取到的全文会自动存进项目的 `papers/`。
- **每篇读完立刻 `fill_idea_card` 落卡，再读下一篇**。卡片字段都要写实：问题 / 方法 / 核心假设 / 局限 / 实验设置。其中"局限"字段最重要——它是后续研究想法的直接原料。
- **落卡前先找关联**：`find_in_graph` / `list_idea_cards`（带关键词）查已有条目，新卡的方法/对比字段显式引用相关论文的 paper_id 或已有卡标题——知识网络靠写入时建的链接维持，模型不会自发回头补链。
- 覆盖目标：核心论文的卡片数不低于图谱论文数的三分之一。只挑两三篇代表作落卡，后面的归纳会失真。
- **批量精读（超过约 8 篇）注意上下文预算**：read_paper 的全文会直接进你的上下文，几十篇全读会撑爆。批量时改用分工模式——原文已自动存在 `papers/` 目录，起你自己的子 agent 分批读盘、按卡片字段产出要点，你只校对内容并调 `fill_idea_card` 落盘（落盘闸门仍在你手里；委派契约见 operations.md §2）。

**记录与整理分离**：落卡 / take_note 是即时追加证据；把观察归纳成母题或通用结论是独立的整理阶段，必须有多卡支撑（save_themes 会校验卡片引用）。单次实验/单篇论文的结果不许直升为通用结论。

## 母题归纳

卡片攒够后，用 `save_themes` 落盘你归纳的母题——跨论文的共性假设、路线分叉、结构性空白。每条含：主题陈述、类型、支撑卡片列表（会校验必须对得上已有卡片）、证据说明、未解张力。"张力"字段认真写：它直接孕育研究想法。跳过母题直接提想法，想法会缺乏证据链。

**重算前先 `list_themes`**：它是全量覆盖式写入，先读现有母题再决定强化/新增/删除；该工具还会标出失效的卡片引用（卡片改名或短名别名脱节），逐条换成真实 `paper_title` 再提交，否则 `save_themes` 会拒绝整次写入。

## 研究想法（idea）

只用 `fill_research_idea` 写，不要用文件工具裸写：

- 模板统一（Gap 来源 / 动机 / 核心假设 / 可检验预测 / 技术路线 / 最小实验设计）。
- Gap 来源每条必须带可校验的锚点——某条母题、某张卡片、或图谱内论文 id；没有锚点会被拒绝。
- 可检验预测要具体到指标、阈值、对照组；最小实验设计要含关键消融。
- 更新已有想法时保持标题不变（同标题就地覆盖）；改标题会产生重复文件，破坏后续报告环节的文件匹配。
- **出炉即轻量查重**：每个想法写完后，拆出核心技术组合词，用 `find_candidates`/`search_recent` + 你原生的 WebSearch 三路检索撞车一轮。撞上强相关工作 → 就地修改想法或在正文明写差异化；查重结论（查了什么、撞没撞、差异在哪）附进想法正文。完整的对抗评审（三个怀疑者+可执行性定级）留给技术可行性报告流程，这里不重复。
- 增量刷新中被实质性改动（结论或机制变了）的想法，传导到报告之前必须重跑一轮轻量查重——新的组合可能撞上新发表的工作。

## 增量刷新（跟踪领域新进展）

1. `search_recent` / `expand_frontier` 把新论文补进图谱
2. 新论文按上面的流程精读落卡
3. 先 `list_themes` 看现有母题与失效引用，再 `save_themes` 重算（全量覆盖式写入，新证据可能强化旧母题或催生新母题）
4. 受影响的想法用 `fill_research_idea` 同标题就地更新，正文加一段"更新记录"说明新证据强化/削弱/撞车了什么。影响面别靠记忆：用 `list_idea_cards` 带新论文的关键词（方法名/缩写）过滤出相关卡片与想法逐条核对。
5. 已有报告受影响时走报告修订路径（见 report-loop.md）

## 已废弃的路径与工具名（不要重建）

瘦身后的形态是「MCP 纯工具面 + 外层 agent 智能」。下列名称在历史版本里存在过，现已删除或不再暴露；凭旧记忆调用会失败，也不要按它们重建目录或流程：

- 旧形态（run/tui/交互式终端）已删除。入口只有 `citecoon mcp`（MCP 服务）与 `status/doctor/pdf` 运维子命令。
- `web_search`、`study_codebase`、`write_tech_report`、`write_report`、`review_report`、`revise_report`、`propose_idea`、`critique_idea`、`synthesize_themes`、`refresh_idea`、`link_frontier`、`mine_surveys`、`find_founding`、`select_roadmap`、`plan_experiment`/`run_experiment`/`collect_results`、`ask_user` —— 这些内嵌 LLM 或子 agent 的工具不在 MCP 工具面上。对应能力改由你自己完成：联网用你原生的 WebSearch；读仓库用 Bash 浅克隆 + Read/Grep 后 `fill_repo_card`；谱系接线用 `read_paper` 抽标题 + `wire_predecessors`；综述挖掘用 `find_surveys`；路线图用 `fill_roadmap`；想法与报告用 `fill_research_idea` / `save_report_section`；跟用户对话直接说，不必调工具。
- 实验结果不再由工具收集：自己跑完实验后把结果写进 `experiments/<实验名>/result.json`，`write_draft` 只认这个路径。

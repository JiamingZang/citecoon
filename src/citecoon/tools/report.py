"""Tech-report writer: a ReAct sub-agent that consults sources while writing.

v1 generated the report via three blind one-shot calls over a static material
bundle — the writer could never look anything up mid-flight, which is exactly
where detail loss and hallucination crept in. v2 spawns a Qoder sub-agent with
Read/Grep over the project directory (cards/ papers/ codebases/ ideas/ ...):
before writing each chapter it reads the relevant sources, verifies numbers and
code facts in place, and does a self-review pass at the end. The chapter
skeleton stays mandated so structure doesn't wander.
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import date, datetime

from pydantic import BaseModel, Field

from stirrup import Tool, ToolResult, ToolUseCountMetadata

from ..context import RunContext
EXPERIMENTS_DIR = "experiments"
PLAN_FILE = "PLAN.md"
IDEAS_DIR = "ideas"

REPORTS_DIR = "reports"
# 目录约定：每份报告一个目录 reports/<slug>/，里面固定文件名：
#   report.md / review.md / <slug>.pdf / context.md / logs/{write,review,revise}.log
REPORT_MD = "report.md"
REVIEW_MD = "review.md"
LOGS_SUBDIR = "logs"


def report_log_dir(rdir) -> "os.PathLike":
    d = rdir / LOGS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_report(reports, name: str):
    """模糊匹配 reports/ 下的报告目录，返回其 report.md；唯一命中返 (Path, None)，否则 (None, 错误文案)。

    匹配前把标点/下划线/空白归一化：实测 agent 用正常标点标题（如『报告：X —— Y』）
    对 slug 化目录名匹配失败后，学到「要用下划线名」并从此把 slug 当标题用，PDF 封面被污染。"""
    def _norm(s: str) -> str:
        return re.sub(r"[^\w\u4e00-\u9fff]+|_+", "", s.lower())

    key = _norm(name.removesuffix(".md").removesuffix("/"))
    dirs = [d for d in reports.iterdir() if d.is_dir() and (d / REPORT_MD).is_file()] if reports.is_dir() else []
    hits = [d for d in dirs if key and key in _norm(d.name)]
    if len(hits) != 1:
        return None, f"报告匹配到 {len(hits)} 个（需唯一）。现有：{sorted(d.name for d in dirs)}"
    return hits[0] / REPORT_MD, None

_MAX_TURNS = 120
_WALL_SECONDS = 2400

_SKELETON = """# 1. 背景与动机
  1.1 问题陈述（要解决什么、瓶颈的量化表现）
  1.2 相关工作（按路线分组，只引用 cards/ 里的真实论文）
  1.3 根本性分析（把现有方法的具体失效机制说清楚：它缺什么信号、错在哪一步、为什么修不了。
      论证视角由这个失效机制本身决定，两三段最短路径说透即可；公式只在承担论证功能时出现）
# 2. 方法
  把 idea 拆成 2-3 个互补贡献（Contribution 1/2/3），每个含设计动机、技术细节
  （公式/接口约定/伪代码）、与现有系统的衔接。代码级细节只能来自 codebases/ 的 repo 卡。
# 3. 实验计划
  3.1 评估指标（当前值/目标值/改进幅度表格） 3.2 消融矩阵（含 oracle 上界与 negative control）
  3.3 基线方法 3.4 数据集要求与预处理 3.5 评估协议 3.6 计算资源估算表
# 4. 可行性评估
  4.1 实现复杂度（与更轻替代路线对比，量化成倍数） 4.2 外部依赖风险表 4.3 错误传播风险
      （含最坏情况分析：新机制失效时系统退化到什么下界？能否结构性保证可回退到基线，
      如加法式改造/zero-init/fallback 路径？可回退设计要写进方法而非只在风险节提及）
  4.4 性能/成本量化（推理开销拆到逐组件耗时预算表，给出总开销、对吞吐的影响与优化后预期）
  4.5 时间线里程碑表 4.6 综合判级 + 两条决策路径建议
# 5. 结论（一段话：方案、预期收益、主要风险、时间框架与目标会议）"""


def _ok(content: str) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content=content, metadata=ToolUseCountMetadata(), success=True)


class ExportReportPdfParams(BaseModel):
    report_file: str = Field(description="reports/ 下的报告 md 文件名（可模糊匹配）。")


def build_report_tools(ctx: RunContext) -> list[Tool]:

    class SaveReportParams(BaseModel):
        title: str = Field(description="报告标题（中文）。目录名由它生成；同标题就地覆盖 report.md。")
        body_markdown: str = Field(description="报告正文 markdown（不含一级标题和头部元信息——工具会加）。引用图谱论文时带 paper_id，引用 idea/卡片时点名文件。【边界】本工具只适合轻量综述/方向盘点；用户要技术可行性报告（逐段查证+待验证标注+审校闭环）时不要用它自由发挥，改走 write_tech_report（内层 ReAct 写作）。")
        basis: str = Field(default="", description="依据说明，如『4 个 idea + 10 张卡片 + 6 条母题』（进头部元信息）。")

    def save_report(p: SaveReportParams) -> ToolResult[ToolUseCountMetadata]:
        """报告落盘（零 LLM）：一报告一目录结构内置——外层 agent 不必再考古老项目
        （实测写报告前先 find 老 reports/ 看结构，格式知识应在工具里）。"""
        slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", p.title)[:60].strip("_")
        rdir = ctx.project_dir(REPORTS_DIR) / slug
        rdir.mkdir(parents=True, exist_ok=True)
        header = (
            f"# {p.title}\n"
            f"\n> 研究方向报告 · {date.today().isoformat()} · 外层 agent 撰写（save_report 装配）"
            + (f" · 依据: {p.basis}" if p.basis else "")
            + "\n\n"
        )
        path = rdir / REPORT_MD
        existed = path.exists()
        body = p.body_markdown.lstrip()
        # 调用方若自带一级标题则剥掉，防止双标题
        if body.startswith("# "):
            body = body[body.find("\n") + 1 :].lstrip()
        # 缩水护栏（复刻 revise 的还原保护）：修订版比现存版短 30% 以上视为异常，
        # 拒绝覆盖——防止"修订"变成大面积删稿；确要精简需先说明再重调
        if existed:
            old_len = len(path.read_text(encoding="utf-8"))
            if old_len > 2000 and len(body) < old_len * 0.7:
                return _ok(
                    f"save_report 拒绝覆盖：新稿 {len(body)} 字符比现存 {old_len} 缩水超 30%——"
                    f"修订应就地改进而非删稿。确要精简请在正文首行注明『[精简重写]』再重调。"
                )
            if body.startswith("[精简重写]"):
                body = body[len("[精简重写]"):].lstrip()
        # 一轮 dump 闸门：技术可行性报告必须走 save_report_section 逐节写（写前查证），
        # 一次性长文只可能是自由发挥（实测一轮写完的报告查证密度掉一个量级）
        if len(body) > 12000:
            return _ok(
                f"save_report 拒绝：正文 {len(body)} 字符——一轮成稿超出轻量综述范围。"
                f"技术可行性报告请改用 save_report_section 逐节写：每节先取证（read_fulltext/"
                f"cards/图谱）再落盘，一轮一节。"
            )
        path.write_text(header + body, encoding="utf-8")
        ctx.ws.trace.add("agent", "report", f"save_report {slug} | {len(body)} chars")
        return _ok(
            f"报告已{'就地覆盖' if existed else '新建'}：{REPORTS_DIR}/{slug}/{REPORT_MD}"
            f"（{len(body)} 字符）。接着用 export_report_pdf('{slug}') 出 PDF（串行，别并发）。"
        )

    class SaveReviewParams(BaseModel):
        report_title: str = Field(description="被审校的报告标题（模糊匹配 reports/ 下唯一目录，标点/下划线不敏感）。")
        review_markdown: str = Field(description="审校清单正文：按 P0（编造/事实错误，必须修）/ P1（论证缺口）/ P2（表述）分级，每条给出位置、问题、修法建议。只审不改。")
        verdict: str = Field(description="总评结论：如『需修订后发布』/『可发布』/『结构性缺陷建议重写』。")
        scores: dict[str, int] = Field(
            default_factory=dict,
            description="可选结构化评分（0-10 整数），建议四维：新颖性/技术成立性/证据充分性/逻辑连贯性——"
            "跨报告横向对比与流程迭代的回归指标（AI-Scientist reviewer rubric 式）。",
        )

    def save_review(p: SaveReviewParams) -> ToolResult[ToolUseCountMetadata]:
        reports = ctx.project_dir(REPORTS_DIR)
        md, err = find_report(reports, p.report_title)
        if err:
            return _ok(err)
        rdir = md.parent
        header = (
            f"# 审校清单：{rdir.name}\n"
            f"\n> 对抗性审校 · {date.today().isoformat()} · 结论：{p.verdict} · 只审不改，"
            f"逐条核实来源后用 save_report 就地修订\n\n"
        )
        if p.scores:
            header += (
                "| 维度 | 评分 (0-10) |\n|---|---|\n"
                + "\n".join(f"| {k} | {min(10, max(0, v))} |" for k, v in p.scores.items())
                + "\n\n"
            )
        (rdir / REVIEW_MD).write_text(header + p.review_markdown.lstrip(), encoding="utf-8")
        ctx.ws.trace.add("agent", "review", f"{rdir.name} | {p.verdict}" + (f" | scores={p.scores}" if p.scores else ""))
        return _ok(
            f"审校清单已落盘：{REPORTS_DIR}/{rdir.name}/{REVIEW_MD}（结论：{p.verdict}）。"
            f"P0/P1 逐条核实来源后用 save_report 同标题覆盖修订，改完 export_report_pdf 重导。"
        )


    # 技术可行性报告固定五章骨架（对齐 Motion-Aware 参考形态）：四问是校验维度不是章名——
    # 实测 agent 把『要不要做/坏了怎么办』直接写成章标题，产出与参考报告风格完全不对齐
    _CHAPTERS = {
        1: ("背景与动机", "Background & Motivation"),
        2: ("方法", "Method"),
        3: ("实验计划", "Experiments Plan"),
        4: ("可行性评估", "Feasibility Assessment"),
        5: ("结论", "Conclusion"),
    }

    class SaveSectionParams(BaseModel):
        report_title: str = Field(description="报告标题（首节调用会创建 reports/<标题>/ 目录；后续节传同一标题）。建议中英双标题，如『技术可行性报告：XXX —— YYY』。")
        section_title: str = Field(description="小节标题，必须带 N.M 编号归属五章骨架（如『1.2 相关工作』『4.5 时间线与里程碑』）：1 背景与动机 / 2 方法 / 3 实验计划 / 4 可行性评估 / 5 结论（结论可传『5. 结论』章级直书）。章标题由工具自动生成，同编号节就地替换（修订用）。")
        section_markdown: str = Field(description="本节正文（不含节标题行；节内更细层级用 #### 或粗体，不许用 ##/###）。单节 300-6000 字符：多节打包或不足 300 字符都会被拒。")
        evidence_refs: list[str] = Field(description="本节论断的证据引用，至少 1 条可校验：图谱 paper_id / cards/xx.json / papers/ 原文文件名。写作前应先取证再写本节。")

    def save_report_section(p: SaveSectionParams) -> ToolResult[ToolUseCountMetadata]:
        """技术可行性报告的逐节落盘（零 LLM）：五章骨架内置，小节按 N.M 编号归位，
        章标题/排序/层级由工具组装——格式知识在工具里，外层只提供内容与证据。"""
        m = re.match(r"^(\d)(?:\.(\d{1,2}))?[\s.、]*(.*)$", p.section_title.strip())
        if not m or int(m.group(1)) not in _CHAPTERS:
            skel = " / ".join(f"{n} {zh}" for n, (zh, _) in _CHAPTERS.items())
            return _ok(
                f"save_report_section 拒绝：节标题『{p.section_title}』没有 N.M 编号或章号越界。"
                f"固定五章骨架：{skel}——小节标题形如『1.2 相关工作』，结论可传『5. 结论』。"
            )
        ch, sub = int(m.group(1)), int(m.group(2)) if m.group(2) else 0
        sec_name = m.group(3).strip() or _CHAPTERS[ch][0]
        # 四问是校验维度不是标题（实测重写后 '要不要做' 仍以小节/小标题潜回报告）
        _FOUR_Q = re.compile(r"要不要做|怎么做|代价多少|坏了怎么办")
        if _FOUR_Q.search(sec_name):
            return _ok(
                f"save_report_section 拒绝：『{sec_name}』是决策四问，不是参考形态的标题——"
                "要不要做写进 1.2 相关工作对比与 4 章综合可行性判断，怎么做写进 2 章，"
                "代价写进 3 章资源估算，风险处置写进 4 章风险小节。换成参考形态的小节名重交。"
            )
        body = p.section_markdown.strip()
        # 正文首行若是与节标题重复的标题行，自动剥掉——实测 agent 传了 section_title
        # 还在正文里再写一遍，拼出双重标题
        first = body.split("\n", 1)[0].strip()
        if first.startswith("#"):
            t = re.sub(r"^#+\s*|[\d.\s]+", "", first)
            ts = re.sub(r"[\d.\s]+", "", sec_name)
            if t and ts and (t in ts or ts in t):
                body = body.split("\n", 1)[1].strip() if "\n" in body else ""
        # 打包闸门：##/### 是骨架层级，正文里出现说明一次塞了多节（实测 4 次调用塞完四章）；
        # 带编号的 #### 同罪——实测重写后 agent 改用 '#### 2.1/2.2/...' 自建编号继续整章打包；
        # 四问短语做小标题也拦（'#### 要不要做' 潜回）；无编号的 #### 细分层级合法
        if re.search(r"^###?[^#]", body, re.M):
            return _ok(
                "save_report_section 拒绝：正文里含 ##/### 标题——一轮只写一节，每节单独调用并单独取证；"
                "节内更细层级用 `####` 或粗体。把这段拆成多次调用重交。"
            )
        if re.search(r"^####\s*\d+(\.\d+)*\s", body, re.M):
            return _ok(
                "save_report_section 拒绝：正文里用 `#### N.M` 自建编号——这是整章打包的变体。"
                "每个编号小节单独调用 save_report_section（section_title 传 N.M）；"
                "节内无编号的 #### 细分小标题（如 '#### 问题定义'）才合法。"
            )
        for hm in re.finditer(r"^####\s*(.+)$", body, re.M):
            if _FOUR_Q.search(hm.group(1)):
                return _ok(
                    f"save_report_section 拒绝：小标题『{hm.group(1).strip()}』是决策四问措辞，参考形态不这么写——"
                    "结论性判断归 4 章『综合可行性判断』（评级+有利不利因素+建议决策路径），改措辞后重交。"
                )
        if len(body) < 300:
            return _ok(
                f"save_report_section 拒绝：本节仅 {len(body)} 字符——决策文档不存在 300 字以下的正经节。"
                "参考形态要求：根因分析到第一性原理、方法精确到可直接改代码的粒度、实验矩阵含消融/基线/算力量化表。"
                "先取证（read_fulltext/cards）再展开写。"
            )
        if len(body) > 6000:
            return _ok(f"本节 {len(body)} 字符超上限 6000——拆成更小的节，每节先取证再写。")
        # 证据校验：paper_id 在图 / 卡片文件存在 / 原文文件存在 / repo 卡存在
        card_names = {f.name.lower() for f in ctx.project_dir("cards").glob("*.json")}
        paper_names = {f.name.lower() for f in ctx.project_dir("papers").glob("*.md")}
        repo_names = {f.name.lower() for f in ctx.project_dir("codebases").glob("*.md")}
        graph_ids = {pid.lower() for pid in ctx.ws.papers}
        valid = 0
        for ref in p.evidence_refs:
            r = ref.strip().lower()
            ids = re.findall(r"(w\d{6,}|arxiv:\d{4}\.\d{4,5})", r)
            if any(i in graph_ids for i in ids):
                valid += 1
            elif any(n in r for n in card_names) or any(n in r for n in paper_names) or any(n in r for n in repo_names):
                valid += 1
        if valid == 0:
            return _ok(
                "save_report_section 拒绝：evidence_refs 没有一条能对上图谱 paper_id / cards/ 卡片 / papers/ 原文 / codebases/ repo 卡。"
                "先 read_fulltext/list_idea_cards/fill_repo_card 取证，再写本节。"
            )
        slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", p.report_title)[:60].strip("_")
        rdir = ctx.project_dir(REPORTS_DIR) / slug
        rdir.mkdir(parents=True, exist_ok=True)
        path = rdir / REPORT_MD
        # 解析现有小节按 N.M 归位，整文重组——章标题/排序/层级由工具生成，
        # 产出结构必然与参考形态一致（之前靠 agent 自由发挥，实测风格完全跑偏）
        secs: dict[tuple[int, int], tuple[str, str]] = {}
        created = date.today().isoformat()
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if (m2 := re.search(r"· (\d{4}-\d{2}-\d{2})", old[:200])):
                created = m2.group(1)
            for mm in re.finditer(r"^### (\d)\.(\d{1,2}) (.+?)\n(.*?)(?=^### |^## |\Z)", old, re.M | re.S):
                secs[(int(mm.group(1)), int(mm.group(2)))] = (mm.group(3).strip(), mm.group(4).strip())
            for mm in re.finditer(r"^## (\d)\.[^\n]*\n(.*?)(?=^### |^## |\Z)", old, re.M | re.S):
                b = mm.group(2).strip()
                if b:
                    secs[(int(mm.group(1)), 0)] = ("", b)
        verb = "已替换" if (ch, sub) in secs else "已追加"
        # 证据引用随节落盘（HTML 注释，渲染不可见）：此前校验完就丢，出版时无法审计
        # 覆盖率（STORM 原则：重要论断 ≥2 独立来源）；重交时先剥旧注释防叠加
        body = re.sub(r"\n?<!-- refs:.*?-->", "", body, flags=re.S).strip()
        body += "\n\n<!-- refs: " + "; ".join(r.strip() for r in p.evidence_refs) + " -->"
        secs[(ch, sub)] = (sec_name, body)
        # H1 去 slug 化：agent 把目录名当标题传时（实测 save_review 匹配失败后学到的坏习惯），
        # 下划线还原成空格，免得 PDF 封面顶着 slug
        title_display = re.sub(r"_+", " ", p.report_title).strip()
        out = [f"# {title_display}", "", f"> 技术可行性报告（逐节装配·写前查证） · {created}", ""]
        for c in sorted({k[0] for k in secs}):
            zh, en = _CHAPTERS[c]
            out += [f"## {c}. {zh} ({en})", ""]
            for key in sorted(k for k in secs if k[0] == c):
                t, b = secs[key]
                if key[1] == 0:
                    out += [b, ""]
                else:
                    out += [f"### {key[0]}.{key[1]} {t}", "", b, ""]
        content = "\n".join(out).rstrip() + "\n"
        path.write_text(content, encoding="utf-8")
        ctx.ws.trace.add("agent", "report", f"section {verb}: {slug} / {ch}.{sub} {sec_name}")
        gaps = _skeleton_gaps([(k[0], v[0]) for k, v in secs.items()], {k[0] for k in secs}, content)
        gap_note = f"骨架待补：{'；'.join(gaps)}。" if gaps else "骨架已齐。"
        thin = f"【本节 {len(body)} 字符偏薄——参考形态要求代码级粒度/量化表，考虑取证后同编号重交加深】" if len(body) < 800 else ""
        return _ok(
            f"节{verb}：{REPORTS_DIR}/{slug}/{REPORT_MD}（{ch}.{sub or ''} {sec_name}，现 {len(secs)} 节，{len(content)} 字符）。{thin}{gap_note}"
            f"继续下一节前先为它取证；全部写完 → save_review 审校 → 修订 → export_report_pdf。"
        )

    def _skeleton_gaps(titles: list[tuple[int, str]], chapters: set[int], text: str = "") -> list[str]:
        """参考形态的必备小节检查（Motion-Aware 报告：环节一个不能少）。传 text 时额外查
        深度维度：方法章单节成章/无代码块、缺指标目标表（实测产物 vs 参考的三处差距）。"""
        def has(ch: int, kw: str) -> bool:
            return any(c == ch and re.search(kw, t) for c, t in titles)
        gaps = []
        if not has(1, "相关工作"):
            gaps.append("1 章『相关工作』")
        if not has(1, "根本|第一性|问题陈述"):
            gaps.append("1 章『问题陈述/根本性分析』")
        if 2 not in chapters:
            gaps.append("2 章『方法』")
        elif sum(1 for c, _ in titles if c == 2) < 2:
            gaps.append("2 章需 ≥2 小节（参考形态：总览+逐贡献/推理流程，方法章应占全文最大篇幅）")
        if not has(3, "消融"):
            gaps.append("3 章『消融实验设计』")
        if not has(3, "资源|算力"):
            gaps.append("3 章『计算资源估算』")
        if not has(4, "风险"):
            gaps.append("4 章『风险』")
        if not has(4, "时间线|里程碑"):
            gaps.append("4 章『时间线与里程碑』")
        if not has(4, "综合可行性|可行性判断"):
            gaps.append("4 章『综合可行性判断』")
        if 5 not in chapters:
            gaps.append("5 章『结论』")
        if text:
            ch2 = re.search(r"^## 2\..*?(?=^## |\Z)", text, re.M | re.S)
            if ch2 and "```" not in ch2.group(0):
                gaps.append("2 章『代码块』（伪代码/改动 diff/关键实现片段，参考形态精确到可改代码粒度）")
            if "目标值" not in text and not re.search(r"目标[\s|]*\|", text):
                gaps.append("『评估指标现值/目标值表』（参考形态：每个指标当前值→目标值，如 60.21→68+）")
        return gaps

    async def export_report_pdf(p: ExportReportPdfParams) -> ToolResult[ToolUseCountMetadata]:
        from ..render.pdf import report_md_to_pdf

        reports = ctx.project_dir(REPORTS_DIR)
        md, err = find_report(reports, p.report_file)
        if err:
            return _ok(err)
        # 审校前置：逐节装配的技术可行性报告没过 save_review 不给出版（闭环第 4 步不可跳）
        head = md.read_text(encoding="utf-8")[:200]
        review = md.parent / REVIEW_MD
        if "逐节装配" in head:
            # 骨架完整性闸门：参考形态的必备小节缺一不可（实测上一版无消融/无时间线/无综合判断）
            txt = md.read_text(encoding="utf-8")
            titles = [(int(a), t) for a, t in re.findall(r"^### (\d)\.\d{1,2} (.+)$", txt, re.M)]
            chapters = {c for c, _ in titles} | {int(a) for a in re.findall(r"^## (\d)\.", txt, re.M)}
            gaps = _skeleton_gaps(titles, chapters, txt)
            if gaps:
                return _ok(
                    "export_report_pdf 拒绝：骨架不完整，缺——" + "；".join(gaps)
                    + "。参考形态这些环节一个不能少，取证后用 save_report_section 补齐再出版。"
                )
        if "逐节装配" in head and not review.is_file():
            return _ok(
                f"export_report_pdf 拒绝：{md.parent.name} 是技术可行性报告但还没有 review.md——"
                f"先起对立视角子 agent 审校（save_review），P0/P1 修完再出版。"
            )
        # 修订未做闸门：审校结论要求修订、但审校之后报告一字没改过（mtime 对比），
        # 说明第 5 步被跳——实测三份报告审校全判『需修订』后直奔出版
        if review.is_file():
            rtxt = review.read_text(encoding="utf-8")[:400]
            needs_fix = any(k in rtxt for k in ("需修订", "建议重写", "结构性缺陷"))
            if needs_fix and md.stat().st_mtime <= review.stat().st_mtime:
                return _ok(
                    f"export_report_pdf 拒绝：审校结论要求修订，但 report.md 在审校之后没改过一字——"
                    f"P0/P1 逐条核实后用 save_report_section 同标题节替换修订，改完再出版。"
                )
        try:
            pdf = await asyncio.to_thread(report_md_to_pdf, md)
        except Exception as e:  # noqa: BLE001
            return _ok(f"PDF 导出失败：{e}")
        size_kb = pdf.stat().st_size // 1024
        # 覆盖率审计（不阻断）：仅单一来源支撑的节列清单——重要论断应 ≥2 独立来源
        cov = []
        txt2 = md.read_text(encoding="utf-8")
        for mm in re.finditer(r"^### (\d\.\d{1,2}) (.+?)\n(.*?)(?=^### |^## |\Z)", txt2, re.M | re.S):
            rm = re.search(r"<!-- refs: (.*?)-->", mm.group(3), re.S)
            refs = {r.strip() for r in rm.group(1).split(";") if r.strip()} if rm else set()
            if len(refs) <= 1:
                cov.append(f"{mm.group(1)} {mm.group(2).strip()}（{len(refs)} 条）")
        cov_note = (
            f"\n覆盖率提示：以下节仅单一来源支撑——{'；'.join(cov)}。"
            "重要论断建议 ≥2 独立来源，修订时补 evidence_refs 同编号重交即可。"
            if cov else ""
        )
        return _ok(f"PDF 已导出：{pdf}（{size_kb} KB）{cov_note}")

    return [
        Tool[ExportReportPdfParams, ToolUseCountMetadata](
            name="export_report_pdf",
            description=(
                "把 reports/ 下的技术报告 markdown 导出为排版好的 PDF（A4、中文衬线正文、"
                "边框表格、KaTeX 公式），输出到同目录同名 .pdf。write_tech_report 会自动导，"
                "此工具用于对存量报告补导或重导。"
            ),
            parameters=ExportReportPdfParams,
            executor=export_report_pdf,
        ),
        Tool[SaveReportParams, ToolUseCountMetadata](
            name="save_report",
            description=(
                "把你写好的报告正文落盘到 reports/<标题>/report.md（零 LLM，一报告一目录结构内置，"
                "无需参考老项目）。同标题就地覆盖，带缩水护栏（修订缩水超 30% 拒绝，防删稿）。"
                "深度报告须先完成查重与逐段查证再落盘；落盘后 export_report_pdf 出 PDF。"
            ),
            parameters=SaveReportParams,
            executor=save_report,
        ),
        Tool[SaveReviewParams, ToolUseCountMetadata](
            name="save_review",
            description=(
                "把对抗性审校清单落盘到报告同目录 review.md（零 LLM，格式内置：P0 编造/事实错误、"
                "P1 论证缺口、P2 表述，附总评结论）。只审不改——修订走 save_report 同标题覆盖。"
                "审校者应与写作者视角对立（可起独立子 agent 扮演）。"
            ),
            parameters=SaveReviewParams,
            executor=save_review,
        ),
        Tool[SaveSectionParams, ToolUseCountMetadata](
            name="save_report_section",
            description=(
                "技术可行性报告的逐节落盘：一轮只写一节（正文含 `## ` 多节打包会被拒，小标题用 ###），"
                "写前必须取证（evidence_refs 至少 1 条对得上图谱 paper_id / cards 卡片 / papers 原文，对不上拒绝）。"
                "单节 300-6000 字符；同标题节就地替换（修订用）。"
                "这是技术可行性报告的唯一写入路径——save_report 会拒绝超长一轮成稿。"
                "全部节写完后 save_review 审校；没 review.md、或审校要求修订但报告未改时，export_report_pdf 都会拒绝出版。"
            ),
            parameters=SaveSectionParams,
            executor=save_report_section,
        ),
    ]

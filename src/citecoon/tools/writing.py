"""Draft-writing tools: assemble a paper draft with REAL references from the graph.

The bibliography is generated from OpenAlex/arXiv metadata already sitting in the
workspace — titles, authors, year, venue, DOI are real, so fabricated citations
(the classic failure mode of AI-written papers) are structurally impossible for
any paper that lives in the graph.
"""

from __future__ import annotations

import json
import re
from datetime import date

from pydantic import BaseModel, Field

from stirrup import Tool, ToolResult, ToolUseCountMetadata

from ..context import RunContext
EXPERIMENTS_DIR = "experiments"
RESULT_FILE = "result.json"

DRAFTS_DIR = "drafts"


def _ok(content: str) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content=content, metadata=ToolUseCountMetadata(), success=True)


def _bibkey(paper) -> str:
    first = (paper.authors[0].split()[-1] if paper.authors else "anon").lower()
    first = re.sub(r"[^a-z]", "", first) or "anon"
    word = re.sub(r"[^A-Za-z]", "", (paper.title or "x").split()[0]).lower() or "x"
    return f"{first}{paper.year or 'nd'}{word}"


def _bibtex_entry(paper) -> str:
    authors = " and ".join(paper.authors) if paper.authors else "Unknown"
    fields = [
        f"  title = {{{paper.title}}}",
        f"  author = {{{authors}}}",
    ]
    if paper.year:
        fields.append(f"  year = {{{paper.year}}}")
    if paper.venue:
        fields.append(f"  journal = {{{paper.venue}}}")
    if paper.doi:
        fields.append(f"  doi = {{{paper.doi}}}")
    aid = (paper.source_ids or {}).get("arxiv")
    if aid:
        fields.append(f"  eprint = {{{aid.split(':')[-1]}}}")
        fields.append("  archivePrefix = {arXiv}")
    return "@article{" + _bibkey(paper) + ",\n" + ",\n".join(fields) + "\n}"


class WriteDraftParams(BaseModel):
    title: str = Field(description="论文标题（英文）。")
    idea_file: str = Field(description="ideas/ 目录下定稿的 idea 文件名。")
    experiment_name: str | None = Field(
        default=None,
        description="实验名——正文的实验数字全部取自它的 result.json。没跑实验可省略（生成 position/proposal 草稿）。",
    )
    abstract: str = Field(description="摘要（英文，150-250 词）。")
    introduction: str = Field(description="引言（英文 markdown，动机+贡献列表）。")
    method: str = Field(description="方法章节（英文 markdown）。")
    experiments_section: str = Field(
        description="实验章节（英文 markdown）。数字必须来自 experiments/<实验名>/result.json 的真实结果（你自己跑完实验后用文件工具写入该路径），禁止编造；"
        "引用文献用 [@bibkey] 占位（bibkey 见工具返回的 bibliography）。"
    )
    conclusion: str = Field(description="结论与局限（英文 markdown）。")
    cite_paper_ids: list[str] = Field(
        description="需要引用的论文 id（必须在图谱里，工具会为它们生成真实 bibtex）。"
    )


class StageIn(BaseModel):
    name: str = Field(description="阶段名（中文，如「未见物体挑战」）。")
    period: str | None = Field(default=None, description="时间段，如「2023–2024」。")
    summary: str = Field(default="", description="该阶段 2-3 句中文概述。")
    papers: list[str] = Field(default_factory=list, description="该阶段代表论文的图谱 paper_id。")
    headline: str | None = Field(default=None, description="一句话阶段题记（可选）。")


class FillDomainReportParams(BaseModel):
    """领域脉络报告的全部智能内容由调用方（外层 agent）提供，本工具零 LLM。"""

    tldr: str = Field(description="一分钟入门：写给完全不懂的人的领域说明（中文）。")
    core_idea: str = Field(default="", description="核心思想一句话。")
    narrative: str = Field(description="领域综述叙事：从起源到前沿的发展弧线（中文，1-3 段）。")
    founding_papers: list[str] = Field(default_factory=list, description="奠基论文 paper_id（1-4 篇，你基于图谱+精读判断）。")
    stages: list[StageIn] = Field(default_factory=list, description="发展阶段分期。")
    main_line: list[str] = Field(default_factory=list, description="主线脉络：从源头到前沿的 paper_id 链。")
    must_read: list[str] = Field(default_factory=list, description="必读清单 paper_id。")
    must_read_reasons: dict[str, str] = Field(default_factory=dict, description="paper_id -> 一句话必读理由。")
    gaps: list[str] = Field(default_factory=list, description="开放问题/研究空白（中文短句）。")
    prerequisites: list[str] = Field(default_factory=list, description="前置知识清单。")
    glossary: dict[str, str] = Field(default_factory=dict, description="关键术语 -> 白话定义。")
    getting_started: list[str] = Field(default_factory=list, description="如何入手：带 paper_id 的行动步骤。")
    cover_title: str | None = Field(default=None, description="报告封面标题（可选）。")
    cover_blurb: str | None = Field(default=None, description="封面题记一句（可选）。")


class ResearchIdeaParams(BaseModel):
    """研究 idea 的全部智能内容由调用方提供；本工具只做锚点校验 + 统一模板渲染（零 LLM）。"""

    title: str = Field(description="idea 标题（中文，具体到机制，不要泛泛）。文件名由它生成，更新时保持标题不变即就地覆盖。")
    novelty_probe: str = Field(description="英文检索词（3-8 词，拆本 idea 的核心机制组合，如 'hierarchical RL unmasking order diffusion LM'）。落盘后会自动跑一轮撞车检索，嫌疑清单附在回执尾部，新颖性判断归你。")
    gap_sources: list[str] = Field(description="Gap 来源（结构依据），每条必须含至少一个可校验锚点：『母题N』/ 已有卡片标题或 cards/xx.json / 图谱 paper_id。没有锚点会被拒绝。")
    motivation: str = Field(description="动机：为什么现在能做、为什么值得做（中文）。")
    core_hypothesis: str = Field(description="核心假设：一句可证伪的主张（中文）。")
    testable_predictions: list[str] = Field(description="可检验预测，逐条具体到指标/阈值/对照。")
    approach: str = Field(description="技术路线：关键机制与创新点（中文）。")
    minimal_experiment: str = Field(description="最小实验设计：数据/基线/指标/关键消融（中文）。")
    related_paper_ids: list[str] = Field(default_factory=list, description="相关论文的图谱 paper_id（会校验在图并自动带出标题）。")
    status: str = Field(default="draft", description="状态：draft / reviewed / final。")


def _idea_filename(title: str) -> str:
    t = re.sub(r"[^\w\u4e00-\u9fff]+", "_", title).strip("_")
    return (t[:60] or "untitled") + ".md"


def build_writing_tools(ctx: RunContext) -> list[Tool]:

    def write_draft(p: WriteDraftParams) -> ToolResult[ToolUseCountMetadata]:
        ws = ctx.ws
        missing = [pid for pid in p.cite_paper_ids if pid not in ws.papers]
        if missing:
            return _ok(
                f"以下引用 id 不在图谱里，拒绝生成（引用必须真实）：{missing}\n"
                f"先把它们加进图谱（find_candidates/add_seed/graph_search），或从引用列表去掉。"
            )

        # attach real experiment numbers, if any
        result_block = ""
        if p.experiment_name:
            rf = ctx.project_dir(EXPERIMENTS_DIR) / p.experiment_name / RESULT_FILE
            if not rf.is_file():
                return _ok(f"实验 {p.experiment_name} 没有 {RESULT_FILE}，不能写实验章节。先跑通实验。")
            data = json.loads(rf.read_text(encoding="utf-8"))
            result_block = (
                "\n## Appendix: Raw Results\n\n```json\n"
                + json.dumps(data, ensure_ascii=False, indent=2)[:4000]
                + "\n```\n"
            )

        papers = [ws.papers[pid] for pid in p.cite_paper_ids]
        bib = "\n\n".join(_bibtex_entry(pp) for pp in papers)
        keymap = "\n".join(f"- [@{_bibkey(pp)}] = {pp.title[:70]} ({pp.year})" for pp in papers)

        slug = re.sub(r"[^\w-]+", "_", p.title)[:60].strip("_")
        d = ctx.project_dir(DRAFTS_DIR)
        md_path = d / f"{slug}.md"
        bib_path = d / f"{slug}.bib"
        md = (
            f"# {p.title}\n\n"
            f"> draft · {date.today().isoformat()} · idea: {p.idea_file}"
            + (f" · experiment: {p.experiment_name}" if p.experiment_name else "")
            + "\n\n"
            f"## Abstract\n\n{p.abstract}\n\n"
            f"## 1 Introduction\n\n{p.introduction}\n\n"
            f"## 2 Method\n\n{p.method}\n\n"
            f"## 3 Experiments\n\n{p.experiments_section}\n\n"
            f"## 4 Conclusion\n\n{p.conclusion}\n\n"
            f"## References\n\n{keymap}\n\n（完整 bibtex 见 {bib_path.name}）\n"
            + result_block
        )
        md_path.write_text(md, encoding="utf-8")
        bib_path.write_text(bib, encoding="utf-8")
        ctx.ws.trace.add("agent", "write_draft", p.title)
        return _ok(
            f"草稿已生成：\n- {md_path}\n- {bib_path}（{len(papers)} 条真实文献）\n\n"
            f"bibkey 对照：\n{keymap}\n\n请用户审阅；修改直接重调 write_draft 覆盖。"
        )

    def fill_domain_report(p: FillDomainReportParams) -> ToolResult[ToolUseCountMetadata]:
        from ..core.models import FieldReport, Stage

        ws = ctx.ws
        # id 校验：智能判断归调用方，但引用必须落在图谱里（防幻觉 id 进报告）
        # id 先过别名解析（emit 合并后 arXiv 版 id 可能已并进 DOI/W 版，实测种子 id 蒸发）
        all_ids = [ws.resolve_id(pid) or pid for pid in (
            p.founding_papers + p.main_line + p.must_read
            + [pid for s in p.stages for pid in s.papers]
        )]
        unknown = sorted({pid for pid in all_ids if pid not in ws.papers})
        if unknown:
            return _ok(
                "fill_domain_report 拒绝：以下 paper_id 不在图谱中——" + ", ".join(unknown[:8])
                + "。用 graph_summary/find_in_graph 核对后重试。"
            )
        _r = lambda pid: ws.resolve_id(pid) or pid  # noqa: E731 — 写进报告的一律用正本 id
        ctx.report = FieldReport(
            query=ws.query,
            founding_papers=[_r(x) for x in p.founding_papers],
            stages=[Stage(**{**s.model_dump(), "papers": [_r(x) for x in s.papers]}) for s in p.stages],
            main_line=[_r(x) for x in p.main_line],
            gaps=p.gaps,
            reading_path=[_r(x) for x in p.main_line],
            must_read=[_r(x) for x in p.must_read],
            must_read_reasons={_r(k): v for k, v in p.must_read_reasons.items()},
            narrative=p.narrative,
            cover_title=p.cover_title,
            cover_blurb=p.cover_blurb,
            tldr=p.tldr,
            core_idea=p.core_idea,
            prerequisites=p.prerequisites,
            glossary=p.glossary,
            getting_started=p.getting_started,
        )
        ctx.founding = [_r(x) for x in p.founding_papers]
        ctx.graph = None  # 让下次 emit 重物化，把 founding 角色标进节点
        ws.trace.add("agent", "report", f"domain report filled: {len(p.stages)} stages, {len(p.founding_papers)} founding")
        return _ok(
            f"领域脉络报告已装配：{len(p.stages)} 个阶段 / {len(p.founding_papers)} 篇奠基 / "
            f"必读 {len(p.must_read)} 篇。调 emit_result 落盘——report.md 会随 result.json 一起渲染。"
        )

    async def fill_research_idea(p: ResearchIdeaParams) -> ToolResult[ToolUseCountMetadata]:
        import json as _json
        from datetime import date as _date

        ws = ctx.ws
        # 锚点校验：母题N（对 _themes.json 长度）/ 卡片标题或文件名 / 图谱 paper_id
        themes_path = ctx.project_dir("cards") / "_themes.json"
        n_themes = 0
        if themes_path.is_file():
            try:
                n_themes = len(_json.loads(themes_path.read_text(encoding="utf-8")))
            except (OSError, _json.JSONDecodeError):
                n_themes = 0
        # 卡片标题全文也载入：缩写式引用（'TCSM'）只存在于标题括号里，
        # 光比文件名会误拒（实测多花一轮往返）；拒绝时也用它给出候选提示
        card_titles: dict[str, str] = {}
        for f in ctx.project_dir("cards").glob("*.json"):
            if f.name.startswith("_"):
                continue
            try:
                card_titles[f.name] = str(_json.loads(f.read_text(encoding="utf-8")).get("title", "")).lower()
            except (OSError, _json.JSONDecodeError):
                card_titles[f.name] = ""
        card_keys = " ".join(card_titles.keys()).lower()
        bad = []
        for src in p.gap_sources:
            ok = False
            for m in re.finditer(r"母题\s*(\d+)", src):
                ok = ok or (1 <= int(m.group(1)) <= n_themes)
            for m in re.finditer(r"([\w.]+\.json)", src):
                ok = ok or (m.group(1).lower() in card_keys)
            for m in re.finditer(r"(W\d{6,}|arxiv:\d{4}\.\d{4,})", src):
                ok = ok or (ws.resolve_id(m.group(1)) is not None)
            if not ok:
                # 卡片标题式引用：与任一卡片文件名词重叠即认
                words = [w for w in re.findall(r"[a-z0-9\u4e00-\u9fff]{3,}", src.lower()) if w in card_keys]
                ok = len(words) >= 2
            if not ok:
                # 全大写缩写（TCSM/VRPO）命中某张卡片标题也认——锚点同样可追溯
                for tok in re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", src):
                    if any(tok.lower() in t for t in card_titles.values()):
                        ok = True
                        break
            if not ok:
                bad.append(src[:60])
        if bad:
            # 拒绝时附词面相近的卡片候选，省掉 agent 自己猜的一轮往返
            hints = []
            for src in bad:
                toks = [t.lower() for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{2,}", src)]
                scored = sorted(
                    ((sum(1 for t in toks if t in title or t in fn.lower()), fn)
                     for fn, title in card_titles.items()),
                    reverse=True,
                )
                if scored and scored[0][0] > 0:
                    hints.append(f"『{src[:40]}』可能指 cards/{scored[0][1]}")
            hint_tail = ("\n锚点候选（词面猜测，确认后用文件名引用）：" + "；".join(hints)) if hints else ""
            return _ok(
                "fill_research_idea 拒绝：以下 Gap 来源缺少可校验锚点（母题N / cards/xx.json / 图谱 paper_id）——"
                + "; ".join(bad)
                + f"。当前母题 {n_themes} 条；先 save_themes/fill_idea_card 再引用。"
                + hint_tail
            )
        rids = [ws.resolve_id(pid) or pid for pid in p.related_paper_ids]
        unknown = [pid for pid in rids if pid not in ws.papers]
        if unknown:
            return _ok("fill_research_idea 拒绝：相关论文不在图谱——" + ", ".join(unknown[:5]))

        lines = [
            f"# {p.title}",
            "",
            f"> 状态: {p.status} · {_date.today().isoformat()} · fill_research_idea 装配（锚点已校验）",
            "",
            "## Gap 来源（结构依据）",
            *[f"- {s}" for s in p.gap_sources],
            "",
            "## 动机",
            p.motivation,
            "",
            "## 核心假设",
            p.core_hypothesis,
            "",
            "可检验预测：",
            *[f"({chr(97 + i)}) {t}" for i, t in enumerate(p.testable_predictions)],
            "",
            "## 技术路线",
            p.approach,
            "",
            "## 最小实验设计",
            p.minimal_experiment,
        ]
        if p.related_paper_ids:
            lines += ["", "## 相关论文"]
            for pid in rids:
                paper = ws.papers[pid]
                lines.append(f"- {pid} — {paper.title}" + (f"（{paper.year}）" if paper.year else ""))
        ideas_dir = ctx.project_dir("ideas")
        path = ideas_dir / _idea_filename(p.title)
        existed = path.exists()
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ctx.ws.trace.add("agent", "idea", f"{path.name} | sources={len(p.gap_sources)}")

        # 撞车检索（机械操作）：落盘后自动拿 novelty_probe 跑一轮多源检索，
        # 嫌疑清单附在回执里——prose 级的“出炉即查重”纪律实测被无视（三连发零查重），
        # 只有 in-band 结果躲不开；新颖性判断仍归外层
        probe_tail = ""
        try:
            batches = await ctx.multi.broad_search(p.novelty_probe)
            from itertools import zip_longest
            seen: dict[str, object] = {}
            for paper in (x for tup in zip_longest(*batches) for x in tup if x is not None):
                seen.setdefault(paper.title.strip().lower(), paper)
            graph_titles = {pp.title.strip().lower() for pp in ws.papers.values()}
            rows = []
            for paper in list(seen.values())[:6]:
                in_graph = ws.resolve_id(paper.id) is not None or paper.title.strip().lower() in graph_titles
                rows.append(
                    f"  {paper.id} | {paper.year or '?'} | cites={paper.citation_count} | "
                    f"{'已在图' if in_graph else '图外'} | {paper.title[:80]}"
                )
            if rows:
                probe_tail = (
                    f"\n\n[撞车嫌疑 novelty_probe='{p.novelty_probe}'（自动检索，新颖性判断归你）]\n"
                    + "\n".join(rows)
                    + "\n逐条判：与本 idea 核心机制高度重合的，read_paper 核实后修订 idea 明写差异化或降级放弃；"
                    "图外命中是主要风险。再用你环境里的联网搜索（WebSearch 等）补一路——结构化源对超新工作索引滞后。"
                )
            else:
                probe_tail = (
                    f"\n\n[撞车检索 novelty_probe='{p.novelty_probe}' 零命中——检索为空≠无撞车，"
                    "用你环境里的联网搜索（WebSearch 等）补查一轮再定稿]"
                )
        except Exception as e:  # noqa: BLE001 — 撞车检索失败不能连带否掉已落盘的 idea
            probe_tail = (
                f"\n\n[撞车检索失败（{type(e).__name__}，可能限流）——"
                "用你环境里的联网搜索（WebSearch 等）自查一轮再定稿]"
            )
        return _ok(
            f"idea 已{'就地覆盖' if existed else '新建'}: ideas/{path.name}（统一模板，锚点 {len(p.gap_sources)} 条已校验）。"
            + probe_tail
        )

    return [
        Tool[WriteDraftParams, ToolUseCountMetadata](
            name="write_draft",
            description=(
                "把定稿 idea + 真实实验结果组装成论文草稿（markdown + bibtex）。硬约束：引用只能是"
                "图谱里的论文（bibtex 由真实元数据生成，杜绝编造引用）；实验数字只能来自 "
                "result.json（你自己跑完实验后写入 experiments/<实验名>/result.json）。这是流程的最后一步。"
            ),
            parameters=WriteDraftParams,
            executor=write_draft,
        ),
        Tool[FillDomainReportParams, ToolUseCountMetadata](
            name="fill_domain_report",
            description=(
                "装配领域脉络报告（零 LLM）：你（调用方）提供全部智能内容——入门导读/术语/发展分期/"
                "奠基判断/主线/必读清单/gaps，本工具只做 paper_id 校验、FieldReport 组装与 founding 登记。"
                "所有引用必须是图谱内的 paper_id。装配后调 emit_result，report.md/view.html 随之渲染。"
            ),
            parameters=FillDomainReportParams,
            executor=fill_domain_report,
        ),
        Tool[ResearchIdeaParams, ToolUseCountMetadata](
            name="fill_research_idea",
            description=(
                "把你构思的研究 idea 落盘到 ideas/（零 LLM，统一模板，无需参考老项目格式）。"
                "硬闸门：gap_sources 每条必须含可校验锚点（母题N / 卡片引用 / 图谱 paper_id），"
                "缺锚点直接拒绝——先 fill_idea_card 落卡、save_themes 归纳母题，再写 idea。"
                "落盘后会自动拿 novelty_probe 跑一轮撞车检索，回执尾部附嫌疑清单，逐条判完新颖性再继续。"
                "同标题就地覆盖（文件名不变纪律内置）。"
            ),
            parameters=ResearchIdeaParams,
            executor=fill_research_idea,
        ),
    ]

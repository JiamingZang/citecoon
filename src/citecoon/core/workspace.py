"""Shared, mutable workspace the agents read from and write to, plus a trace."""
from __future__ import annotations

from .models import CitationGraph, GraphEdge, GraphNode, Paper


def _has_title(p: Paper) -> bool:
    t = (p.title or "").strip()
    return bool(p.id) and bool(t) and t.lower() != "(untitled)"


class Trace:
    """Append-only event log — the proof of agentic behavior (for demo/inspection)."""

    def __init__(self, on_event=None) -> None:
        self.events: list[dict] = []
        self._on_event = on_event  # optional live callback(event_dict)

    def add(self, agent: str, kind: str, content) -> None:
        ev = {"i": len(self.events), "agent": agent, "type": kind, "content": content}
        self.events.append(ev)
        if self._on_event is not None:
            try:
                self._on_event(ev)
            except Exception:
                pass

    def dump(self) -> list[dict]:
        return self.events


class Workspace:
    def __init__(self, query: str, settings, on_event=None) -> None:
        self.query = query
        self.settings = settings
        self.papers: dict[str, Paper] = {}
        self.depth: dict[str, int] = {}
        self.seeds: list[str] = []
        self.expanded_fwd: set[str] = set()  # nodes whose citers we've fetched
        self.expanded_bwd: set[str] = set()  # nodes whose references we've pulled
        self.link_attempted: set[str] = set()  # nodes link_frontier already tried (success or not)
        self._last_disconnect_sig: tuple | None = None  # summary() 去重：断连名单没变就不重播
        self._last_top_sig: tuple | None = None  # 同款去重：Top 基石列表没变就不重播
        self.trace = Trace(on_event=on_event)

    # -- mutation ---------------------------------------------------------
    def add_paper(self, p: Paper, depth: int) -> bool:
        if not _has_title(p):
            return False
        if p.id in self.papers:
            return False
        # Collection cap is intentionally larger than max_nodes (the final target):
        # otherwise early most-cited expansion fills every slot and the late-fetched
        # frontier can't get in. Relevance pruning trims to the on-topic final graph.
        cap = self.settings.max_collect or self.settings.max_nodes * 4
        if len(self.papers) >= cap:
            return False
        self.papers[p.id] = p
        self.depth[p.id] = depth
        return True

    def add_seed(self, p: Paper) -> None:
        # 论文已在图中（先 expand 后补 add_seed 的顺序，MCP 会话实测发生）时
        # add_paper 返回 False，但种子身份仍须登记——否则 seeds=[] 让相关性锚点
        # 失去种子标题兜底，off-topic prune 会误剪正主（实测剪掉 Self-Refine）
        self.add_paper(p, 0)
        if p.id in self.papers and p.id not in self.seeds:
            self.seeds.append(p.id)

    def resolve_id(self, pid: str) -> str | None:
        """id → 图内正本 id（含别名）。emit 时同题记录会被合并（arXiv 版并进 DOI/W 版），
        重启回灌后旧 id 蒸发——agent 拿着用了一整个会话的 arxiv id 突然查无此文（实测）。
        别名藏在正本的 source_ids/doi 里，这里统一解析。"""
        if pid in self.papers:
            return pid
        key = pid.strip().lower()
        bare = key.split(":", 1)[-1]
        for cid, p in self.papers.items():
            sids = p.source_ids or {}
            ax = str(sids.get("arxiv") or "").lower()
            if ax and bare in (ax, ax.split(":", 1)[-1]) and (key.startswith("arxiv") or key == ax):
                return cid
            if p.doi and key in (p.doi.lower(), f"doi:{p.doi.lower()}"):
                return cid
            s2 = str(sids.get("s2") or "").lower()
            if s2 and key == f"s2:{s2}":
                return cid
        return None

    # -- views for the agent ---------------------------------------------
    def note_edge(self, citing: str, cited: str) -> None:
        """扩边时当场登记 citing→cited 引用边。arXiv/S2 入图的论文不带参考文献列表
        （arXiv API 没有、S2 搜索面不取），扩边工具明明知道引用关系却只加节点，
        边全靠 refs 事后字面匹配——llada 实测 48 篇 0 边。关系写回正本的
        referenced_works，随 papers 序列化天然持久化（与 wire_predecessors 同一约定）。"""
        s = self.resolve_id(citing)
        d = self.resolve_id(cited)
        if not s or not d or s == d:
            return
        p = self.papers[s]
        if d not in p.referenced_works:
            p.referenced_works.append(d)

    def _edge_pairs(self) -> list[tuple[str, str]]:
        """图内引用边（去重），refs 里的 W/DOI/裸 arXiv id 先折回正本再匹配——
        OpenAlex 给的 refs 是 W 形态而节点正本常是 arxiv: 形态，字面匹配一条都对不上（边版本的别名断裂）。"""
        alias: dict[str, str] = {}
        for cid, p in self.papers.items():
            alias.setdefault(cid, cid)
            for k, v in (p.source_ids or {}).items():
                v = str(v)
                alias.setdefault(v, cid)
                if k == "arxiv":
                    bare = v.split(":", 1)[-1]
                    alias.setdefault(bare, cid)
                    alias.setdefault(f"arxiv:{bare}", cid)
            if p.doi:
                alias.setdefault(p.doi, cid)
                alias.setdefault(f"doi:{p.doi}", cid)
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for pid, p in self.papers.items():
            for r in p.referenced_works:
                t = alias.get(r)
                if t and t != pid and (pid, t) not in seen:
                    pairs.append((pid, t))
                    seen.add((pid, t))
        return pairs

    def frontier(self, k: int = 8) -> list[Paper]:
        """High-citation nodes not yet expanded forward — best expansion targets."""
        cand = [p for pid, p in self.papers.items() if pid not in self.expanded_fwd]
        return sorted(cand, key=lambda p: p.citation_count, reverse=True)[:k]

    def degree(self) -> dict[str, int]:
        """In+out citation-edge degree of each paper, counting only edges to other
        papers currently in the graph (mirrors the edges to_graph() would emit)."""
        deg: dict[str, int] = {pid: 0 for pid in self.papers}
        for s, t in self._edge_pairs():
            deg[s] += 1
            deg[t] += 1
        return deg

    def summary(self, k: int = 8, newest_first: bool = False) -> str:
        n = len(self.papers)
        top = sorted(self.papers.values(), key=lambda p: p.citation_count, reverse=True)[:k]
        # 超出目标规模时说明原因，否则 "77/60" 这种分子大于分母的显示像个 bug
        over = "（种子/前驱直加不占扩边预算，最终建图按相关性剪回目标数）" if n > self.settings.max_nodes else ""
        lines = [f"Graph now has {n}/{self.settings.max_nodes} papers.{over}"]
        deg = self.degree()
        isolated = [pid for pid, d in deg.items() if d == 0]
        if isolated:
            # referenced_works is only ever populated by OpenAlex (arXiv-sourced Paper
            # objects never carry it) — so whether THAT field is empty, not in-graph
            # degree, is what tells us a paper is genuinely missing reference data vs.
            # just not-yet-expanded (its known references simply aren't in this graph).
            need_fulltext = sorted(
                (
                    pid for pid in isolated
                    if not self.papers[pid].referenced_works and self.papers[pid].source_ids.get("arxiv")
                ),
                key=lambda pid: self.papers[pid].citation_count,
                reverse=True,
            )
            need_expand = sorted(
                (pid for pid in isolated if self.papers[pid].referenced_works and pid not in self.expanded_bwd),
                key=lambda pid: self.papers[pid].citation_count,
                reverse=True,
            )
            lines.append(f"⚠ {len(isolated)}/{n} papers have ZERO citation edges (disconnected from the graph)")
            # 同一份断连名单每次 add_seed/expand 都全文重播（实测同段话+id 列表连发 6 遍），
            # 名单没变化就只留一行提示，处理建议看上一次输出
            sig = (tuple(need_fulltext), tuple(need_expand))
            if sig == self._last_disconnect_sig:
                lines.append("  （断连名单与上次反馈相同，列表与处理建议略——见前一次输出）")
            else:
                self._last_disconnect_sig = sig
                if need_fulltext:
                    shown = ", ".join(need_fulltext[:20])
                    more = f" (+{len(need_fulltext) - 20} more)" if len(need_fulltext) > 20 else ""
                    lines.append(
                        f"  — {len(need_fulltext)} have NO reference data at all (OpenAlex never indexed "
                        f"their refs) but DO have arXiv full text: read_paper 读全文、从相关工作/方法章节"
                        f"抽出前驱论文标题，然后 wire_predecessors 接线: {shown}{more}"
                    )
                if need_expand:
                    shown = ", ".join(need_expand[:20])
                    more = f" (+{len(need_expand) - 20} more)" if len(need_expand) > 20 else ""
                    lines.append(
                        f"  — {len(need_expand)} already HAVE reference data (from OpenAlex) but haven't "
                        f"been expand_backward'd yet — their cited papers just aren't in this graph yet, "
                        f"use expand_backward（不必读全文重推，OpenAlex 已经给了 refs）: {shown}{more}"
                    )
        else:
            self._last_disconnect_sig = None
        top_sig = tuple(p.id for p in top)
        if top_sig == self._last_top_sig:
            lines.append("Top by citations:（与上次相同，略）")
        else:
            self._last_top_sig = top_sig
            lines.append("Top by citations:")
            for p in top:
                mark = "" if p.id in self.expanded_fwd else "  [not expanded]"
                lines.append(f"  {p.id} | {p.year} | cites={p.citation_count} | {p.title[:60]}{mark}")
        if newest_first:
            # 引用数 Top 全是 ResNet/ViT 类基石，子题盘点看不到自己关心的新论文——
            # 补一个年份序视图（实测：精化子题的新节点全在低引区，top 列表零信息量）
            recent = sorted(self.papers.values(), key=lambda p: (p.year or 0, p.citation_count), reverse=True)[:k]
            lines.append("Newest in graph:")
            for p in recent:
                lines.append(f"  {p.id} | {p.year} | cites={p.citation_count} | {p.title[:60]}")
        return "\n".join(lines)

    # -- output -----------------------------------------------------------
    def to_graph(self) -> CitationGraph:
        edges = [GraphEdge(source=s, target=t, type="cites") for s, t in self._edge_pairs()]
        nodes = [
            GraphNode(
                paper_id=p.id,
                title=p.title,
                year=p.year,
                citation_count=p.citation_count,
                depth=self.depth.get(p.id, 0),
            )
            for p in self.papers.values()
        ]
        return CitationGraph(
            query=self.query, seeds=self.seeds, nodes=nodes, edges=edges, papers=dict(self.papers)
        )

"""行为级 eval：项目状态渲染 render_project_status。零 LLM。

固化的人肉验收标准：
- verdict 长词优先——"需修订后发布"含"可发布"子串，匹配顺序反了会误判（已修 bug 的回归锁）
- idea 评审状态、报告待验证计数、实验推进态如实反映
"""

from citecoon.tools.health import render_project_status


def test_verdict_long_word_priority(make_project):
    # 回归：曾因短词"可发布"在前，把"需修订后发布"误判成"可发布"
    p = (make_project()
         .add_report("r_need", verdict="需修订后发布")
         .add_report("r_ok", verdict="可发布")
         .add_report("r_bad", verdict="不可信需重写"))
    out = render_project_status(p.root)
    assert "审校判定[需修订后发布]" in out
    assert "审校判定[可发布]" in out
    assert "审校判定[不可信需重写]" in out
    # "需修订后发布"那行绝不能被渲染成 [可发布]
    need_line = next(ln for ln in out.splitlines() if "r_need" in ln)
    assert "[需修订后发布]" in need_line and "[可发布]" not in need_line


def test_idea_critique_state(make_project):
    p = (make_project()
         .add_idea("done_one", critiqued=True)
         .add_idea("raw_one", critiqued=False))
    out = render_project_status(p.root)
    assert "done_one" in out and "raw_one" in out
    done_line = next(ln for ln in out.splitlines() if "done_one" in ln)
    raw_line = next(ln for ln in out.splitlines() if "raw_one" in ln)
    assert "已评审" in done_line
    assert "未见评审记录" in raw_line


def test_report_pending_and_pdf(make_project):
    p = make_project().add_report("rep", verdict="可发布", pending=5, pdf=True)
    out = render_project_status(p.root)
    line = next(ln for ln in out.splitlines() if "rep" in ln and "待验证" in ln)
    assert "待验证×5" in line
    assert "PDF✓" in line


def test_experiment_states(make_project):
    p = (make_project()
         .add_experiment("planned", plan=True, approved=False)
         .add_experiment("done_exp", plan=True, approved=True, verdict="supported"))
    out = render_project_status(p.root)
    assert "planned — 有方案" in out
    done_line = next(ln for ln in out.splitlines() if "done_exp" in ln)
    assert "已实测" in done_line and "supported" in done_line


def test_non_project_dir_graceful(make_project, tmp_path):
    # 没有 result.json 的目录不该崩，给出清晰提示
    out = render_project_status(tmp_path / "not_a_project")
    assert "不是项目目录" in out

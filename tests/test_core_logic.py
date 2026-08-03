"""核心纯逻辑回归测试。

跑法：.venv/bin/python -m pytest tests/ -q
"""
from citecoon.core.layout import LAYOUT, dir_for, find_result, is_project_dir
from citecoon.core.utils import extract_json, looks_like_id
from citecoon.sources.resolve import overlap


# ---- extract_json ----


def test_extract_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_fenced():
    assert extract_json('前言\n```json\n{"a": 1}\n```\n后记') == {"a": 1}


def test_extract_with_prose():
    assert extract_json('好的，结果如下：{"a": [1, 2]} 希望有帮助') == {"a": [1, 2]}


def test_extract_malformed_repaired():
    # 尾逗号：标准 json 拒绝，json_repair 兜底
    assert extract_json('{"a": 1,}') == {"a": 1}


def test_extract_none_and_garbage():
    assert extract_json(None) is None
    assert extract_json("") is None
    assert extract_json("完全没有 JSON 的一段话") is None


# ---- looks_like_id ----


def test_id_patterns():
    assert looks_like_id("W2626778328")
    assert looks_like_id("10.1109/ACCESS.2025")
    assert looks_like_id("arxiv:2403.11510")
    assert not looks_like_id("test-time optimization 6D pose")


# ---- overlap（标题词重叠，防误接线的阈值依据）----


def test_overlap():
    assert overlap("GenFlow: Generalizable Recurrent Flow", "GenFlow Recurrent Flow for Refinement") >= 2
    assert overlap("Attention Is All You Need", "GenFlow") == 0


# ---- 产物分层布局 ----


def test_layout_paths(tmp_path):
    # 新布局写入后 find_result / dir_for 命中新路径
    gdir = tmp_path / "01_graph"
    gdir.mkdir()
    (gdir / "result.json").write_text("{}")
    assert find_result(tmp_path) == gdir / "result.json"
    assert is_project_dir(tmp_path)
    cards = tmp_path / "02_reading" / "cards"
    cards.mkdir(parents=True)
    assert dir_for(tmp_path, "cards") == cards


def test_layout_legacy_fallback(tmp_path):
    # 旧平铺布局：result.json 在根，cards/ 平铺——迁移期仍可读
    (tmp_path / "result.json").write_text("{}")
    (tmp_path / "cards").mkdir()
    assert find_result(tmp_path) == tmp_path / "result.json"
    assert dir_for(tmp_path, "cards") == tmp_path / "cards"


def test_layout_map_covers_all_dirs():
    assert set(LAYOUT) == {"cards", "papers", "ideas", "reports", "drafts", "codebases", "experiments"}


# ---- 懒加载符号哨兵 ----
# read.py 在函数体里 from .batch_read import ...（运行时才解析），
# 重构误删函数时模块级 import 冒烟抓不到——实测 read_paper 现场炸过


def test_lazy_imports_alive():
    from citecoon.tools.batch_read import (  # noqa: F401
        _fetch_pdf_and_parse,
        _fulltext,
        _persist_fulltext,
    )
    from citecoon.sources.resolve import resolve_titles  # noqa: F401


def test_no_library_residue():
    # library 层已拆除：残留的 lib_* 调用只会在运行时炸（实测 read_paper NameError），
    # 静态扫一遍源码兜住
    import pathlib

    for f in pathlib.Path(__file__).parents[1].joinpath("src", "citecoon").rglob("*.py"):
        if "__pycache__" in str(f):
            continue
        text = f.read_text(encoding="utf-8")
        for token in ("lib_store", "lib_fetch", "lib_search"):
            assert token not in text, f"{f} 残留 {token}"

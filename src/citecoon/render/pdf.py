"""Markdown 技术报告 → PDF 导出。

渲染链：python-markdown（tables/fenced_code）→ 内嵌 CSS 的独立 HTML →
Chrome headless 打印 PDF。公式用 KaTeX auto-render（CDN；离线时降级为原样
LaTeX 文本，不阻塞导出）。不引入 pandoc/weasyprint 这类重依赖——机器上有
Chrome 就能跑。

数学片段在 markdown 转换前先抽出来换成占位符，转换后再放回，避免下划线、
星号被 markdown 当强调语法吃掉。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import date
from pathlib import Path

_CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]

# 版式基调来自参考技术报告：A4、中文衬线正文、无衬线标题、细边框表格。
_CSS = """
@page { size: A4; margin: 22mm 18mm; }
html { -webkit-print-color-adjust: exact; }
body {
  font-family: "Songti SC", "STSong", "Noto Serif CJK SC", serif;
  font-size: 10.5pt; line-height: 1.75; color: #1a1a1a;
  max-width: 100%; margin: 0;
}
h1, h2, h3, h4 {
  font-family: "PingFang SC", "Helvetica Neue", sans-serif;
  color: #111; line-height: 1.35; page-break-after: avoid;
}
h1.doc-title { font-size: 19pt; margin: 0 0 2pt; }
p.doc-subtitle { font-family: "PingFang SC", sans-serif; font-size: 12pt; color: #444; margin: 0 0 4pt; }
p.doc-meta {
  font-family: "PingFang SC", sans-serif; font-size: 8.5pt; color: #888;
  border-bottom: 1.2pt solid #222; padding-bottom: 10pt; margin: 0 0 18pt;
}
h1 { font-size: 14.5pt; margin: 22pt 0 8pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 6pt; }
h3 { font-size: 11pt;  margin: 12pt 0 4pt; }
p { margin: 4pt 0 8pt; text-align: justify; }
strong { color: #000; }
table {
  border-collapse: collapse; width: 100%; margin: 8pt 0 12pt;
  font-family: "PingFang SC", sans-serif; font-size: 9pt; page-break-inside: avoid;
}
th, td { border: 0.6pt solid #999; padding: 4pt 7pt; text-align: left; }
th { background: #f2f2f2; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
pre {
  background: #f6f6f6; border: 0.6pt solid #ddd; border-radius: 3pt;
  padding: 8pt 10pt; font-size: 8.5pt; line-height: 1.5;
  white-space: pre-wrap; word-break: break-all; page-break-inside: avoid;
}
code { font-family: "SF Mono", Menlo, monospace; font-size: 0.92em; background: #f4f4f4; padding: 0 3pt; border-radius: 2pt; }
pre code { background: none; padding: 0; }
blockquote {
  margin: 6pt 0; padding: 2pt 10pt; border-left: 2.5pt solid #bbb;
  color: #666; font-size: 9pt; font-family: "PingFang SC", sans-serif;
}
ul, ol { margin: 4pt 0 8pt; padding-left: 22pt; }
li { margin: 2pt 0; }
hr { border: none; border-top: 0.6pt solid #ccc; margin: 14pt 0; }
.katex { font-size: 1.02em; }
.katex-display { margin: 8pt 0; }
"""

_KATEX = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{delimiters:[
    {left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],throwOnError:false});"></script>
"""


def find_chrome() -> str | None:
    if os.environ.get("SAAS_CHROME"):
        return os.environ["SAAS_CHROME"]
    for c in _CHROME_CANDIDATES:
        if Path(c).is_file():
            return c
    return shutil.which("chromium") or shutil.which("google-chrome")


def _shield_math(md: str) -> tuple[str, list[str]]:
    """把 $$...$$ / $...$ 抽成占位符，防止 markdown 转换吃掉下划线星号。"""
    spans: list[str] = []

    def _keep(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"\x00MATH{len(spans) - 1}\x00"

    md = re.sub(r"\$\$.+?\$\$", _keep, md, flags=re.S)
    md = re.sub(r"(?<![\\$])\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\d)", _keep, md)
    return md, spans


def markdown_to_html(md_text: str, title: str, subtitle: str = "", meta: str = "") -> str:
    import markdown as mdlib

    body_md, spans = _shield_math(md_text)
    body = mdlib.markdown(
        body_md, extensions=["tables", "fenced_code", "sane_lists", "smarty"]
    )
    for i, s in enumerate(spans):
        body = body.replace(f"\x00MATH{i}\x00", s)

    head = f'<h1 class="doc-title">{title}</h1>\n'
    if subtitle:
        head += f'<p class="doc-subtitle">{subtitle}</p>\n'
    head += f'<p class="doc-meta">{meta or date.today().isoformat()}</p>\n'
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{_CSS}</style>{_KATEX}</head><body>{head}{body}</body></html>"
    )


def html_to_pdf(html: str, pdf_path: Path, timeout: int = 90) -> None:
    chrome = find_chrome()
    if not chrome:
        raise RuntimeError("找不到 Chrome/Chromium，请安装或设置 SAAS_CHROME 指向可执行文件")
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html)
        html_path = f.name
    # 独立临时 profile：避免多实例串行锁争用（并行导出互相卡 90s）；
    # --use-mock-keychain 避免 macOS 弹“访问钥匙串”授权窗（HOME 被重定向时必现）
    profile_dir = tempfile.mkdtemp(prefix="saas-chrome-")
    started = time.time()
    try:
        # 实测 macOS 上 Chrome 写完 PDF 后常常不得好死：新 profile 首跑会拉起
        # GoogleUpdater 注册（挂十几秒、退出码非零）。退出码/超时都不作数，
        # 唯一验收标准是本次运行新写出的 PDF（末尾 mtime + 尺寸检查）
        try:
            subprocess.run(
                [
                    chrome, "--headless=new", "--disable-gpu",
                    f"--user-data-dir={profile_dir}",
                    "--use-mock-keychain", "--password-store=basic",
                    "--no-first-run", "--disable-sync",
                    "--disable-background-networking", "--disable-component-update",
                    # 给 KaTeX CDN 加载 + 渲染留虚拟时间；离线时到点直接打印原样文本
                    "--virtual-time-budget=15000",
                    "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}",
                    f"file://{html_path}",
                ],
                check=False, capture_output=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            pass  # 子进程已被 kill；PDF 多半早已落盘
    finally:
        os.unlink(html_path)
        shutil.rmtree(profile_dir, ignore_errors=True)
    if (not pdf_path.is_file() or pdf_path.stat().st_size < 1024
            or pdf_path.stat().st_mtime < started - 1):
        raise RuntimeError("Chrome 本次运行未写出新 PDF（渲染失败或超时时尚未落盘）")


def report_md_to_pdf(md_path: Path, pdf_path: Path | None = None) -> Path:
    """报告 md → PDF。首行 '# 标题' 作封面标题，紧随的 '> ...' 溯源行作 meta。"""
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    title, meta, start = md_path.stem, "", 0
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        start = 1
    # 吃掉紧跟标题的溯源引用块（> 技术可行性报告 · ...），转为 meta 行
    while start < len(lines) and (not lines[start].strip() or lines[start].startswith(">")):
        if lines[start].startswith(">"):
            meta = (meta + " " + lines[start].lstrip("> ").strip()).strip()
        start += 1
    # 主标题里的中英文副标题拆行显示
    subtitle = ""
    m = re.match(r"^(.*?)[（(]([A-Za-z][^）)]*)[)）]\s*$", title)
    if m:
        title, subtitle = m.group(1).strip(), m.group(2).strip()
    html = markdown_to_html("\n".join(lines[start:]), title, subtitle, meta)
    # 新目录结构下 md 固定叫 report.md，PDF 若也叫 report.pdf，发给别人就分不清是哪份——
    # 改用报告目录名命名（reports/<slug>/<slug>.pdf）；旧平铺布局仍同名 .pdf
    if pdf_path is None:
        pdf_path = (md_path.parent / f"{md_path.parent.name}.pdf"
                    if md_path.stem == "report" else md_path.with_suffix(".pdf"))
    out = pdf_path
    html_to_pdf(html, out)
    return out

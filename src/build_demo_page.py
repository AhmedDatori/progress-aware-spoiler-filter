"""Build the web demo: put results/demo_comments.json into src/demo_page.html.

Writes docs/index.html (served by GitHub Pages) and report/demo.html (local copy). No GPU needed;
make_demo_data.py makes the data file.
Usage: python build_demo_page.py
"""
from common import RESULTS, ROOT

page = (ROOT / "src/demo_page.html").read_text()
data = (RESULTS / "demo_comments.json").read_text().replace("</", "<\\/")   # keep "</" out of the <script>
page = page.replace("__DATA__", data)
for out in [ROOT / "docs/index.html", ROOT / "report/demo.html"]:
    out.parent.mkdir(exist_ok=True)
    out.write_text(page)
    print("wrote", out.relative_to(ROOT))

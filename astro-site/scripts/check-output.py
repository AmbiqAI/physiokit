"""Validate the built documentation routes, local links, and migrated assets."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class PageLinks(HTMLParser):
    """Collect links and anchors from one rendered page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name == "id" and value:
                self.anchors.add(value)
            if name in {"href", "src"} and value:
                self.links.append(value)


site = Path("dist")
repo = Path("..")
pages: dict[Path, PageLinks] = {}
for file in site.rglob("*.html"):
    parser = PageLinks()
    parser.feed(file.read_text(errors="replace"))
    pages[file.resolve()] = parser

required = [
    "index.html",
    "tutorial/index.html",
    "tutorial/quickstart/index.html",
    "tutorial/your-data/index.html",
    "reference/index.html",
    *[f"reference/{name}/index.html" for name in ("ecg", "ppg", "rsp", "imu", "hrv", "signal")],
    "api/index.html",
    "api/physiokit/index.html",
]
errors: list[str] = []
for route in required:
    if not (site / route).is_file():
        errors.append(f"Missing route: {route}")

source_plots = {file.name for file in Path("public/assets").glob("*.html")}
output_plots = {file.name for file in (site / "assets").glob("*.html")}
if len(source_plots) != 22:
    errors.append(f"Expected 22 migrated plots, found {len(source_plots)}")
if output_plots != source_plots:
    errors.append(f"Plot assets differ: missing {source_plots - output_plots}; extra {output_plots - source_plots}")
if not (repo / "notebooks/docs.ipynb").is_file():
    errors.append("Plot source notebook is missing")

for page, parsed in pages.items():
    for link in parsed.links:
        url = urlsplit(link)
        if url.scheme or url.netloc or link.startswith(("mailto:", "data:", "javascript:")):
            continue
        path = unquote(url.path)
        if not path:
            target = page
        elif path.startswith("/physiokit/"):
            target = site / path.removeprefix("/physiokit/")
        elif path.startswith("/"):
            errors.append(f"{page.relative_to(site.resolve())}: link outside base: {link}")
            continue
        else:
            target = page.parent / path
        if target.is_dir() or not target.suffix:
            target = target / "index.html"
        target = target.resolve()
        if not target.is_file():
            errors.append(f"{page.relative_to(site.resolve())}: missing target: {link}")
        elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].anchors:
            errors.append(f"{page.relative_to(site.resolve())}: missing anchor: {link}")

if errors:
    raise SystemExit("\n".join(errors[:100]))
print(f"Checked {len(pages)} pages, {len(source_plots)} plots, notebook source, routes, and local links.")

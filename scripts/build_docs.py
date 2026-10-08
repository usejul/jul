"""Build the documentation site: docs/*.md -> site/docs/*.html, in the Macintosh style of site/index.html.

    pip install -e ".[docs]"
    python scripts/build_docs.py            # writes site/docs/
    python scripts/build_docs.py --check    # also fails on a broken internal link or an undocumented JUL_* variable
    python -m http.server -d site 8000      # then open http://localhost:8000/docs/

The Markdown files in docs/ stay the source of truth, readable on GitHub as they are. This script adds
what a site needs and GitHub does not: a navigation tree, a page per command generated from the CLI's
own parser (so it cannot drift from the code), a search index, and links rewritten between pages.
Nothing here is committed: .github/workflows/pages.yml builds it on every push to main.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = ROOT / "site" / "docs"
REPO = "https://github.com/usejul/jul"

#: The navigation, in reading order: (section, [(page, title)]). A page is docs/<page>.md, except the
#: generated ones (cli-reference) and the changelog, read from the repo root.
NAV = [
    ("START HERE", [
        ("index", "Docs home"),
        ("quickstart", "Quickstart"),
        ("concepts", "Concepts"),
        ("hub", "The hub"),
        ("questions", "Writing good questions"),
    ]),
    ("GUIDES", [
        ("installation", "Installation"),
        ("cli", "Command line"),
        ("tuning", "Adapting to your data"),
        ("serve", "Serving over HTTP"),
        ("deployment", "Deploying a fixed need"),
        ("aws-lambda", "AWS Lambda, step by step"),
        ("telemetry", "Telemetry (OpenTelemetry)"),
    ]),
    ("MODELS", [
        ("models", "Models and readings"),
        ("benchmarks", "Benchmarks"),
    ]),
    ("REFERENCE", [
        ("python-api", "Python API"),
        ("cli-reference", "CLI reference"),
        ("configuration", "Configuration"),
        ("troubleshooting", "FAQ and troubleshooting"),
        ("glossary", "Glossary"),
    ]),
    ("PROJECT", [
        ("development", "Development"),
        ("publishing", "Publishing"),
        ("changelog", "Changelog"),
    ]),
]

#: Repo files a page may link to with a relative path: rewritten to GitHub.
SPECIAL = {"changelog": ROOT / "CHANGELOG.md"}


# --- markdown ------------------------------------------------------------------------------------

def slugify(value: str, separator: str = "-") -> str:
    """GitHub's anchors, so a link that works on GitHub works here: `autotune(...) — when` -> `autotune--when`."""
    value = value.strip().lower()
    value = re.sub(r"[^\w\- ]", "", value)
    return value.replace(" ", separator)


def render(text: str) -> tuple[str, list[tuple[int, str, str]]]:
    """HTML and the (level, id, title) of every h2/h3."""
    import markdown
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists", "md_in_html"],
                           extension_configs={"toc": {"slugify": slugify, "permalink": False}})
    body = md.convert(text)
    toc = []

    def walk(tokens):
        for t in tokens:
            if t["level"] in (2, 3):
                toc.append((t["level"], t["id"], html.unescape(t["name"])))
            walk(t["children"])
    walk(md.toc_tokens)
    return body, toc


def source(page: str) -> Path | None:
    if page in SPECIAL:
        return SPECIAL[page]
    path = DOCS / f"{page}.md"
    return path if path.exists() else None


def title_of(page: str) -> str:
    for _, pages in NAV:
        for p, t in pages:
            if p == page:
                return t
    return page


def rewrite_links(body: str, page: str) -> str:
    """docs/x.md -> x.html; ../README.md and repo files -> GitHub; GitHub links to docs -> local pages."""
    names = {p for _, pages in NAV for p, _ in pages}

    def fix(m):
        href = html.unescape(m.group(1))
        url, _, frag = href.partition("#")
        frag = f"#{frag}" if frag else ""
        gh = re.match(rf"{re.escape(REPO)}/blob/main/docs/([\w\-]+)\.md$", url)
        if gh and gh.group(1) in names:
            return f'href="{gh.group(1)}.html{frag}"'
        if url == REPO and frag:          # the README's sections
            return m.group(0)
        if re.match(r"^[a-z]+:", url) or not url:
            return m.group(0)
        if url.endswith(".md"):
            name = Path(url).stem
            if url.startswith("../"):
                if name == "CHANGELOG":
                    return f'href="changelog.html{frag}"'
                return f'href="{REPO}{frag}"' if name == "README" else f'href="{REPO}/blob/main/{url[3:]}{frag}"'
            if name in names:
                return f'href="{name}.html{frag}"'
            if page == "changelog" and url.startswith("docs/"):
                return f'href="{Path(url).stem}.html{frag}"'
        if url.endswith(".html"):
            return m.group(0)
        clean = url[3:] if url.startswith("../") else url
        return f'href="{REPO}/blob/main/{clean}{frag}"'

    return re.sub(r'href="([^"]*)"', fix, body)


# --- the generated CLI reference -----------------------------------------------------------------

def cli_reference() -> str:
    """One section per command, from the parser `jul` itself runs: it cannot miss a flag."""
    sys.path.insert(0, str(ROOT / "lib"))
    sys.path.insert(0, str(ROOT / "cli"))
    from jul_cli.main import build_parser
    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    helps = {a.dest: a.help for a in sub._choices_actions}
    out = ["# CLI reference", "",
           "Every command and every flag, generated from the parser `jul` runs (`jul <command> --help` prints",
           "the same). For what the commands are for, with examples and file formats, read",
           "[Command line](cli.md) first.", "",
           "| Command | What it does |", "| --- | --- |"]
    out += [f"| [`jul {name}`](#jul-{name}) | {helps.get(name, '')} |" for name in sub.choices]
    for name, p in sub.choices.items():
        usage = " ".join(p.format_usage().replace("usage: ", "").split())
        out += ["", f"## jul {name}", "", helps.get(name, "") + ".", "", "```text", usage, "```", ""]
        rows = []
        for a in p._actions:
            if isinstance(a, argparse._HelpAction):
                continue
            flag = ", ".join(a.option_strings) if a.option_strings else a.dest
            if a.metavar and a.option_strings:
                flag += f" {a.metavar}"
            elif a.option_strings and a.nargs != 0 and not isinstance(a, argparse._StoreTrueAction):
                flag += f" {a.dest.upper()}"
            values = ", ".join(map(str, a.choices)) if a.choices else ""
            default = "" if a.default in (None, False, argparse.SUPPRESS) or a.nargs == argparse.REMAINDER \
                else str(a.default)
            text = (a.help or "").replace("|", "\\|").replace("\n", " ")
            req = " **required**" if getattr(a, "required", False) and a.option_strings else ""
            rows.append(f"| `{flag}` | {values} | {default} | {text}{req} |")
        if rows:
            out += ["| Argument | Values | Default | What it does |", "| --- | --- | --- | --- |", *rows]
    return "\n".join(out) + "\n"


# --- page template ---------------------------------------------------------------------------------

def nav_html(current: str) -> str:
    lines = []
    for section, pages in NAV:
        lines.append(f'<p class="nav-h">{section}</p><ul class="tree">')
        for i, (p, t) in enumerate(pages):
            branch = "└─" if i == len(pages) - 1 else "├─"
            cur = ' aria-current="page"' if p == current else ""
            lines.append(f'<li><span class="br" aria-hidden="true">{branch}</span><a href="{p}.html"{cur}>{t}</a></li>')
        lines.append("</ul>")
    return "\n".join(lines)


def toc_html(toc) -> str:
    if len(toc) < 2:
        return ""
    items = "\n".join(f'<li class="l{lvl}"><a href="#{i}">{html.escape(t)}</a></li>' for lvl, i, t in toc)
    return f'<section class="win toc" data-title="On this page"><ul>{items}</ul></section>'


def page_html(page: str, body: str, toc, prev, nxt) -> str:
    title = title_of(page)
    edit = (f'{REPO}/edit/main/{source(page).relative_to(ROOT)}' if source(page)
            else f"{REPO}/blob/main/cli/jul_cli/main.py")
    pager = '<nav class="pager" aria-label="Pages">'
    pager += f'<a href="{prev[0]}.html" rel="prev">◄ {prev[1]}</a>' if prev else "<span></span>"
    pager += f'<a href="{nxt[0]}.html" rel="next">{nxt[1]} ►</a>' if nxt else "<span></span>"
    pager += "</nav>"
    desc = re.sub(r"<[^>]+>", "", body)
    desc = html.escape(" ".join(desc.split())[:160], quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · JuL docs</title>
<meta name="description" content="{desc}">
<meta name="theme-color" content="#ffffff">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' shape-rendering='crispEdges'%3E%3Crect width='16' height='16' fill='%23fff'/%3E%3Crect x='1' y='1' width='14' height='14' fill='none' stroke='%23000'/%3E%3Crect x='1' y='1' width='14' height='4' fill='%23000'/%3E%3Ctext x='3' y='13' font-family='monospace' font-size='8' fill='%23000'%3EJuL%3C/text%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,700;1,400&family=Jersey+15&family=Silkscreen&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/docs.css">
</head>
<body>
<a class="skip" href="#doc">Skip to the page</a>
<nav class="menubar" aria-label="Main">
  <a class="brand" href="../index.html">JuL</a>
  <a href="index.html">Docs</a>
  <a href="quickstart.html" class="hide-sm">Quickstart</a>
  <a href="python-api.html" class="hide-sm">API</a>
  <a href="cli-reference.html" class="hide-sm">CLI</a>
  <a href="{REPO}" class="hide-sm">GitHub</a>
  <span class="spacer"></span>
  <label class="search"><span aria-hidden="true">Find /</span><input id="q" type="search" placeholder="search" autocomplete="off" aria-label="Search the docs"></label>
</nav>
<div id="results" class="win results" data-title="Find" hidden aria-live="polite"></div>
<div class="layout">
  <aside class="side">
    <button class="nav-toggle" type="button" aria-expanded="false" aria-controls="dir">☰ Contents</button>
    <section class="win dir" id="dir" data-title="Docs">
{nav_html(page)}
    </section>
    {toc_html(toc)}
  </aside>
  <main id="doc">
    <section class="win doc" data-title="{html.escape(title)}">
<article class="prose">
{body}
</article>
{pager}
<p class="foot dim"><a href="{edit}">Edit this page</a> · JuL, <i>Just use Less</i> · Apache-2.0 · Not affiliated with TypeSafe AI or Jev.</p>
    </section>
  </main>
</div>
<script src="assets/docs.js"></script>
</body>
</html>
"""


# --- build -----------------------------------------------------------------------------------------

def sections_for_search(page: str, body: str) -> list[dict]:
    """One entry per h1/h2/h3 section: its title, anchor and plain text, for assets/search.json."""
    parts = re.split(r'(<h[123][^>]*>.*?</h[123]>)', body, flags=re.S)
    out, head, anchor = [], title_of(page), ""
    for chunk in parts:
        m = re.match(r'<h[123](?: id="([^"]*)")?[^>]*>(.*?)</h[123]>', chunk, flags=re.S)
        if m:
            anchor, head = m.group(1) or "", html.unescape(re.sub(r"<[^>]+>", "", m.group(2)))
            continue
        text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", chunk)).split())
        if text or anchor:
            out.append({"p": page, "t": title_of(page), "h": head, "a": anchor, "x": text[:1500]})
    return out


def check_env(pages: dict[str, str]) -> list[str]:
    """Every JUL_* variable the code reads must be documented on the configuration page."""
    used = set()
    for path in list((ROOT / "lib").rglob("*.py")) + list((ROOT / "cli").rglob("*.py")):
        used |= set(re.findall(r"\bJUL_[A-Z_]+[A-Z]\b", path.read_text()))
    documented = pages.get("configuration", "")
    return [f"configuration.md does not document {v}" for v in sorted(used) if v not in documented]


def check_links(rendered: dict[str, str]) -> list[str]:
    ids = {p: set(re.findall(r'id="([^"]+)"', b)) for p, b in rendered.items()}
    errors = []
    for page, body in rendered.items():
        for href in re.findall(r'href="([^"]+)"', body):
            if re.match(r"^[a-z]+:", href) or href.startswith("../"):
                continue
            url, _, frag = href.partition("#")
            target = Path(url).stem if url else page
            if url and not url.endswith(".html"):
                errors.append(f"{page}: link to {href}")
            elif target not in ids:
                errors.append(f"{page}: link to a missing page {href}")
            elif frag and frag not in ids[target]:
                errors.append(f"{page}: link to a missing anchor {href}")
    return errors


def build(check: bool = False) -> int:
    order = [(p, t) for _, pages in NAV for p, t in pages]
    raw, rendered, tocs = {}, {}, {}
    for page, _ in order:
        if page == "cli-reference":
            text = cli_reference()
        else:
            path = source(page)
            if path is None:
                print(f"missing docs/{page}.md", file=sys.stderr)
                return 1
            text = path.read_text()
        raw[page] = text
        body, toc = render(text)
        rendered[page] = rewrite_links(body, page)
        tocs[page] = toc

    errors = []
    if check:
        errors = check_links(rendered) + check_env(raw)
        missing = sorted(p.stem for p in DOCS.glob("*.md") if p.stem not in dict(order))
        errors += [f"docs/{m}.md is not in the navigation (scripts/build_docs.py, NAV)" for m in missing]
    for e in errors:
        print("error:", e, file=sys.stderr)
    if errors:
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "assets").mkdir(parents=True)
    for f in (ROOT / "site" / "assets").glob("docs.*"):
        shutil.copy(f, OUT / "assets" / f.name)
    if (DOCS / "assets").exists():
        for f in (DOCS / "assets").iterdir():
            shutil.copy(f, OUT / "assets" / f.name)
    search = []
    for i, (page, _) in enumerate(order):
        prev = order[i - 1] if i else None
        nxt = order[i + 1] if i + 1 < len(order) else None
        body = rendered[page].replace('src="assets/', 'src="assets/')
        (OUT / f"{page}.html").write_text(page_html(page, body, tocs[page], prev, nxt))
        search += sections_for_search(page, rendered[page])
    (OUT / "assets" / "search.json").write_text(json.dumps(search, ensure_ascii=False, separators=(",", ":")))
    print(f"{len(order)} pages, {len(search)} search entries -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="fail on broken internal links or undocumented JUL_* variables")
    sys.exit(build(ap.parse_args().check))

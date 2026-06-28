#!/usr/bin/env python3
"""Inject the sam.onl deprecation banner partial into rendered docs/ HTML.

Why this exists: a full `quarto render` is impractically slow in this working
copy (Quarto walks ~210k files). The banner is identical static HTML on every
page, and the committed docs/ is current with sources, so we inject the partial
directly into the already-rendered pages. The source partial is still wired into
quarto/_format.yml, so a future full render reproduces the same result.

Behavior:
- Inserts the contents of _include/deprecation-banner.html immediately after the
  opening <body ...> tag of each docs/**/*.html page.
- SKIPS reveal.js slide decks (pages containing class="reveal") — the banner
  must not appear on decks (matches the source wiring, which only touches
  format.html, not format.revealjs).
- Idempotent: a page that already contains the banner marker is left unchanged,
  so re-running never double-injects.

Usage:
  python3 plans/inject-banner.py [--check]
    (no args) inject into all eligible pages, print a summary.
    --check   report what would change without writing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PARTIAL = REPO / "_include" / "deprecation-banner.html"
DOCS = REPO / "docs"

MARKER = "samonl-deprecation-banner"
BODY_RE = re.compile(rb"<body\b[^>]*>", re.IGNORECASE)
# A reveal.js deck has the reveal container; this is our deck signal.
DECK_SIGNAL = b'class="reveal"'


def eligible_html_files() -> list[Path]:
    files = []
    for p in DOCS.rglob("*.html"):
        # Skip vendored site libraries — never content pages.
        if "site_libs" in p.parts:
            continue
        files.append(p)
    return sorted(files)


def main() -> int:
    check = "--check" in sys.argv[1:]

    if not PARTIAL.is_file():
        print(f"ERROR: partial not found: {PARTIAL}", file=sys.stderr)
        return 2
    snippet = PARTIAL.read_text(encoding="utf-8").strip() + "\n"
    snippet_bytes = snippet.encode("utf-8")

    injected = skipped_deck = skipped_already = skipped_nobody = 0
    changed_files: list[str] = []

    for path in eligible_html_files():
        raw = path.read_bytes()

        if DECK_SIGNAL in raw:
            skipped_deck += 1
            continue
        if MARKER.encode("utf-8") in raw:
            skipped_already += 1
            continue

        m = BODY_RE.search(raw)
        if not m:
            skipped_nobody += 1
            continue

        new = raw[: m.end()] + b"\n" + snippet_bytes + raw[m.end():]
        if not check:
            path.write_bytes(new)
        injected += 1
        changed_files.append(str(path.relative_to(REPO)))

    verb = "would inject" if check else "injected"
    print(f"{verb}: {injected}")
    print(f"skipped (reveal.js deck): {skipped_deck}")
    print(f"skipped (already has banner): {skipped_already}")
    print(f"skipped (no <body> tag): {skipped_nobody}")
    if check and changed_files:
        print("--- files that would change ---")
        for f in changed_files:
            print(f"  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

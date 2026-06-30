#!/usr/bin/env python3
"""Replace an already-injected deprecation banner block with the current
_include/deprecation-banner.html content, in place, across docs/**/*.html.

Use this after editing the partial (e.g. CSS tweaks) to propagate the change
to the already-rendered pages without a full quarto render. Idempotent: a page
whose banner already matches the partial is left byte-identical.

The injected block runs from the start-of-banner HTML comment through the
closing </script> of the banner IIFE. We match that span and swap it for the
freshly-read partial. reveal.js decks never had a banner, so they're skipped
naturally (no marker present).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PARTIAL = REPO / "_include" / "deprecation-banner.html"
DOCS = REPO / "docs"

# The banner block always starts with this exact comment line and ends with
# the IIFE's closing </script>. Both the old and new partials share the start
# comment text (domain aside) — match either sam.onl or samf.sh to be safe.
START_RE = re.compile(
    r"<!-- Deprecation banner: announces move to (?:sam\.onl|samf\.sh) with a per-page link\."
)
# End marker: the unique final line of the script, then its closing tag.
END_MARK = "})();\n</script>"


def main() -> int:
    check = "--check" in sys.argv[1:]
    partial = PARTIAL.read_text(encoding="utf-8").strip() + "\n"

    updated = skipped_nomarker = unchanged = 0
    changed = []
    for path in sorted(DOCS.rglob("*.html")):
        if "site_libs" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        m = START_RE.search(text)
        if not m:
            skipped_nomarker += 1
            continue
        start = m.start()
        end_idx = text.find(END_MARK, start)
        if end_idx == -1:
            skipped_nomarker += 1
            continue
        end = end_idx + len(END_MARK)
        old_block = text[start:end]
        new_text = text[:start] + partial.rstrip("\n") + text[end:]
        if new_text == text:
            unchanged += 1
            continue
        if not check:
            path.write_text(new_text, encoding="utf-8")
        updated += 1
        changed.append(str(path.relative_to(REPO)))

    verb = "would update" if check else "updated"
    print(f"{verb}: {updated}")
    print(f"unchanged (already current): {unchanged}")
    print(f"skipped (no banner marker): {skipped_nomarker}")
    if check:
        for c in changed:
            print(f"  {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

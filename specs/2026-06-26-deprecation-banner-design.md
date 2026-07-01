# Deprecation banner linking samforeman.me → sam.onl

> **Superseded 2026-06-29:** the new site was renamed `sam.onl` → `samf.sh`
> (repo `saforem2/sam.onl` → `saforem2/samf.sh`). The live banner, resolver,
> tests, and rendered `docs/` now use `samf.sh`. This document is kept as the
> original design record; read every `sam.onl` below as `samf.sh`.

**Date:** 2026-06-26
**Status:** Approved design, pending implementation plan

## Problem

The personal site at `https://samforeman.me` (this Quarto project, output committed
to `docs/`) is being deprecated in favor of a new site at `https://sam.onl` (a
separate Astro project, repo `saforem2/sam.onl`). We want a banner at the top of
**every page** of the old site announcing the move, and — critically — each page's
banner should link to the **corresponding page on the new site**, not just the new
homepage.

## Goals

- A deprecation banner appears at the top of every HTML article/landing/post/talk
  page on the old site.
- The banner's primary link points to the matching page on `sam.onl` for that
  specific old page.
- Pages with no counterpart on the new site degrade gracefully (link to the new
  homepage).
- The banner is dismissible and unobtrusive, theme-aware, and accessible.

## Non-goals

- Banner on the fullscreen reveal.js **slide decks** (`slides.html`). These are
  presentation surfaces where a top banner would overlay content; excluded.
- Server-side redirects / HTTP 301s. This is a visible in-page banner only.
- Any change to the new sam.onl site.
- Patching the already-rendered `docs/` HTML by script. The banner is added to
  Quarto source and applied via a normal `quarto render` (decision below).

## Key findings from investigation (these shaped the design)

1. **Clean global injection point exists.** Quarto's
   `format.html.include-before-body` in `quarto/_format.yml` is applied to every
   HTML page on render. This is the single hook for the banner. (The
   `format.revealjs` block is separate, so decks are naturally excluded.)

2. **The new site is an Astro file-routed site.** Routes come from
   `web/src/pages/**/*.{md,mdx}` (route = file path minus extension, with
   `/index` dropped). The authoritative route list is the file tree in that repo.

3. **sam.onl is a SPA that returns HTTP 200 for *every* URL**, including
   nonexistent ones. A "page not found" is detectable **only by response body**:
   the fallback is ~252,321 bytes with `<title>Sam Foreman</title>` and no `<h1>`;
   real pages are larger with a real title/`<h1>`. (Consequence: we cannot verify
   target pages by HTTP status — the mapping was verified by comparing response
   bodies.)

4. **Route casing is PRESERVED on the new site — it is NOT lowercased.**
   `/posts/AuroraGPT/checkpoints/` is a real page; the lowercased
   `/posts/auroragpt/checkpoints/` returns the 404 fallback. Because the **old**
   site already uses this same casing, the old→new mapping is **identity (the same
   path) for ~95% of pages**. Only a handful of renamed slugs, the slide decks,
   and one orphaned page need special handling.

   > Note on process: an earlier homepage scrape (via WebFetch) appeared to show
   > lowercased paths, and a subagent reported the old Quarto sitemap as if it were
   > the new site. Both were wrong. The casing behavior above was confirmed
   > empirically against the live site by comparing response bodies, and is the
   > basis for this design.

## Architecture

A single self-contained HTML partial:

```
_include/deprecation-banner.html
```

containing:
- the banner markup,
- a scoped `<style>` block (theme-aware, no external assets),
- a `<script>` block that resolves the current page to its sam.onl counterpart at
  load time and wires up dismiss/persistence.

Wired into the site via `quarto/_format.yml`:

```yaml
format:
  html:
    include-before-body:
      - _include/deprecation-banner.html
      # (existing GTM noscript entry stays)
```

The partial is shared by all pages, so the per-page target URL is resolved
**client-side** from `location.pathname`. Applied on the next `quarto render`.

### Why client-side resolution

The same partial is injected into every page; it cannot know its own path at
include time. Reading `location.pathname` in the browser and mapping it is the
simplest correct approach, needs no per-page Quarto templating, and keeps the
whole feature in one file.

## URL resolution

`resolve(pathname) -> { url, hasPageLink }`

1. **Normalize** `pathname` to a `key`: strip a trailing `/index.html` or
   `.html`, then strip any trailing slash. Empty → `/`.
2. **Home:** if `key` is `/` (or empty), return `{ url: 'https://sam.onl', hasPageLink: false }`.
3. **Override table (exact match on `key`) — checked before the slides rule.**
   This table contains every non-identity case (all targets live-verified as REAL,
   all "bare directory" / lowercased alternatives verified as the 404 fallback):

   | Old `key` | New target |
   |---|---|
   | `/posts/AuroraGPT/determinstic-flash-attn` | `/posts/AuroraGPT/determinstic-flash-attn/deterministic-flash-attn` |
   | `/talks/hpc-user-forum` | `/talks/hpc-user-forum/AuroraGPT` |
   | `/talks/alcf-hpc-workshop-2024` | `/talks/alcf-hpc-workshop-2024/alcf-hpc-workshop-2024` |
   | `/talks/aurora-gpt-fm-for-electric-grid` | `/talks/aurora-gpt-fm-for-electric-grid/AuroraGPT-FM-for-electric-grid` |
   | `/talks/AuroraGPT/alcf-hpc-workshop-2024` | `/talks/AuroraGPT/alcf-hpc-workshop-2024/AuroraGPT-ALCF-Hands-On-HPC-Workshop-2024` |
   | `/talks/lattice23` | *(no page — see step 6)* |

   The five renamed entries also have a `/slides` sibling on the old site
   (`/talks/hpc-user-forum/slides`, etc.). The implementation handles these by
   applying the slides rule (step 5) **first to compute the base `key`**, then the
   override lookup — i.e. order is: normalize → strip `/slides` if present → look up
   override → else identity. (Either ordering works as long as the renamed-talk
   `/slides` paths resolve to the renamed target; the plan will pick one and test
   it. `lattice23/slides` must resolve to the no-page case.)

4. **`lattice23`** (and `/talks/lattice23/slides`): no counterpart on sam.onl. It
   exists on the new site only as an external deck link in
   `web/src/legacy-talks.ts` (`https://saforem2.github.io/lattice23`), not as a
   sam.onl page. Resolve to `{ url: 'https://sam.onl', hasPageLink: false }`.

5. **Slides rule:** if `key` ends in `/slides`, its base is the parent path
   (drop `/slides`). The new site has no standalone slide routes, so a slide deck
   maps to its parent talk landing page (`/talks/2025/09/24/slides` →
   `/talks/2025/09/24`) and `/ideas/slides` → `/ideas`. Verified: those parent
   pages are REAL on sam.onl.

6. **Default — identity:** `{ url: 'https://sam.onl' + key, hasPageLink: true }`.
   Correct for every casing-preserved page (e.g. `/posts/AuroraGPT/checkpoints`,
   `/posts/ai-for-physics/l2hmc-qcd/2dU1`, `/talks/AuroraGPT-SIAM25`,
   `/posts/2025/06/14`, `/about`, `/now`, `/projects`, …).

The override map is a small literal object in the script; amending it later is a
one-line change.

## Banner content & behavior

- **Text (tunable):** `📦 This site has moved to sam.onl. → View this page on the
  new site`
  - `sam.onl` → link to `https://sam.onl` (home).
  - `→ View this page on the new site` → link to the resolved per-page URL.
- **No-match pages** (`/`, `lattice23`): omit the per-page arrow link; show only
  the deprecation text with the `sam.onl` home link.
- **Dismissible:** a `✕` button hides the banner and records dismissal in
  `localStorage` (e.g. key `samonl-deprecation-dismissed`) so it stays dismissed
  across pages and visits.
- **Accessibility:** `role="region"` with an `aria-label` (e.g. "Site deprecation
  notice"); the dismiss button is a real `<button>` with an `aria-label`, keyboard
  focusable; links are real `<a>` elements.

## Placement & styling

- **Placement:** sticky strip at the very top of the viewport. The site navbar is
  near the top of `<body>` and may be fixed/sticky; the banner must not overlap it.
  The implementation will offset/adjust the navbar (or the banner's position) so
  both are visible, and **verify visually against the rendered DOM** (the
  source+render workflow makes this checkable).
- **Styling:** self-contained scoped CSS using the site's existing light/dark
  theme CSS variables, with `prefers-color-scheme` fallbacks. Subtle accent
  background, not visually jarring. No external assets or fonts.

## Testing / verification

`quarto render` a representative subset and confirm in the rendered HTML / browser:

- **Identity page** (e.g. `/posts/AuroraGPT/checkpoints`, `/about`): arrow links to
  the same path on sam.onl, and that page is real.
- **Each renamed page** (the 5 override rows): arrow links to the renamed target.
- **A slide deck landing** and `/ideas/slides`: arrow links to the parent page.
- **`lattice23`** and **home**: no arrow link; only the sam.onl home link shows.
- **Dismiss** hides the banner and persists across navigation.
- **reveal.js decks** (`slides.html`) do **not** show the banner and are visually
  unaffected.
- Banner and navbar do not overlap in light and dark themes.

## Files touched

- `_include/deprecation-banner.html` (new) — markup + style + script.
- `quarto/_format.yml` (edit) — add the partial to
  `format.html.include-before-body`.
- Re-rendered `docs/**/*.html` output from `quarto render` (generated).

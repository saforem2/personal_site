# Deprecation Banner Implementation Plan

> **Superseded 2026-06-29:** the new site was renamed `sam.onl` → `samf.sh`
> (repo `saforem2/sam.onl` → `saforem2/samf.sh`). The live banner, resolver,
> tests, and rendered `docs/` now use `samf.sh`. This plan is kept as the
> original implementation record; read every `sam.onl` below as `samf.sh`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dismissible banner to the top of every page on the old Quarto site (`samforeman.me`) announcing the move to `sam.onl`, with a link to the *corresponding* page on the new site.

**Architecture:** A single self-contained HTML partial (`_include/deprecation-banner.html`: markup + scoped `<style>` + `<script>`) is injected into every HTML page via Quarto's `format.html.include-before-body`. The per-page target URL is computed client-side from `location.pathname` using identity-mapping plus a small override table (the new Astro site preserves path casing, so most pages map to the same path). The banner is excluded from reveal.js slide decks because only `format.html` is modified.

**Tech Stack:** Quarto 1.9.18, plain HTML/CSS/vanilla JS. No build step, no dependencies, no external assets.

## Global Constraints

- **Old site** = this repo; Quarto output committed to `docs/` (`output-dir: docs` in `_quarto.yml`). Live host: `https://samforeman.me`.
- **New site** = `https://sam.onl` (separate Astro repo `saforem2/sam.onl`).
- **New site preserves URL casing** — do NOT lowercase paths. Mapping is identity except for the override table below.
- **New site is a SPA returning HTTP 200 for every URL** — never rely on HTTP status to validate a target; this was verified by response-body comparison.
- No external assets/fonts/CDN in the banner. Self-contained only.
- Theme-aware via existing Bootstrap runtime CSS variables: `--bs-body-bg`, `--bs-body-color`, `--bs-border-color`, `--bs-link-color`, `--bs-emphasis-color` (all confirmed present in both light & dark compiled CSS). Banner `z-index` must be ≥ 1040 (site navbar `.fixed-top` is `z-index:1030`).
- **Override table (old `key` → new sam.onl path), all targets live-verified REAL; do not alter without re-verifying:**
  | Old key | New path |
  |---|---|
  | `/posts/AuroraGPT/determinstic-flash-attn` | `/posts/AuroraGPT/determinstic-flash-attn/deterministic-flash-attn` |
  | `/talks/hpc-user-forum` | `/talks/hpc-user-forum/AuroraGPT` |
  | `/talks/alcf-hpc-workshop-2024` | `/talks/alcf-hpc-workshop-2024/alcf-hpc-workshop-2024` |
  | `/talks/aurora-gpt-fm-for-electric-grid` | `/talks/aurora-gpt-fm-for-electric-grid/AuroraGPT-FM-for-electric-grid` |
  | `/talks/AuroraGPT/alcf-hpc-workshop-2024` | `/talks/AuroraGPT/alcf-hpc-workshop-2024/AuroraGPT-ALCF-Hands-On-HPC-Workshop-2024` |
  | `/talks/lattice23` | *(no page → home, no per-page link)* |
- **Banner copy:** `📦 This site has moved to sam.onl. → View this page on the new site` (the `sam.onl` substring links to `https://sam.onl`; the `→ View this page on the new site` substring links to the resolved per-page URL).
- **localStorage dismissal key:** `samonl-deprecation-dismissed`.

---

## File Structure

- `_include/deprecation-banner.html` (new) — the entire feature: markup, scoped CSS, resolution + dismiss JS.
- `quarto/_format.yml` (modify) — add the partial to `format.html.include-before-body`.
- `plans/url-resolver.test.mjs` (new, temporary) — Node test for the pure URL-resolution function, extracted as a standalone module during testing.
- `plans/url-resolver.mjs` (new, temporary) — the resolution logic as an importable ES module for testing; its body is copied verbatim into the partial's `<script>`. Kept in `plans/` so it is never published from `docs/`.

> **Why a separate test module:** the resolution logic is the only part with real branching/correctness risk and a known correct-answer table. We TDD it as a pure function in Node, then paste the identical function body into the partial. The DOM/CSS wiring is verified by rendering and visual inspection (Task 4), not unit tests.

---

## Task 1: URL resolver — identity, overrides, slides, no-match (TDD as pure function)

**Files:**
- Create: `plans/url-resolver.mjs`
- Test: `plans/url-resolver.test.mjs`

**Interfaces:**
- Produces: `resolveSamOnl(pathname: string) -> { url: string, hasPageLink: boolean }`
  - `url`: absolute `https://sam.onl…` URL.
  - `hasPageLink`: `true` when there is a real per-page counterpart (show the arrow link); `false` for home and no-match pages (show only the home link).

- [ ] **Step 1: Write the failing test**

Create `plans/url-resolver.test.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resolveSamOnl } from './url-resolver.mjs';

const HOME = 'https://sam.onl';

// --- identity (casing preserved) ---
test('identity: simple post keeps path', () => {
  assert.deepEqual(resolveSamOnl('/posts/AuroraGPT/checkpoints/index.html'),
    { url: HOME + '/posts/AuroraGPT/checkpoints', hasPageLink: true });
});
test('identity: mixed-case nested path preserved', () => {
  assert.deepEqual(resolveSamOnl('/posts/ai-for-physics/l2hmc-qcd/2dU1/'),
    { url: HOME + '/posts/ai-for-physics/l2hmc-qcd/2dU1', hasPageLink: true });
});
test('identity: top-level page', () => {
  assert.deepEqual(resolveSamOnl('/about/index.html'),
    { url: HOME + '/about', hasPageLink: true });
});
test('identity: trailing slash and no extension', () => {
  assert.deepEqual(resolveSamOnl('/projects/'),
    { url: HOME + '/projects', hasPageLink: true });
});

// --- home ---
test('home: root', () => {
  assert.deepEqual(resolveSamOnl('/'),
    { url: HOME, hasPageLink: false });
});
test('home: bare index.html', () => {
  assert.deepEqual(resolveSamOnl('/index.html'),
    { url: HOME, hasPageLink: false });
});

// --- override: renamed slugs ---
test('override: determinstic-flash-attn renamed file', () => {
  assert.deepEqual(resolveSamOnl('/posts/AuroraGPT/determinstic-flash-attn/index.html'),
    { url: HOME + '/posts/AuroraGPT/determinstic-flash-attn/deterministic-flash-attn', hasPageLink: true });
});
test('override: hpc-user-forum', () => {
  assert.deepEqual(resolveSamOnl('/talks/hpc-user-forum/'),
    { url: HOME + '/talks/hpc-user-forum/AuroraGPT', hasPageLink: true });
});
test('override: alcf-hpc-workshop-2024', () => {
  assert.deepEqual(resolveSamOnl('/talks/alcf-hpc-workshop-2024/index.html'),
    { url: HOME + '/talks/alcf-hpc-workshop-2024/alcf-hpc-workshop-2024', hasPageLink: true });
});
test('override: aurora-gpt-fm-for-electric-grid', () => {
  assert.deepEqual(resolveSamOnl('/talks/aurora-gpt-fm-for-electric-grid/'),
    { url: HOME + '/talks/aurora-gpt-fm-for-electric-grid/AuroraGPT-FM-for-electric-grid', hasPageLink: true });
});
test('override: nested AuroraGPT/alcf-hpc-workshop-2024', () => {
  assert.deepEqual(resolveSamOnl('/talks/AuroraGPT/alcf-hpc-workshop-2024/index.html'),
    { url: HOME + '/talks/AuroraGPT/alcf-hpc-workshop-2024/AuroraGPT-ALCF-Hands-On-HPC-Workshop-2024', hasPageLink: true });
});

// --- override + slides: renamed talk's slide deck resolves to renamed target ---
test('override slides: hpc-user-forum/slides -> renamed target', () => {
  assert.deepEqual(resolveSamOnl('/talks/hpc-user-forum/slides.html'),
    { url: HOME + '/talks/hpc-user-forum/AuroraGPT', hasPageLink: true });
});
test('override slides: nested AuroraGPT workshop slides -> renamed target', () => {
  assert.deepEqual(resolveSamOnl('/talks/AuroraGPT/alcf-hpc-workshop-2024/slides.html'),
    { url: HOME + '/talks/AuroraGPT/alcf-hpc-workshop-2024/AuroraGPT-ALCF-Hands-On-HPC-Workshop-2024', hasPageLink: true });
});

// --- slides -> parent landing (non-renamed) ---
test('slides: dated talk slide deck -> landing', () => {
  assert.deepEqual(resolveSamOnl('/talks/2025/09/24/slides.html'),
    { url: HOME + '/talks/2025/09/24', hasPageLink: true });
});
test('slides: ideas/slides -> ideas', () => {
  assert.deepEqual(resolveSamOnl('/ideas/slides.html'),
    { url: HOME + '/ideas', hasPageLink: true });
});

// --- no match -> home, no per-page link ---
test('no-match: lattice23', () => {
  assert.deepEqual(resolveSamOnl('/talks/lattice23/index.html'),
    { url: HOME, hasPageLink: false });
});
test('no-match: lattice23 slides', () => {
  assert.deepEqual(resolveSamOnl('/talks/lattice23/slides.html'),
    { url: HOME, hasPageLink: false });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node --test plans/url-resolver.test.mjs`
Expected: FAIL — `Cannot find module './url-resolver.mjs'` (or all tests error).

- [ ] **Step 3: Write the minimal implementation**

Create `plans/url-resolver.mjs`. The resolution order is: normalize → strip `/slides` to get a base key → look up base key in override table → else identity. This ordering makes renamed-talk slide decks resolve to the renamed target, and `lattice23/slides` resolve to the no-match case.

```js
// Pure resolver: old samforeman.me pathname -> { url, hasPageLink } on sam.onl.
// NOTE: keep this body byte-for-byte identical to the <script> in
// _include/deprecation-banner.html. New site PRESERVES casing — do not lowercase.
export function resolveSamOnl(pathname) {
  const HOME = 'https://sam.onl';

  // 1. Normalize: drop /index.html or trailing .html, then trailing slash.
  let key = pathname || '/';
  if (key.endsWith('/index.html')) key = key.slice(0, -'/index.html'.length);
  else if (key.endsWith('.html')) key = key.slice(0, -'.html'.length);
  if (key.length > 1 && key.endsWith('/')) key = key.slice(0, -1);
  if (key === '') key = '/';

  // 2. Strip a trailing /slides segment to get the base content key.
  let base = key;
  if (base === '/slides') base = '/';
  else if (base.endsWith('/slides')) base = base.slice(0, -'/slides'.length);

  // 3. Override table (exact match on base). null = no counterpart -> home.
  const OVERRIDES = {
    '/posts/AuroraGPT/determinstic-flash-attn':
      '/posts/AuroraGPT/determinstic-flash-attn/deterministic-flash-attn',
    '/talks/hpc-user-forum': '/talks/hpc-user-forum/AuroraGPT',
    '/talks/alcf-hpc-workshop-2024':
      '/talks/alcf-hpc-workshop-2024/alcf-hpc-workshop-2024',
    '/talks/aurora-gpt-fm-for-electric-grid':
      '/talks/aurora-gpt-fm-for-electric-grid/AuroraGPT-FM-for-electric-grid',
    '/talks/AuroraGPT/alcf-hpc-workshop-2024':
      '/talks/AuroraGPT/alcf-hpc-workshop-2024/AuroraGPT-ALCF-Hands-On-HPC-Workshop-2024',
    '/talks/lattice23': null,
  };
  if (Object.prototype.hasOwnProperty.call(OVERRIDES, base)) {
    const target = OVERRIDES[base];
    if (target === null) return { url: HOME, hasPageLink: false };
    return { url: HOME + target, hasPageLink: true };
  }

  // 4. Home.
  if (base === '/') return { url: HOME, hasPageLink: false };

  // 5. Identity (casing preserved).
  return { url: HOME + base, hasPageLink: true };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `node --test plans/url-resolver.test.mjs`
Expected: PASS — all tests pass (18 passing).

- [ ] **Step 5: Commit**

```bash
git add plans/url-resolver.mjs plans/url-resolver.test.mjs
git commit -m "test: add sam.onl URL resolver with verified mapping table"
```

---

## Task 2: Build the banner partial (markup + scoped CSS + JS)

**Files:**
- Create: `_include/deprecation-banner.html`

**Interfaces:**
- Consumes: the `resolveSamOnl` function body from Task 1 (pasted verbatim into the `<script>`).
- Produces: a standalone HTML fragment safe to inject at the top of `<body>` on every page.

- [ ] **Step 1: Write the partial**

Create `_include/deprecation-banner.html` with the following exact content. The `<script>` is an IIFE that builds the banner only after checking the dismissal flag, so dismissed users never see a flash. CSS uses Bootstrap runtime variables (theme-aware) with hardcoded fallbacks; `z-index:1040` sits above the `.fixed-top` navbar (1030). The banner is `position: sticky; top: 0` so it stays pinned but scrolls within flow.

```html
<!-- Deprecation banner: announces move to sam.onl with a per-page link.
     Injected on every HTML page via format.html.include-before-body.
     Self-contained: no external assets. Excluded from reveal.js decks. -->
<style>
  #samonl-deprecation-banner {
    position: sticky;
    top: 0;
    z-index: 1040;
    box-sizing: border-box;
    width: 100%;
    margin: 0;
    padding: 0.5rem 2.5rem 0.5rem 1rem;
    font-size: 0.9rem;
    line-height: 1.4;
    text-align: center;
    background: var(--bs-body-bg, #1c1c1c);
    color: var(--bs-body-color, #dee2e6);
    border-bottom: 1px solid var(--bs-border-color, rgba(131,131,131,0.7));
  }
  #samonl-deprecation-banner a {
    color: var(--bs-link-color, #82b1ff);
    text-decoration: underline;
    font-weight: 600;
  }
  #samonl-deprecation-banner a:hover { text-decoration: none; }
  #samonl-deprecation-banner .samonl-sep { opacity: 0.6; margin: 0 0.4rem; }
  #samonl-deprecation-dismiss {
    position: absolute;
    top: 50%;
    right: 0.6rem;
    transform: translateY(-50%);
    border: 0;
    background: transparent;
    color: inherit;
    font-size: 1.1rem;
    line-height: 1;
    cursor: pointer;
    padding: 0.25rem 0.4rem;
    opacity: 0.7;
  }
  #samonl-deprecation-dismiss:hover { opacity: 1; }
  #samonl-deprecation-dismiss:focus-visible { outline: 2px solid var(--bs-link-color, #82b1ff); }
</style>
<script>
(function () {
  var STORAGE_KEY = 'samonl-deprecation-dismissed';
  try { if (localStorage.getItem(STORAGE_KEY) === '1') return; } catch (e) {}

  function resolveSamOnl(pathname) {
    var HOME = 'https://sam.onl';
    var key = pathname || '/';
    if (key.endsWith('/index.html')) key = key.slice(0, -'/index.html'.length);
    else if (key.endsWith('.html')) key = key.slice(0, -'.html'.length);
    if (key.length > 1 && key.endsWith('/')) key = key.slice(0, -1);
    if (key === '') key = '/';

    var base = key;
    if (base === '/slides') base = '/';
    else if (base.endsWith('/slides')) base = base.slice(0, -'/slides'.length);

    var OVERRIDES = {
      '/posts/AuroraGPT/determinstic-flash-attn':
        '/posts/AuroraGPT/determinstic-flash-attn/deterministic-flash-attn',
      '/talks/hpc-user-forum': '/talks/hpc-user-forum/AuroraGPT',
      '/talks/alcf-hpc-workshop-2024':
        '/talks/alcf-hpc-workshop-2024/alcf-hpc-workshop-2024',
      '/talks/aurora-gpt-fm-for-electric-grid':
        '/talks/aurora-gpt-fm-for-electric-grid/AuroraGPT-FM-for-electric-grid',
      '/talks/AuroraGPT/alcf-hpc-workshop-2024':
        '/talks/AuroraGPT/alcf-hpc-workshop-2024/AuroraGPT-ALCF-Hands-On-HPC-Workshop-2024',
      '/talks/lattice23': null
    };
    if (Object.prototype.hasOwnProperty.call(OVERRIDES, base)) {
      var target = OVERRIDES[base];
      if (target === null) return { url: HOME, hasPageLink: false };
      return { url: HOME + target, hasPageLink: true };
    }
    if (base === '/') return { url: HOME, hasPageLink: false };
    return { url: HOME + base, hasPageLink: true };
  }

  function build() {
    var r = resolveSamOnl(location.pathname);
    var banner = document.createElement('div');
    banner.id = 'samonl-deprecation-banner';
    banner.setAttribute('role', 'region');
    banner.setAttribute('aria-label', 'Site deprecation notice');

    var msg = document.createElement('span');
    msg.appendChild(document.createTextNode('📦 This site has moved to '));
    var home = document.createElement('a');
    home.href = 'https://sam.onl';
    home.textContent = 'sam.onl';
    msg.appendChild(home);
    msg.appendChild(document.createTextNode('.'));

    if (r.hasPageLink) {
      var sep = document.createElement('span');
      sep.className = 'samonl-sep';
      sep.textContent = '→'; // →
      msg.appendChild(sep);
      var page = document.createElement('a');
      page.href = r.url;
      page.textContent = 'View this page on the new site';
      msg.appendChild(page);
    }
    banner.appendChild(msg);

    var btn = document.createElement('button');
    btn.id = 'samonl-deprecation-dismiss';
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Dismiss deprecation notice');
    btn.textContent = '✕'; // ✕
    btn.addEventListener('click', function () {
      banner.remove();
      try { localStorage.setItem(STORAGE_KEY, '1'); } catch (e) {}
    });
    banner.appendChild(btn);

    document.body.insertBefore(banner, document.body.firstChild);
  }

  if (document.body) build();
  else document.addEventListener('DOMContentLoaded', build);
})();
</script>
```

- [ ] **Step 2: Sanity-check the JS parses**

Run: `node --check <(sed -n '/<script>/,/<\/script>/p' _include/deprecation-banner.html | sed '1d;$d')`
Expected: no output, exit code 0 (the extracted script is valid JS).

> If your shell lacks process substitution, instead run:
> `sed -n '/<script>/,/<\/script>/p' _include/deprecation-banner.html | sed '1d;$d' > /tmp/banner.js && node --check /tmp/banner.js && echo OK`
> Expected: `OK`.

- [ ] **Step 3: Verify the resolver body matches Task 1**

Confirm the `resolveSamOnl` logic in the partial is identical (apart from `var` vs `export function` and `const`→`var`) to `plans/url-resolver.mjs`. Specifically check the OVERRIDES table has all 6 keys and the same target strings.

Run: `grep -c "AuroraGPT-ALCF-Hands-On-HPC-Workshop-2024\|AuroraGPT-FM-for-electric-grid\|deterministic-flash-attn\|hpc-user-forum/AuroraGPT\|alcf-hpc-workshop-2024/alcf-hpc-workshop-2024\|'/talks/lattice23': null" _include/deprecation-banner.html`
Expected: `6`

- [ ] **Step 4: Commit**

```bash
git add _include/deprecation-banner.html
git commit -m "feat: add sam.onl deprecation banner partial"
```

---

## Task 3: Wire the partial into Quarto HTML output

**Files:**
- Modify: `quarto/_format.yml` (the `format.html.include-before-body` block, around lines 138–142)

**Interfaces:**
- Consumes: `_include/deprecation-banner.html` from Task 2.
- Produces: the banner injected into every rendered `format: html` page (not reveal.js).

- [ ] **Step 1: Inspect the current include-before-body block**

Run: `grep -n "include-before-body" quarto/_format.yml`
Expected: two matches — one under `html:` (~line 138) and one under `revealjs:` (~line 250). **Only modify the `html:` one.**

The current `html:` block is:

```yaml
    include-before-body:
      - text: |
          <!-- Google Tag Manager (noscript) -->
          <noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-TC329HJ" height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
          <!-- End Google Tag Manager (noscript) -->
```

- [ ] **Step 2: Add the partial as the first entry**

Edit the `html:` `include-before-body` block so the banner file is included before the GTM noscript entry. Result:

```yaml
    include-before-body:
      - _include/deprecation-banner.html
      - text: |
          <!-- Google Tag Manager (noscript) -->
          <noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-TC329HJ" height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
          <!-- End Google Tag Manager (noscript) -->
```

Leave the `revealjs:` `include-before-body` block unchanged (decks must not get the banner).

- [ ] **Step 3: Verify the YAML edit**

Run: `grep -n "deprecation-banner" quarto/_format.yml`
Expected: exactly one match, under the `html:` block (line number below the first `include-before-body` ~138, well above the revealjs one ~250).

- [ ] **Step 4: Commit**

```bash
git add quarto/_format.yml
git commit -m "build: inject deprecation banner into HTML pages via include-before-body"
```

---

## Task 4: Render a representative subset and verify behavior in-browser

**Files:**
- No source changes. Generates `docs/**` output and verifies it.

**Interfaces:**
- Consumes: Tasks 2 and 3.
- Produces: confirmed-correct rendered pages; the visual/layout sign-off the spec requires.

- [ ] **Step 1: Render a representative subset**

Render one page per resolution branch (identity, renamed post, renamed talk, dated-talk slides, ideas/slides, lattice23 no-match, home) plus confirm decks. `_freeze` is present so this is fast.

Run:
```bash
quarto render index.qmd about/index.qmd ideas/index.qmd \
  "posts/AuroraGPT/determinstic-flash-attn/index.qmd" \
  "talks/hpc-user-forum/index.qmd" \
  "talks/2025/09/24/index.qmd" 2>&1 | tail -20
```
Expected: renders complete without error (`Output created: ...` lines). If a specific source path differs, locate it with `git ls-files | grep -i <slug>` and substitute; the goal is to render at least: home, one identity page, one renamed post, one renamed talk, and one talk that has a slide deck.

> Note: `lattice23` and `ideas/slides` may not have current `.qmd` sources (they exist in the committed `docs/` output). If their sources are absent, verify those two cases with the Node resolver instead (Step 3) — the JS path-resolution is identical, so a unit-level check is authoritative for the link target.

- [ ] **Step 2: Confirm the banner HTML is present in rendered output and absent from decks**

Run:
```bash
echo "html page has banner:"; grep -l "samonl-deprecation-banner" docs/about/index.html
echo "deck does NOT have banner:"; grep -L "samonl-deprecation-banner" docs/talks/2025/09/24/slides.html
```
Expected: first prints `docs/about/index.html`; second prints `docs/talks/2025/09/24/slides.html` (i.e. the deck does **not** contain the banner). If `docs/talks/2025/09/24/slides.html` wasn't regenerated, check any existing `docs/**/slides.html` instead.

- [ ] **Step 3: Verify link targets with the resolver (authoritative for URL correctness)**

Run:
```bash
node --test plans/url-resolver.test.mjs
```
Expected: PASS (18 tests). This is the source of truth for every link target, including `lattice23` and `ideas/slides`.

- [ ] **Step 4: Visual / layout verification in a browser**

Start a local preview and open pages in a browser (Playwright MCP or manual):
```bash
quarto preview --no-browser --port 4209
```
Then load `http://localhost:4209/about/index.html` and check, in BOTH light and dark theme (toggle in navbar):
1. Banner appears at the very top, full width, readable contrast.
2. Banner and the fixed navbar (`#quarto-header.fixed-top`) do **not** overlap or hide each other. The navbar uses Headroom.js (hides on scroll-down) — scroll down and back up and confirm no visual collision or content obscured at rest.
3. The `sam.onl` link points to `https://sam.onl`; the `View this page on the new site` link points to `https://sam.onl/about`.
4. Click ✕ → banner disappears; navigate to another page → banner stays gone (localStorage). Clear the `samonl-deprecation-dismissed` localStorage key to re-enable.
5. Load `/talks/hpc-user-forum/index.html` → the per-page link points to `https://sam.onl/talks/hpc-user-forum/AuroraGPT`.
6. Load the home page (`/index.html`) → only the `sam.onl` link shows (no "View this page" arrow).

**If overlap with the fixed navbar is observed in step 2:** the fix is to add, inside the banner's `<style>`, a rule pushing the fixed header down by the banner height — e.g. give the banner a fixed height and add `body { } #quarto-header.fixed-top { top: <banner-height>; }`. Prefer a JS-measured offset: after inserting the banner, set `document.getElementById('quarto-header')?.style.setProperty('top', banner.offsetHeight + 'px')` and restore it on dismiss. Only add this if the visual check shows a collision; keep the change minimal and re-run steps 1–2.

- [ ] **Step 5: Commit any layout fix and the rendered subset**

```bash
git add _include/deprecation-banner.html docs
git commit -m "test: verify deprecation banner renders and links correctly"
```

> If no layout fix was needed and you prefer not to commit a partial `docs/` render, you may skip staging `docs` here and do the full render+commit in Task 5 instead.

---

## Task 5: Full site render and final commit

**Files:**
- Regenerates all of `docs/**`.

**Interfaces:**
- Consumes: Tasks 2–4 (banner verified correct).
- Produces: the deployable site with the banner on every HTML page.

- [ ] **Step 1: Full render**

Run: `quarto render 2>&1 | tail -25`
Expected: completes with no errors; many `Output created:` lines. (`_freeze` + `execute: cache` make this fast; computational cells are not re-executed.)

- [ ] **Step 2: Spot-check banner coverage across many pages**

Run:
```bash
echo "pages WITH banner:"; grep -rl "samonl-deprecation-banner" docs --include=index.html | wc -l
echo "decks WITHOUT banner (should be >0 and none listed below):"; grep -rl "samonl-deprecation-banner" docs --include=slides.html | wc -l
```
Expected: first count is large (≈ all article/landing pages, ~80+); second count is `0` (no slide deck contains the banner).

- [ ] **Step 3: Review the diff scope**

Run: `git status --short | head -30 && echo "---" && git diff --stat | tail -5`
Expected: changes confined to `docs/**` (rendered output) plus the already-committed source files. No unexpected source modifications.

- [ ] **Step 4: Commit the full render**

```bash
git add docs
git commit -m "docs: render site with sam.onl deprecation banner"
```

- [ ] **Step 5: Clean up temporary test files**

The `plans/url-resolver.mjs` + `.test.mjs` are scaffolding for TDD; the canonical logic now lives in the partial. Keep them in `plans/` as regression tests for the mapping (they never publish from `docs/`), OR remove them if you prefer a clean tree:

```bash
# Option A — keep as regression tests (recommended): do nothing.
# Option B — remove:
# git rm plans/url-resolver.mjs plans/url-resolver.test.mjs
# git commit -m "chore: remove temporary resolver test scaffolding"
```

---

## Self-Review Notes

- **Spec coverage:** injection via `include-before-body` (Task 3) ✓; client-side identity+override resolution (Task 1) ✓; 6-row override table verbatim (Tasks 1–2) ✓; slides→landing rule (Task 1) ✓; lattice23/home no-link fallback (Task 1) ✓; concise dismissible banner with per-page + home links (Task 2) ✓; localStorage persistence (Task 2) ✓; theme-aware scoped styling via `--bs-*` vars (Task 2) ✓; reveal.js decks excluded + verified (Tasks 3–5) ✓; sticky placement + navbar non-overlap with visual verification and a concrete fix path (Task 4) ✓; testing across representative subset (Tasks 4–5) ✓.
- **No lowercasing** anywhere — identity preserves casing per the verified finding. ✓
- **Type/name consistency:** `resolveSamOnl(pathname) -> {url, hasPageLink}` used identically in Tasks 1, 2, 4. Storage key `samonl-deprecation-dismissed` and element id `samonl-deprecation-banner` consistent across tasks. ✓
- **Location note:** spec at `specs/`, plan at `plans/` (not `docs/superpowers/...`) because `docs/` is Quarto's published output dir in this repo.

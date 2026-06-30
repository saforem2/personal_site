// Pure resolver: old samforeman.me pathname -> { url, hasPageLink } on samf.sh.
// NOTE: keep this body byte-for-byte identical to the <script> in
// _include/deprecation-banner.html. New site PRESERVES casing — do not lowercase.
export function resolveSamOnl(pathname) {
  const HOME = 'https://samf.sh';

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

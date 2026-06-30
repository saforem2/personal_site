import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resolveSamOnl } from './url-resolver.mjs';

const HOME = 'https://samf.sh';

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

import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// The same source works at a project URL, a domain root, or localhost.
export default defineConfig({
  site: process.env.DOCS_SITE || 'https://zpy52.github.io',
  base: process.env.DOCS_BASE || '/mlzo',
  trailingSlash: 'always',
  integrations: [starlight({
    title: 'MLZO',
    description: 'The complete guide to MLZO 0.1.0a1: deferred PyTorch training, data, checkpoints, and single-worker execution.',
    favicon: '/favicon.svg',
    logo: { src: './src/assets/mark.svg', replacesTitle: false },
    social: [{ icon: 'github', label: 'Source on GitHub', href: 'https://github.com/zPy52/mlzo' }],
    customCss: ['./src/styles/custom.css'],
    components: { SiteTitle: './src/components/SiteTitle.astro' },
    sidebar: [
      { label: 'Start here', items: [
        { label: 'Welcome to MLZO', slug: '' },
        { slug: 'start/installation' }, { slug: 'start/quickstart' }, { slug: 'start/concepts' },
      ] },
      { label: 'Training', items: [
        { slug: 'guides/models-and-planning' }, { slug: 'guides/data' },
        { slug: 'guides/training' }, { slug: 'guides/custom-training' },
        { slug: 'guides/optimizers' }, { slug: 'guides/checkpoints' },
        { slug: 'guides/inference' }, { slug: 'guides/transforms' }, { slug: 'guides/dqn' },
      ] },
      { label: 'Remote execution', items: [
        { slug: 'remote/setup' }, { slug: 'remote/submission' }, { slug: 'remote/operations' },
      ] },
      { label: 'Reference', collapsed: true, items: [
        { slug: 'reference/configuration' }, { slug: 'reference/cli' },
        { label: 'Python API', items: [{ autogenerate: { directory: 'reference/api' } }] },
      ] },
      { label: 'Release & help', items: [
        { slug: 'help/troubleshooting' }, { slug: 'help/support' }, { slug: 'help/release' },
      ] },
    ],
    expressiveCode: { themes: ['github-dark', 'github-light'], styleOverrides: { borderRadius: '0.65rem' } },
  })],
});

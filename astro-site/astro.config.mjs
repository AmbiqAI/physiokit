import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';
import { heliaStarlight } from '@ambiqai/helia-ui/starlight';
import { sections } from './src/navigation.mjs';

export default defineConfig({
  site: 'https://ambiqai.github.io',
  base: '/physiokit',
  integrations: [
    react(),
    starlight({
      title: 'physioKIT',
      description: 'Python tools for physiological signal processing.',
      favicon: '/assets/favicon.png',
      components: { Hero: './src/components/HomeHero.astro' },
      customCss: ['./src/styles/site.css'],
      plugins: [heliaStarlight({
        accent: 'kit-physio',
        sections,
        sidebar: 'always',
        header: {
          title: 'physioKIT',
          hub: { label: 'HELIA', href: 'https://ambiqai.github.io/helia-developer-hub/' },
        },
        discoverability: { markdown: true, llms: true, jsonLd: true, ogImage: true },
        footer: {
          logo: 'ambiq',
          tagline: 'Part of the Ambiq HELIA AI platform',
          links: [{ label: 'physioKIT source on GitHub', href: 'https://github.com/AmbiqAI/physiokit' }],
        },
      })],
    }),
  ],
});

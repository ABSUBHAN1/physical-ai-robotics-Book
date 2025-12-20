/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'From Digital Intelligence to Embodied Machines',
  favicon: 'img/favicon.ico',

  url: 'https://physical-ai-textbook.com',
  baseUrl: '/',

  organizationName: 'physical-ai-research',
  projectName: 'physical-ai-textbook',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/physical-ai-research/textbook/tree/main/',
          routeBasePath: '/',
        },
        blog: {
          showReadingTime: true,
          editUrl: 'https://github.com/physical-ai-research/textbook/tree/main/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'Physical AI Textbook',
        logo: {
          alt: 'Physical AI Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Modules',
          },
          {to: '/blog', label: 'Research Blog', position: 'left'},
          {
            href: 'https://github.com/physical-ai-research/textbook',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Curriculum',
            items: [
              {
                label: 'Module 1: Architecture of Embodiment',
                to: '/module-1/architecture-of-embodiment',
              },
              {
                label: 'Module 2: Perception for Embodied Agents',
                to: '/module-2/perception-for-embodied-agents',
              },
            ],
          },
          {
            title: 'Research',
            items: [
              {
                label: 'Research Blog',
                to: '/blog',
              },
              {
                label: 'Citations & References',
                to: '/references',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/physical-ai-research/textbook',
              },
              {
                label: 'Discussions',
                href: 'https://github.com/physical-ai-research/textbook/discussions',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI Research Collective. Built with Docusaurus.`,
      },
      prism: {
        theme: require('prism-react-renderer/themes/github'),
        darkTheme: require('prism-react-renderer/themes/dracula'),
        additionalLanguages: ['docker', 'bash', 'yaml'],
      },
    }),
};

module.exports = config;
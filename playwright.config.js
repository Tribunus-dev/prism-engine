const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: '.',
  testMatch: ['**/prism-matrix.spec.js'],
  use: {
    headless: true,
    locale: 'en-US',
  },
  projects: [
    {
      name: 'chrome',
      use: {
        channel: 'chrome',
        launchOptions: {
          args: ['--disable-features=TranslateUI'],
        },
      },
    },
  ],
  webServer: {
    command: 'python3 -m http.server 4173 -d .',
    port: 4173,
    reuseExistingServer: true,
    timeout: 60 * 1000,
  },
});

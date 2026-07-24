import { test, expect } from '@playwright/test';

const pages = [
  'index.html',
  'architecture.html',
  'computeimage.html',
  'heterogeneous.html',
  'evidence.html',
  'capabilities.html',
  'run.html',
  'roadmap.html',
];

test.describe('Prism Observatory core surfaces', () => {
  for (const pageName of pages) {
    test(`${pageName} has a complete semantic shell`, async ({ page }) => {
      const errors = [];
      page.on('pageerror', (error) => errors.push(String(error?.message || error)));
      page.on('console', (message) => {
        if (message.type() === 'error') errors.push(message.text());
      });
      await page.goto(`http://127.0.0.1:4173/docs/${pageName}?prismGpu=off`, { waitUntil: 'domcontentloaded' });
      await expect(page.locator('body')).toBeVisible();
      await expect(page.locator('header[data-observatory-shell]')).toHaveCount(1);
      await expect(page.locator('main')).toHaveCount(1);
      await expect(page.locator('main h1')).toHaveCount(1);
      await expect(page.locator('nav[data-observatory-navigation]')).toHaveCount(1);
      await expect(page.locator('footer.site-footer')).toHaveCount(1);
      await page.waitForTimeout(500);
      expect(errors, errors.join('\n')).toEqual([]);
    });
  }

  test('capability registry renders and filters', async ({ page }) => {
    await page.goto('http://127.0.0.1:4173/docs/capabilities.html?prismGpu=off');
    await expect(page.locator('.capability-card')).toHaveCount(12);
    await page.getByRole('button', { name: 'Runtime' }).click();
    await expect(page.locator('.capability-card')).toHaveCount(6);
    await expect(page.locator('.capability-card[data-domain="runtime"]')).toHaveCount(6);
  });

  test('runtime-off preserves core meaning', async ({ page }) => {
    await page.goto('http://127.0.0.1:4173/docs/index.html?prismRuntime=off');
    await expect(page.getByRole('heading', { level: 1 })).toContainText('Computation was never meant');
    await expect(page.getByRole('link', { name: /Run the current path/ })).toBeVisible();
    await expect(page.locator('main')).toBeVisible();
  });
});

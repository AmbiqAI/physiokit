import { expect, test } from '@playwright/test';

test('home, examples, API and plots survive the migration', async ({ page }) => {
  await page.goto('/physiokit/');
  await expect(page.getByRole('heading', { name: /Understand the signals/ })).toBeVisible();
  await page.getByRole('link', { name: 'Get started' }).click();
  await expect(page.getByRole('heading', { name: 'Getting started' })).toBeVisible();
  await page.getByRole('link', { name: 'Use your own data' }).last().click();
  await expect(page.getByRole('heading', { name: 'Clean ECG and locate peaks' })).toBeVisible();
  await page.goto('/physiokit/');
  await page.getByRole('link', { name: 'Explore examples' }).click();
  await expect(page.getByRole('heading', { name: 'Signal guides' })).toBeVisible();
  await page.goto('/physiokit/reference/ecg/');
  await expect(page.getByRole('heading', { name: 'Electrocardiography (ECG)' })).toBeVisible();
  const plot = page.locator('iframe[src="/physiokit/assets/pk-synthetic-ecg-raw.html"]');
  await expect(plot).toBeVisible();
  await page.goto('/physiokit/api/physiokit/ecg/');
  await expect(page.getByRole('heading', { name: /ecg/i }).first()).toBeVisible();
});

test('mobile navigation and documentation assets work', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/physiokit/tutorial/quickstart/');
  await expect(page.getByRole('heading', { name: 'Install and quickstart' })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'uv project' })).toBeVisible();
  await page.locator('summary[aria-label^="Current section"]').click();
  await page.getByRole('navigation', { name: 'Choose section' })
    .getByRole('link', { name: 'Guides' }).click();
  await expect(page.getByRole('heading', { name: 'Signal guides' })).toBeVisible();
  const plot = await page.request.get('/physiokit/assets/pk-synthetic-ecg-clean.html');
  expect(plot.ok()).toBeTruthy();
  await page.goto('/physiokit/tutorial/quickstart/');
  await expect(page.getByRole('link', { name: 'plot source notebook' }))
    .toHaveAttribute('href', 'https://github.com/AmbiqAI/physiokit/blob/main/notebooks/docs.ipynb');
});

test('official Ambiq footer art follows the chosen theme', async ({ page }) => {
  await page.goto('/physiokit/');
  const logo = page.locator('.ambiq-logo');
  for (const [theme, color, asset] of [
    ['Light', 'rgb(0, 71, 186)', 'ambiq-logo.'],
    ['Dark', 'rgb(255, 255, 255)', 'ambiq-logo-white.'],
  ]) {
    await page.locator('button[aria-label^="Color theme"]:visible').click();
    await page.locator('button:visible').filter({ hasText: new RegExp(`^${theme}`) }).first().click();
    await expect.poll(() => logo.evaluate((element) => getComputedStyle(element).backgroundColor)).toBe(color);
    await expect.poll(() => logo.evaluate((element) => getComputedStyle(element).maskImage)).toContain(asset);
  }
});


test('unknown routes return the built not-found page', async ({ page }) => {
  const response = await page.goto('/physiokit/this-route-does-not-exist/');
  expect(response?.status()).toBe(404);
  await expect(page.getByRole('heading', { name: '404', exact: true })).toBeVisible();
  await expect(page.locator('.helia-site-header__title')).toHaveAttribute('href', '/physiokit/');
});

test('API catalog search opens the selected function anchor', async ({ page }) => {
  await page.goto('/physiokit/api/');
  await page.getByRole('searchbox', { name: 'Search Python API' }).fill('simulate_daubechies');
  const result = page.locator('.helia-reference-browser').getByRole('link', { name: 'simulate_daubechies', exact: true });
  await expect(result).toHaveCount(1);
  await result.click();
  await expect(page).toHaveURL(/\/api\/physiokit\/ecg\/synthesize\/#physiokit\.ecg\.synthesize\.simulate_daubechies$/);
  await expect(page.locator('[id="physiokit.ecg.synthesize.simulate_daubechies"]')).toBeVisible();
});

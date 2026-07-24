import { test, expect } from '@playwright/test';

const baseFlags = {
  prismDiagnostics: 'on',
  prismObserver: 'on',
  prismGpu: 'on',
  prismShell: 'on',
  prismReceipts: 'on',
  prismNavigation: 'on',
  prismScroll: 'on',
  prismAccessibility: 'on',
  prismContinuity: 'on',
  prismRuntime: 'on',
};

const buildQuery = (overrides = {}) => {
  const merged = { ...baseFlags, ...overrides, prismRuntimeHardAbort: overrides.prismRuntimeHardAbort ?? 'off', prismDebug: overrides.prismDebug ?? 'on' };
  if (overrides.prismRuntimeHardAbort === undefined) delete merged.prismRuntimeHardAbort;
  if (overrides.prismDebug === undefined) delete merged.prismDebug;
  return Object.entries(merged)
    .map(([name, value]) => `${name}=${value}`)
    .join('&');
};

const matrixScenarios = [
  {
    name: 'baseline-core',
    path: 'index.html',
    query: `${buildQuery({ prismRuntimeHardAbort: 'off' })}`,
  },
  {
    name: 'no-gpu-effects',
    path: 'architecture.html',
    query: `${buildQuery({ prismGpu: 'off', prismReceipts: 'off', prismRuntimeHardAbort: 'off' })}`,
  },
  {
    name: 'shell-only',
    path: 'demo.html',
    query: `${buildQuery({ prismGpu: 'off', prismReceipts: 'off', prismScroll: 'off', prismRuntimeHardAbort: 'off' })}`,
  },
  {
    name: 'minimal',
    path: 'field-guide.html',
    query: `${buildQuery({ prismObserver: 'off', prismShell: 'off', prismReceipts: 'off', prismNavigation: 'off', prismScroll: 'off', prismGpu: 'off' })}`,
  },
  {
    name: 'hard-off-fail-soft',
    path: 'prism-ml.html',
    query: `${buildQuery({ prismRuntimeHardAbort: 'off' })}&prismSystems=observatory-shell,unknown-system`,
  },
  {
    name: 'hard-abort-fail-stop',
    path: 'work-with-prism.html',
    query: `${buildQuery({ prismShell: 'off', prismReceipts: 'off', prismNavigation: 'off', prismScroll: 'off', prismGpu: 'off', prismRuntimeHardAbort: 'on' })}&prismSystems=observatory-shell,unknown-system`,
  },
];

const collectProjectionHealth = async (page, scenarioUrl) => {
  return page.evaluate(() => {
    const body = document.body;
    const debugHandle = window.__prismRuntimeDebug?.debug;
    const startupFailure = debugHandle?.startupFailure || null;
    return {
      loaded: Boolean(body),
      bodyConnected: Boolean(body?.isConnected),
      projectionCount: body?.querySelectorAll('[data-prism-observation-projected]').length || 0,
      hasPortal: Boolean(body?.querySelector('#prism-portal-root')),
      hasEffectsRoot: Boolean(body?.querySelector('#prism-effects-root')),
      hasEffectsShell: Boolean(body?.querySelector('#prism-effects-shell')),
      shellCount: body ? body.querySelectorAll('.observatory-shell').length : 0,
      headerCount: body ? body.querySelectorAll('header').length : 0,
      mainCount: body ? body.querySelectorAll('main').length : 0,
      navCount: body ? body.querySelectorAll('[data-observatory-navigation], .chapter-rail, .chapter-nav, #primary-navigation').length : 0,
      portalRootCount: body ? body.querySelectorAll('#prism-portal-root').length : 0,
      effectsShellCount: body ? body.querySelectorAll('#prism-effects-shell').length : 0,
      effectsRootCount: body ? body.querySelectorAll('#prism-effects-root').length : 0,
      renderFailures: body ? body.querySelectorAll('[data-prism-failed]').length : 0,
      prismObservationProjected: body ? body.dataset.prismObservationProjected : null,
      prismProjectionRoute: body ? body.dataset.prismProjectionRoute : null,
      projectionProjectedAttr: body ? body.getAttribute('data-prism-observation-projected') : null,
      headerConnected: Boolean(document.querySelector('header')),
      mainConnected: Boolean(document.querySelector('main')),
      bodyChildren: body ? body.children.length : 0,
      pageUrl: location.href,
      errorCount: window.__prismReplayErrors?.length || 0,
      startupFailureCode: startupFailure?.error?.code || null,
      startupFailurePhase: startupFailure?.error?.phase || null,
    };
  });
};

const collectLogState = async (page) => {
  const raw = await page.evaluate(() => {
    const state = window.repro ? window.repro : null;
    return { repro: state, errorCount: (window.__prismReplayErrors?.length || 0), errors: window.__prismReplayErrors || [] };
  });
  return raw;
};

test('projection startup matrix', async ({ page }) => {
  test.setTimeout(180000);
  const report = {
    startedAt: new Date().toISOString(),
    scenarios: [],
  };

  for (const scenario of matrixScenarios) {
    const errors = [];
    page.on('pageerror', (error) => errors.push(`pageerror: ${String(error?.message || error)}`));
    page.on('console', (message) => {
      if (message.type() === 'error') {
        errors.push(`console-error: ${message.text()}`);
      }
    });
    await page.route('**/*', (route) => route.continue());
    const pagePath = scenario.path || 'index.html';
    await page.goto(`http://127.0.0.1:4173/docs/${pagePath}?${scenario.query}`, {
      waitUntil: 'domcontentloaded',
      timeout: 45000,
    });

    await page.waitForFunction(
      () => document?.body?.dataset?.prismObservationProjected === 'true',
      { timeout: 12000 },
    ).catch(() => {});

    const params = new URLSearchParams(scenario.query);
    const shellEnabled = params.get('prismShell') !== 'off';
    const hardAbort = params.get('prismRuntimeHardAbort') === 'on';

    const health = await collectProjectionHealth(page, scenario.query);
    const pathMatch = page.url().includes(scenario.path);
    const repro = await collectLogState(page);
    const expectedPortal = shellEnabled;
    const success = Boolean(
      health.bodyConnected &&
      health.prismObservationProjected === 'true' &&
      pathMatch &&
      (!expectedPortal || (health.hasPortal && health.hasEffectsRoot && health.hasEffectsShell)) &&
      health.headerCount === 1 &&
      health.mainCount >= 1 &&
      health.effectsShellCount <= 1 &&
      health.portalRootCount <= 1 &&
      health.effectsRootCount <= 1 &&
      (hardAbort ? health.startupFailureCode === null : true),
    );

    report.scenarios.push({
      scenario: scenario.name,
      query: scenario.query,
      path: scenario.path,
      success,
      health,
      errors,
      repro,
    });

    if (!success) {
      console.log(`[matrix][${scenario.name}] failed`, JSON.stringify({ health, repro, errors }, null, 2));
    }
    page.removeAllListeners('pageerror');
    page.removeAllListeners('console');

    expect(success).toBeTruthy();
    expect(health.bodyChildren).toBeGreaterThan(0);
    expect(health.shellCount).toBeLessThanOrEqual(1);
    expect(health.headerCount).toBe(1);
    expect(health.mainCount).toBeGreaterThan(0);
    expect(health.portalRootCount).toBeLessThanOrEqual(1);
    expect(health.effectsRootCount).toBeLessThanOrEqual(1);
    expect(health.effectsShellCount).toBeLessThanOrEqual(1);
  }

  console.log(JSON.stringify(report, null, 2));
  const failed = report.scenarios.filter((scenario) => !scenario.success && scenario.name !== 'hard-abort-fail-stop');
  if (failed.length) {
    console.log('FAILED_SCENARIOS=' + JSON.stringify(failed.map((scenario) => ({
      scenario: scenario.scenario,
      errors: scenario.errors,
      health: scenario.health,
    })), null, 2));
  }
  expect(failed.length).toBe(0);
});

test('subsystem isolation matrix', async ({ page }) => {
  test.setTimeout(180000);
  const isolationDimensions = [
    { flag: 'prismShell', path: 'index.html' },
    { flag: 'prismReceipts', path: 'architecture.html' },
    { flag: 'prismNavigation', path: 'demo.html' },
    { flag: 'prismScroll', path: 'field-guide.html' },
    { flag: 'prismAccessibility', path: 'general-compute.html' },
    { flag: 'prismObserver', path: 'roadmap.html' },
    { flag: 'prismContinuity', path: 'prism-ml.html' },
    { flag: 'prismGpu', path: 'heterogeneous.html' },
  ];

  for (const dimension of isolationDimensions) {
    const query = `${buildQuery({ [dimension.flag]: 'off' })}`;
    await page.goto(`http://127.0.0.1:4173/docs/${dimension.path}?${query}`, {
      waitUntil: 'domcontentloaded',
      timeout: 45000,
    });

    await page.waitForFunction(
      () => document?.body?.dataset?.prismObservationProjected === 'true',
      { timeout: 12000 },
    ).catch(() => {});

    const health = await collectProjectionHealth(page, query);
    const logs = await collectLogState(page);
    const success = Boolean(
      health.bodyConnected &&
      health.headerConnected &&
      health.mainConnected &&
      health.headerCount === 1 &&
      health.mainCount >= 1 &&
      health.portalRootCount <= 1 &&
      health.effectsShellCount <= 1 &&
      health.prismObservationProjected === 'true' &&
      health.errorCount === 0 &&
      health.renderFailures === 0,
    );

    if (!success) {
      console.log(`[isolation][${dimension.flag}-off] failed`, JSON.stringify({ health, logs }, null, 2));
    }

    expect(success).toBeTruthy();
    expect(health.bodyChildren).toBeGreaterThan(0);
    expect(health.headerCount).toBe(1);
    expect(health.mainCount).toBeGreaterThan(0);
  }
});

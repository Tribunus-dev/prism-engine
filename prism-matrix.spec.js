import { test, expect } from '@playwright/test';

const matrixScenarios = [
  {
    name: 'baseline-core',
    query: 'prismRuntime=on&prismDiagnostics=on&prismObserver=on&prismGpu=on&prismShell=on&prismReceipts=on&prismNavigation=on&prismScroll=on&prismRuntimeHardAbort=off',
  },
  {
    name: 'no-gpu-effects',
    query: 'prismRuntime=on&prismDiagnostics=on&prismObserver=on&prismGpu=off&prismShell=on&prismReceipts=off&prismNavigation=on&prismScroll=on&prismRuntimeHardAbort=off',
  },
  {
    name: 'shell-only',
    query: 'prismRuntime=on&prismDiagnostics=on&prismObserver=on&prismGpu=off&prismShell=on&prismReceipts=off&prismNavigation=on&prismScroll=off&prismRuntimeHardAbort=off',
  },
  {
    name: 'minimal',
    query: 'prismRuntime=on&prismDiagnostics=on&prismObserver=off&prismGpu=off&prismShell=off&prismReceipts=off&prismNavigation=off&prismScroll=off&prismRuntimeHardAbort=off',
  },
  {
    name: 'hard-off-fail-soft',
    query: 'prismRuntime=on&prismDiagnostics=on&prismObserver=on&prismGpu=on&prismShell=on&prismReceipts=on&prismNavigation=on&prismScroll=on&prismRuntimeHardAbort=off&prismSystems=observatory-shell,unknown-system',
  },
  {
    name: 'hard-abort-fail-stop',
    query: 'prismRuntime=on&prismDiagnostics=on&prismObserver=on&prismGpu=off&prismShell=off&prismReceipts=off&prismNavigation=off&prismScroll=off&prismRuntimeHardAbort=on&prismSystems=observatory-shell,unknown-system',
  },
];

const collectProjectionHealth = async (page, scenarioUrl) => {
  return page.evaluate(() => {
    const body = document.body;
    return {
      loaded: Boolean(body),
      bodyConnected: Boolean(body?.isConnected),
      projectionCount: body?.querySelectorAll('[data-prism-observation-projected]').length || 0,
      hasPortal: Boolean(body?.querySelector('#prism-portal-root')),
      hasEffectsRoot: Boolean(body?.querySelector('#prism-effects-root')),
      hasEffectsShell: Boolean(body?.querySelector('#prism-effects-shell')),
      shellCount: body ? body.querySelectorAll('.observatory-shell').length : 0,
      navCount: body ? body.querySelectorAll('[data-observatory-navigation], .chapter-rail, .chapter-nav, #primary-navigation').length : 0,
      renderFailures: body ? body.querySelectorAll('[data-prism-failed]').length : 0,
      prismObservationProjected: body ? body.dataset.prismObservationProjected : null,
      prismProjectionRoute: body ? body.dataset.prismProjectionRoute : null,
      projectionProjectedAttr: body ? body.getAttribute('data-prism-observation-projected') === 'true' : false,
      headerConnected: Boolean(document.querySelector('header')),
      mainConnected: Boolean(document.querySelector('main')),
      bodyChildren: body ? body.children.length : 0,
      pageUrl: location.href,
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
  test.setTimeout(120000);
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
    await page.goto(`http://127.0.0.1:4173/docs/index.html?${scenario.query}`, {
      waitUntil: 'domcontentloaded',
      timeout: 45000,
    });

    await page.waitForTimeout(800);

    const health = await collectProjectionHealth(page, scenario.query);
    const repro = await collectLogState(page);
    const success = Boolean(health.bodyConnected && health.hasPortal && health.prismObservationProjected === 'true');

    report.scenarios.push({
      scenario: scenario.name,
      query: scenario.query,
      success,
      health,
      errors,
      repro,
    });

    if (!success) {
      console.log(`[matrix][${scenario.name}] failed`, JSON.stringify({ health, repro, errors }, null, 2));
    }
    expect(success || scenario.name === 'hard-abort-fail-stop').toBeTruthy();
    page.removeAllListeners('pageerror');
    page.removeAllListeners('console');
  }

  console.log(JSON.stringify(report, null, 2));
  const failed = report.scenarios.filter((scenario) => !scenario.success);
  if (failed.length) {
    console.log('FAILED_SCENARIOS=' + JSON.stringify(failed.map((scenario) => ({
      scenario: scenario.scenario,
      errors: scenario.errors,
      health: scenario.health,
    })), null, 2));
  }
  expect(failed.length).toBe(1);
});

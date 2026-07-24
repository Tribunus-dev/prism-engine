/*
 * Composition root. Dependencies are constructed here before presentation
 * systems are loaded; the browser receives one assembled Prism runtime.
 */
import { createKernel } from './observatory-kernel.js';
import { createRuntime, defaultRegistries } from './runtime/create-runtime.js';
import { createRepositoryService } from './runtime/repository-service.js';
import { createContinuityService } from './runtime/continuity-service.js';
import { createObservatoryClient } from './runtime/client.js';
import { createDomRuntime } from './runtime/dom-runtime.js';
import { createRuntimeConfig } from './runtime/config.js';
import { createObservationGraphSystem } from './core/observation-graph.js';
import { createStateProjectionSystem } from './systems/state-projection.js';
import { createObservatoryShellSystem } from './systems/observatory-shell.js';
import { createCanonicalJourneyRenderer } from './renderers/canonical-journey.js';
import { createCanonicalFocusSystem } from './canonical-focus.js';
import { createReceiptRenderer } from './renderers/receipt.js';
import { createComputeImageRenderer, createComputeImageControls } from './renderers/computeimage.js';
import { createCanonicalObjectSystem } from './systems/canonical-object.js';
import { createNavigationSystem } from './systems/navigation.js';
import { createShellInstrumentsSystem } from './systems/shell-instruments.js';
import { createVisitorStateSystem } from './visitor-state.js';
import { createAccessibilitySystem } from './systems/accessibility.js';
import { createScrollObserverSystem } from './systems/scroll-observer.js';
import { createCanonicalStageSystem } from './systems/canonical-stage.js';
import { createEffectBudgetSystem } from './systems/effect-budget.js';
import { createSiteShellSystem } from './site-shell.js';
import { createGpuPrismSystem } from './gpu-prism.js';
import { createPrismError, ERROR_CODES } from './runtime/errors.js';

const createDiagnosticsHandle = (runtime, domRuntime, kernel) => {
  const handle = {
    timeline: () => [...domRuntime.timeline],
    mutations: () => [...domRuntime.mutations],
    snapshots: () => domRuntime.snapshot() ? [domRuntime.snapshot()] : [],
    health: () => domRuntime.health,
    verify: domRuntime.verify,
    export: () => domRuntime.exportDiagnostics(),
    startup: () => handle.startup || [],
    phase: () => ({
      current: handle.activePhase,
      completed: handle.completedPhases,
    }),
  };
  handle.startup = [];
  handle.completedPhases = [];
  handle.activePhase = 'created';
  handle.record = (phase, detail) => {
    const event = {
      at: Math.round(performance.now()),
      phase,
      ...(detail || {}),
    };
    handle.startup.push(event);
    domRuntime.diagnostics.record('startup', event);
    domRuntime.mark('startup-event', event);
    return event;
  };
  return handle;
};

const publishDiagnosticsHandle = ({ runtime, domRuntime, kernel, diagnostics, config }) => {
  if (!config?.diagnostics) return null;
  const payload = {
    version: 'prism-runtime-v1',
    startedAt: new Date().toISOString(),
    query: location.search,
    config,
    startup: diagnostics.startup(),
    phase: diagnostics.phase(),
    runtime: {
      degraded: runtime?.degraded ?? false,
      debug: runtime?.debug ? {
        hasExport: typeof runtime.debug.export === 'function',
        health: runtime.debug.health?.(),
      } : null,
    },
    dom: {
      health: domRuntime.health,
      snapshot: domRuntime.snapshot(),
      verify: runtime?.debug?.export?.().health,
      claims: domRuntime.exportDiagnostics().ownership,
      timeline: domRuntime.exportDiagnostics().timeline,
      mutations: domRuntime.exportDiagnostics().mutations,
      events: domRuntime.exportDiagnostics().diagnostics?.events,
    },
    error: null,
  };
  return Object.freeze(payload);
};

const attachDebugBus = (value) => {
  if (typeof window === 'undefined' || !window) return;
  try {
    Object.defineProperty(window, '__prismRuntimeDebug', {
      configurable: true,
      writable: true,
      value,
    });
  } catch {
    try {
      window.__prismRuntimeDebug = value;
    } catch {
      // host CSP or security policy blocked debug bus injection.
    }
  }
};

const ensureRuntimeRoot = () => {
  const shellRoot = document.querySelector('#prism-portal-root') || (() => {
    const created = document.createElement('div');
    created.id = 'prism-portal-root';
    created.setAttribute('aria-hidden', 'true');
    const insertionPoint = document.querySelector('.component-header')
      || document.querySelector('header')
      || document.body?.lastElementChild
      || document.body;
    if (insertionPoint && insertionPoint.parentElement === document.body) {
      insertionPoint.parentElement.append(created);
    } else {
      document.body.append(created);
    }
    return created;
  })();
  const effects = document.querySelector('#prism-effects-root') || (() => {
    const created = document.createElement('div');
    created.id = 'prism-effects-root';
    created.setAttribute('aria-hidden', 'true');
    shellRoot.append(created);
    return created;
  })();
  if (!document.querySelector('#prism-effects-shell')) {
    const shell = document.createElement('div');
    shell.id = 'prism-effects-shell';
    shell.setAttribute('aria-hidden', 'true');
    shell.append(effects);
    shellRoot.append(shell);
  }
  return {
    portalRoot: shellRoot,
    effectsRoot: document.querySelector('#prism-effects-root') || effects,
    effectsShell: document.querySelector('#prism-effects-shell'),
  };
};

const normalizeEnabledSystems = (config, allSystems) => {
  if (config?.systems === 'all') {
    return allSystems.map(({ id }) => id);
  }
  const list = (config.systems || '')
    .split(',')
    .map((entry) => String(entry || '').trim())
    .filter(Boolean);
  if (!list.length) return allSystems.map(({ id }) => id);
  const configured = new Set(list);
  return allSystems
    .map(({ id }) => ({ id, enabled: configured.has(id) }))
    .filter(({ enabled }) => enabled);
};

const buildSystems = ({ config, context, domRuntime }) => {
  const computeImage = createComputeImageRenderer(context);

  const createComputeImageSystem = () => ({
    start() {
      if (!computeImage || !computeImage.mountAll || typeof computeImage.mountAll !== 'function') {
        throw createPrismError(ERROR_CODES.RENDERER_MOUNT_FAILED, 'computeimage renderer unavailable');
      }
      if (config.receipts === false) return { stop() {} };
      const mountedElements = [...document.querySelectorAll('[data-computeimage-life], [data-computeimage-renderer]')];
      if (!mountedElements.length) return { stop() {} };
      domRuntime.claim('computeimage-renderer', '[data-computeimage-life], [data-computeimage-renderer]');
      domRuntime.mark('computeimage-ownership-claimed');
      computeImage.mountAll();
      return {
        stop: () => {
          mountedElements.forEach((element) => computeImage.renderer.get(element)?.destroy());
          mountedElements.forEach((element) => {
            element.removeAttribute('data-computeimageRenderer');
            element.removeAttribute('data-computeimageMode');
          });
          domRuntime.mark('computeimage-system-stopped', { elements: mountedElements.length });
        },
      };
    },
  });

  const systems = [
    { id: 'observation-graph', enabled: true, factory: createObservationGraphSystem },
    { id: 'accessibility', enabled: config.accessibility, factory: createAccessibilitySystem },
    { id: 'state-projection', enabled: true, factory: createStateProjectionSystem },
    { id: 'observatory-shell', enabled: config.observer, factory: createObservatoryShellSystem },
    { id: 'canonical-stage', enabled: true, factory: createCanonicalStageSystem },
    { id: 'effect-budget', enabled: config.effects, factory: createEffectBudgetSystem },
    { id: 'scroll-observer', enabled: config.scroll, factory: createScrollObserverSystem },
    { id: 'computeimage-renderer', enabled: true, factory: createComputeImageSystem },
    { id: 'computeimage-controls', enabled: config.receipts, factory: createComputeImageControls },
    { id: 'canonical-journey-renderer', enabled: true, factory: createCanonicalJourneyRenderer },
    { id: 'receipt-renderer', enabled: config.receipts, factory: createReceiptRenderer },
    { id: 'canonical-object', enabled: true, factory: createCanonicalObjectSystem },
    { id: 'canonical-focus', enabled: config.shell, factory: createCanonicalFocusSystem },
    { id: 'visitor-state', enabled: config.shell, factory: createVisitorStateSystem },
    { id: 'navigation', enabled: config.navigation, factory: createNavigationSystem },
    { id: 'shell-instruments', enabled: config.shell, factory: createShellInstrumentsSystem },
    { id: 'site-shell', enabled: config.shell, factory: createSiteShellSystem },
    { id: 'gpu-prism', enabled: config.gpu, factory: createGpuPrismSystem },
  ];

  context.runtime.computeImageRenderer = computeImage.renderer;
  const enabled = normalizeEnabledSystems(config, systems);
  if (config.systems && enabled.length) {
    return systems
      .filter(({ id }) => enabled.includes(id))
      .map(entry => ({ ...entry, enabled: true }));
  }
  return systems.filter((entry) => entry.enabled);
};

const startPhase = async ({
  name,
  config,
  runtime,
  domRuntime,
  diagnostics,
  action,
  optional = false,
}) => {
  domRuntime.mark('startup-phase-begin', { phase: name });
  diagnostics.record(name, { state: 'begin', optional });
  diagnostics.activePhase = name;
  try {
    const result = await action();
    diagnostics.record(name, { state: 'committed' });
    diagnostics.completedPhases.push(name);
    domRuntime.mark('startup-phase-committed', { phase: name, optional });
    return result;
  } catch (error) {
    diagnostics.record(name, {
      state: 'failed',
      code: error?.code || ERROR_CODES.STARTUP_PHASE_FAILED,
      message: error?.message || String(error),
    });
    domRuntime.mark('startup-phase-failed', {
      phase: name,
      optional,
      code: error?.code || ERROR_CODES.STARTUP_PHASE_FAILED,
      message: error?.message || String(error),
    });
    if (!optional && config.runtimeHardAbort) {
      throw createPrismError(
        ERROR_CODES.STARTUP_PHASE_FAILED,
        `Startup phase failed: ${name}`,
        { phase: name, cause: error?.message || String(error) },
      );
    }
    return null;
  }
};

const startSystemTransaction = ({
  system,
  domRuntime,
  runtime,
  activeSystems,
  context,
}) => {
  const phase = context?.phase || 'systems';
  domRuntime.mark('system-start-begin', { id: system.id, phase });
  const fence = domRuntime?.beginRenderFence?.({ owner: system.id, phase });
  let started;
  try {
    started = startSystemInstance(system, domRuntime, runtime, context);
  } catch (error) {
    try {
      fence?.rollback?.();
    } catch {}
    throw error;
  }
  if (fence) {
    fence.commit();
  }
  if (started) {
    started._systemContext = context;
  }
  if (!started) {
    domRuntime.mark('system-start-empty', { id: system.id, phase });
    return null;
  }
  const entry = { id: system.id, instance: started, phase };
  activeSystems.push(entry);
  domRuntime.mark('system-start-transactioned', { id: system.id, phase });
  return entry;
};

const startSystemInstance = (system, domRuntime, runtime, context) => {
  const code = ERROR_CODES.RENDERER_MOUNT_FAILED;
  let created = null;
  try {
    created = system.factory(context);
  } catch (error) {
    throw createPrismError(code, `System factory failed for ${system.id}`, {
      id: system.id,
      cause: error?.message || String(error || ''),
    });
  }

  if (!created) {
    domRuntime.mark('system-start-missing', { id: system.id });
    return null;
  }

  if (typeof created.start !== 'function') {
    domRuntime.mark('system-start-empty', { id: system.id });
    return created;
  }

  let instance = null;
  try {
    instance = created.start(context);
    domRuntime.mark('system-started', { id: system.id, mountFailed: false });
    return instance || { stop() {} };
  } catch (error) {
    const failure = error?.code
      ? error
      : createPrismError(code, `System mount failed for ${system.id}`, {
        id: system.id,
        cause: error?.message || String(error || ''),
      });
    if (created && typeof created.stop === 'function') {
      try {
        created.stop();
      } catch {
        domRuntime.mark('system-stop-failed', { id: system.id });
      }
    }
    throw failure;
  }
};

const isBlockingSystemFailure = (error, config) => {
  if (!error) return false;
  const code = error?.code || ERROR_CODES.SYSTEM_START_FAILED;
  if (config?.runtimeHardAbort) return true;
  const hardCodes = new Set([
    ERROR_CODES.DOM_ROOT_DETACHED,
    ERROR_CODES.OWNERSHIP_CONFLICT,
    ERROR_CODES.OWNERSHIP_ASSERTION_FAILED,
    ERROR_CODES.OWNERSHIP_REGISTRY_GAP,
    ERROR_CODES.ROUTE_PROJECTION_FAILED,
    ERROR_CODES.REPOSITORY_SYNC_FAILED,
    ERROR_CODES.STARTUP_PHASE_FAILED,
  ]);
  return (
    hardCodes.has(code) ||
    code === ERROR_CODES.SYSTEM_START_FAILED
  );
};

const rollbackSystems = (systems, context, activeSystems) => {
  for (const { id, instance, phase } of [...systems].reverse()) {
    try {
      instance?.stop?.();
      context?.domRuntime?.mark?.('system-rollback', { id, phase });
    } catch (error) {
      context?.domRuntime?.mark?.('system-rollback-failed', {
        id,
        code: error?.code || ERROR_CODES.SYSTEM_START_FAILED,
        message: error?.message || String(error),
      });
    }
  }
  if (Array.isArray(activeSystems)) {
    const rollbackIds = new Set(systems.map(system => system.id));
    for (let index = activeSystems.length - 1; index >= 0; index -= 1) {
      if (rollbackIds.has(activeSystems[index]?.id)) {
        activeSystems.splice(index, 1);
      }
    }
  }
  systems.length = 0;
};

const start = async () => {
  const config = createRuntimeConfig(location.search);
  const activeSystems = [];
  const continuity = createContinuityService();
  const kernel = createKernel({ continuity });
  const domRuntime = createDomRuntime({ observerEnabled: config?.observer });
  const repository = createRepositoryService();

  if (!config.continuity) {
    continuity.initial = () => ({});
    continuity.save = value => value;
    continuity.reset = () => ({ visits: 0, lastObservation: null, lastStage: null });
  }

  const runtime = createRuntime({
    kernel,
    domRuntime,
    registries: defaultRegistries,
    adapters: async () => [],
    continuity,
    repository,
  });

  domRuntime.attachRuntime?.(runtime);

  runtime.client = createObservatoryClient({ kernel });
  runtime.continuity = continuity;
  runtime.config = config;
  runtime.degraded = false;

  const diagnostics = createDiagnosticsHandle(runtime, domRuntime, kernel);
  const minimalBootHandle = Object.freeze({
    startup: () => diagnostics.startup(),
    phase: () => diagnostics.phase(),
    activePhase: () => diagnostics.activePhase,
  });
  if (config.debug) {
    attachDebugBus({
      runtime: minimalBootHandle,
      debug: diagnostics,
      activePhase: diagnostics.activePhase,
      config,
    });
  }
  runtime.debug = diagnostics;
  runtime.diagnostics = diagnostics;

  const context = { runtime, kernel, domRuntime, config, repository, continuity };
  const systems = buildSystems({ config, context, domRuntime });
  const runtimeRoots = ensureRuntimeRoot();
  domRuntime.claim('runtime-root', '#prism-portal-root');
  if (runtimeRoots.effectsShell) {
    domRuntime.claim('runtime-effects', '#prism-effects-shell, #prism-effects-root');
  }
  context.runtimeRoots = runtimeRoots;

  const invalidSystems = systems.filter((system) => config.systems && !system.factory);
  if (invalidSystems.length) {
    domRuntime.mark('system-factory-invalid', {
      count: invalidSystems.length,
      ids: invalidSystems.map(({ id }) => id).join(','),
    });
  }

  const auditSystems = () => {
    const ids = new Set(systems.map((entry) => entry.id));
    return (config.systems || '').split(',').map((entry) => String(entry || '').trim()).filter(Boolean).filter(id => !ids.has(id));
  };

  const shutdown = () => {
    activeSystems.reverse().forEach(({ instance }) => {
      try {
        instance?.stop?.();
      } catch {
        // shutdown must not throw during unmount.
      }
    });
  };

  const registerStartupFailure = ({ id, error }) => {
    runtime.degraded = true;
    const code = error?.code || ERROR_CODES.SYSTEM_START_FAILED;
    domRuntime.mark('system-start-failed', {
      id,
      code,
      message: error?.message || String(error),
    });
    return code;
  };

  try {
    await startPhase({
      name: 'bootstrap',
      config,
      runtime,
      domRuntime,
      diagnostics,
      action: async () => {
        domRuntime.mark('runtime-composed', {
          dependencies: ['kernel', 'domRuntime', 'registries', 'repository', 'client'],
        });
      },
      optional: false,
    });

    await startPhase({
      name: 'runtime-start',
      config,
      runtime,
      domRuntime,
      diagnostics,
      action: async () => {
        if (!config.runtime) {
          domRuntime.mark('runtime-disabled', { query: location.search });
          return;
        }
        await runtime.start();
      },
      optional: false,
    });

    await startPhase({
      name: 'systems',
      config,
      runtime,
      domRuntime,
      diagnostics,
      action: async () => {
        const phaseSystems = [];
        for (const entry of systems) {
          if (!entry.enabled) continue;
          try {
            domRuntime.mark("system-transaction-begin", { id: entry.id });
            const started = startSystemTransaction({
              system: entry,
              domRuntime,
              runtime,
              activeSystems,
              context: { ...context, phase: "systems" },
            });
            if (!started) {
              domRuntime.mark("system-transaction-skipped", { id: entry.id, phase: "systems" });
              continue;
            }
            domRuntime.mark("system-transaction-committed", { id: entry.id });
            phaseSystems.push(started);
          } catch (error) {
            const code = registerStartupFailure({ id: entry.id, error });
            rollbackSystems(phaseSystems, { domRuntime }, activeSystems);
            if (isBlockingSystemFailure(error, config)) {
              throw error;
            }
            domRuntime.mark('system-transaction-degraded', {
              id: entry.id,
              code,
              soft: true,
            });
          }
        }
        const invalid = auditSystems();
        if (invalid.length) {
          domRuntime.mark('system-audit-invalid', { ids: invalid.join(',') });
          registerStartupFailure({ id: 'system-audit', error: createPrismError(ERROR_CODES.SYSTEM_START_FAILED, `Unknown systems requested: ${invalid.join(',')}`, { ids: invalid }) });
        }
      },
      optional: false,
    });

    await startPhase({
      name: 'verification',
      config,
      runtime,
      domRuntime,
      diagnostics,
      action: async () => {
        if (!config.runtime) return;
        const external = domRuntime.detectExternalProjectionEnvironment?.();
        const report = domRuntime.verify({ strict: true });
        const requiredHeaders = document.querySelectorAll('header').length;
        const requiredMain = document.querySelectorAll('main').length;
        const portalRoots = document.querySelectorAll('#prism-portal-root').length;
        const effectShellRoots = document.querySelectorAll('#prism-effects-shell').length;
        if (requiredHeaders === 0) {
          throw createPrismError(ERROR_CODES.SYSTEM_START_FAILED, 'Startup verification failed: header root missing');
        }
        if (requiredMain === 0) {
          throw createPrismError(ERROR_CODES.SYSTEM_START_FAILED, 'Startup verification failed: main root missing');
        }
        if (portalRoots === 0 && config.shell) {
          runtime.degraded = true;
          domRuntime?.mark?.('runtime-portal-missing', { portalRoots });
        }
        if (effectShellRoots === 0 && config.shell && config.gpu) {
          runtime.degraded = true;
          domRuntime?.mark?.('runtime-gpu-shell-missing', { effectShellRoots });
        }
        if (requiredHeaders > 1 || requiredMain > 1) {
          runtime.degraded = true;
          domRuntime?.mark?.('runtime-duplicate-core-structure', {
            headers: requiredHeaders,
            main: requiredMain,
          });
        }
        if (config.diagnostics) {
          domRuntime.diagnostics.setProjectionTelemetry?.('attempt', report);
        }

        if (!report.structural.valid) {
          runtime.degraded = true;
          throw createPrismError(
            ERROR_CODES.DOM_ROOT_DETACHED,
            'Structural roots became invalid during startup verification.',
            { structural: report.structural.invalid },
          );
        }
        if (!report.renderers.valid) {
          runtime.degraded = true;
          domRuntime.mark('runtime-renderer-degraded', { renderers: report.renderers.invalid });
          domRuntime.mark('runtime-renderer-soft-failure', {
            soft: true,
            reason: ERROR_CODES.RENDERER_MOUNT_FAILED,
          });
        }
        if (!report.runtime.valid) {
          runtime.degraded = true;
          domRuntime.mark('runtime-runtime-projection-degraded', {
            invalid: report.runtime.invalid,
          });
          if (config.runtimeHardAbort) {
            const failure = createPrismError(
              ERROR_CODES.RUNTIME_STATE_INVALID,
              'Runtime projection failed during startup verification.',
              { runtime: report.runtime.invalid },
            );
            throw failure;
          }
        }
        if (external?.present && !report.runtime.valid) {
          runtime.degraded = true;
          domRuntime.mark('runtime-ownership-external-inconsistent', {
            type: 'external-projection',
            activeNodes: external.activeNodes,
            bodyChildren: external.bodyChildren,
          });
          const failure = createPrismError(
            ERROR_CODES.EXTERNAL_PROJECTION_INCONSISTENT,
            'Projected route state is missing after external shell mutation was detected.',
            {
              activeNodes: external.activeNodes,
              runtimeValid: report.runtime.valid,
              report: report.runtime,
            },
          );
          domRuntime.mark('runtime-external-projection-inconsistent', {
            activeNodes: external.activeNodes,
            bodyChildren: external.bodyChildren,
          });
          if (config.runtimeHardAbort) {
            throw failure;
          }
        }
      },
      optional: false,
    });

    await startPhase({
      name: 'readiness',
      config,
      runtime,
      domRuntime,
      diagnostics,
      action: async () => {
        if (!config.runtime) return;
        kernel.record({
          type: 'prism-runtime-started',
          visible: `Active systems: ${activeSystems.map((system) => system.id).join(', ')}`,
          transformed: 'orchestrated startup',
          hidden: 'legacy integration complexity',
        });

        if (config.diagnostics || config.debug) {
          runtime.debug = diagnostics;
        }

        if (config.debug) {
          attachDebugBus({
            runtime,
            dom: domRuntime,
            debug: diagnostics,
          });
          const payload = publishDiagnosticsHandle({ runtime, domRuntime, kernel, diagnostics, config });
          if (payload) {
            runtime.debug.boot = () => payload;
          }
          if (runtime.degraded) {
            kernel.record?.({
              type: 'runtime-diagnostic',
              visible: `Runtime degraded: ${runtime.degraded}`,
              transformed: 'visibility preserved',
              hidden: 'strict assertions withheld',
            });
          }
        }
      },
      optional: true,
    });

    if (config.runtime && config.diagnostics) {
      const beforeUnload = () => {
        diagnostics.record('ready-beforeunload', {});
        domRuntime.debug();
      };
      window.addEventListener('beforeunload', beforeUnload, { once: true });
    }
  } catch (error) {
    domRuntime.mark('runtime-start-failed', {
      message: error?.message || String(error),
      code: error?.code,
      phase: diagnostics.activePhase,
    });
      try {
      const payload = {
        version: 'prism-runtime-v1',
        startedAt: new Date().toISOString(),
        query: location.search,
        config,
        phase: diagnostics.phase(),
        startup: diagnostics.startup(),
        error: {
          message: error?.message || String(error),
          code: error?.code,
          phase: diagnostics.activePhase,
        },
      };
      if (config.debug && runtime?.debug) {
        runtime.debug.startupFailure = payload;
      }
    } catch {
      // diagnostics channel is best effort only.
    }
    if (config.runtimeHardAbort) {
      shutdown();
      throw error;
    }
    return;
  }

  window.addEventListener('beforeunload', shutdown, { once: true });
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', start, { once: true });
} else {
  start();
}

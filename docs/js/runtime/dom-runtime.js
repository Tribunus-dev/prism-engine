import { createPrismError, ERROR_CODES } from './errors.js';

export const createDomRuntime = (options = {}) => {
  const optionsSnapshot = {
    observerEnabled: options?.observerEnabled !== false,
  };
  let runtime = options?.runtime;
  const started = performance.now();
  let renderFence = false;
  const owners = new WeakMap();
  const ownershipIndex = new Map();
  const ownershipClaims = {};
  const rootClaims = new Map();
  const timeline = [];
  const mutations = [];
  const health = { structural: null, runtime: null, renderers: null, external: null };
  const diagnostics = {
    timeline: [],
    mutations: [],
    snapshots: [],
    events: [],
    errors: [],
  };

  let rootReferences = null;
  const outOfBand = {
    events: [],
    ownership: {
      claimed: 0,
      conflicts: 0,
      assertionFailures: 0,
    },
    structural: {
      strict: 0,
      attempts: 0,
    },
    projections: {
      attempts: 0,
      mismatches: 0,
    },
    lifecycle: {
      claims: 0,
      mutations: 0,
      verification: 0,
      fences: 0,
    },
  };

  const rootRegistry = {
    shell: null,
    observation: null,
    subject: null,
    navigation: null,
    projection: null,
    effects: null,
  };

  const record = (type, detail = {}) => {
    const event = {
      type,
      at: Math.round(performance.now() - started),
      ...detail,
    };
    timeline.push(event);
    return event;
  };

  const diagnostic = (type, detail = {}) => {
    const event = { type, at: Math.round(performance.now() - started), ...detail };
    diagnostics.events.push(event);
    return event;
  };

  const recordChannel = (channel, type, detail = {}) => {
    const event = {
      channel,
      type,
      at: Math.round(performance.now() - started),
      ...detail,
    };
    diagnostics.events.push(event);
    return event;
  };

  const requireConnectedRoot = (selector, root) => {
    const node = typeof root === 'string' ? document.querySelector(root) : root;
    if (!node) {
      throw createPrismError(
        ERROR_CODES.DOM_ROOT_DETACHED,
        `Prism render invariant failed: required root missing (${selector})`,
        { selector },
      );
    }
    if (!node.isConnected) {
      throw createPrismError(
        ERROR_CODES.DOM_ROOT_DETACHED,
        `Prism render invariant failed: required root disconnected (${selector})`,
        { selector },
      );
    }
    return node;
  };

  const nodeIdentity = (node) => {
    if (!node || node.nodeType !== Node.ELEMENT_NODE) return null;
    return `${node.tagName?.toLowerCase?.()}#${node.id || 'n'}${node.className ? `.${node.className.split(/\s+/).filter(Boolean).slice(0, 2).join('.')}` : ''}`;
  };

  const isExternal = node => {
    if (node?.nodeType !== Node.ELEMENT_NODE) return false;
    return (
      node.id === 'browser-mcp-container' ||
      node.closest?.('#browser-mcp-container') ||
      node.matches?.('[data-browser-mcp], [data-mcp-container]')
    );
  };

  const normalizeRootSelector = selector => {
    if (selector === 'document.body') return document.body;
    if (selector === 'header.nav') return document.querySelector('header.nav') || null;
    return document.querySelector(selector);
  };

  const snapshot = () => ({
    shell: document.querySelectorAll('[data-observatory-shell], .observatory-shell, .component-header, header.nav').length,
    observationRoot: document.querySelectorAll('[data-observatory-shell] .observatory-shell, .observatory-shell, [data-observation-root], [data-canonical-stage], [data-computeimage-renderer]').length,
    subject: document.querySelectorAll('[data-computational-subject], [data-computeimage-renderer], [data-computeimage-life]').length,
    inspector: document.querySelectorAll('[data-observatory-inspector], .physics-inspector').length,
    navigation: document.querySelectorAll('[data-observatory-navigation], #primary-navigation, .chapter-rail, .chapter-nav').length,
    projection: document.querySelectorAll('[data-prism-observation-projected="true"]').length,
    effects: document.querySelectorAll('.gpu-prism-field, #prism-effects-root, #prism-effects-shell').length,
  });

  const resolveOwnedRoots = () => ({
    shell: document.querySelector('[data-observatory-shell], .observatory-shell, .component-header, header.nav'),
    observation: document.querySelector('[data-observation-root], [data-canonical-stage], [data-computeimage-renderer]'),
    subject: document.querySelector('[data-computational-subject], [data-computeimage-renderer], [data-computeimage-life]'),
    navigation: document.querySelector('[data-observatory-navigation], #primary-navigation, .chapter-rail, .chapter-nav'),
    projection: document.querySelector('[data-prism-observation-projected="true"]'),
    effects: document.querySelector('#prism-effects-root') || document.querySelector('#prism-effects-shell'),
  });

  const resolveOwnershipOwner = (node) => {
    if (!node) return null;
    if (ownershipIndex.has(node)) return ownershipIndex.get(node);
    const owner = owners.get(node);
    if (owner) {
      ownershipIndex.set(node, owner);
      return owner;
    }
    const previous = node.closest?.('[data-prism-owned]');
    if (!previous) return null;
    return previous.getAttribute('data-prism-owned');
  };

  const markOwned = (node, owner) => {
    if (!node || node.nodeType !== Node.ELEMENT_NODE) return;
    ownershipIndex.set(node, owner);
    owners.set(node, owner);
    if (typeof node.setAttribute === 'function') {
      node.setAttribute('data-prism-owned', owner);
    }
  };

  const claimNode = (owner, node) => {
    if (renderFence) {
      throw createPrismError(
        ERROR_CODES.DOM_RENDERING_LOCKED,
        'DOM claim attempted while render fence is active',
        { owner },
      );
    }
    if (!node || node.nodeType !== Node.ELEMENT_NODE) {
      throw createPrismError(
        ERROR_CODES.DOM_ROOT_DETACHED,
        `Prism render invariant failed: invalid owned node`,
        { owner, nodeType: node?.nodeType || 'missing' },
      );
    }
    const previous = owners.get(node);
    if (previous && previous !== owner) {
      outOfBand.ownership.conflicts += 1;
      outOfBand.errors = outOfBand.errors || [];
      throw createPrismError(
        ERROR_CODES.OWNERSHIP_CONFLICT,
        `DOM ownership conflict: ${node.id || node.tagName.toLowerCase()} claimed by ${previous} and ${owner}`,
        { owner, previousOwner: previous, nodeId: node.id, nodeName: node.nodeName },
      );
    }
    owners.set(node, owner);
    markOwned(node, owner);
    const ownerSet = ownershipClaims[owner] || [];
    ownerSet.push(node);
    ownershipClaims[owner] = ownerSet;
    outOfBand.lifecycle.claims += 1;
    return node;
  };

  const ensureConnected = node => {
    if (!node || !node.isConnected) {
      throw createPrismError(
        ERROR_CODES.DOM_ROOT_DETACHED,
        `Prism render invariant failed: node not connected`,
        { selector: node?.tagName, owner: resolveOwnershipOwner(node) || 'unknown' },
      );
    }
  };

  const verifyStructuralRoots = ({ strict = false } = {}) => {
    outOfBand.structural.attempts += 1;
    if (!rootReferences || Object.values(rootReferences).some(node => !node)) {
      rootReferences = resolveOwnedRoots();
      Object.keys(rootRegistry).forEach(key => {
        rootRegistry[key] = rootReferences[key] || null;
      });
    }
    const invalid = (Object.entries(rootReferences) || [])
      .filter(([, node]) => !node || !node.isConnected)
      .map(([key]) => key);
    const result = {
      valid: invalid.length === 0,
      invalid,
      counts: snapshot(),
      ownersKnown: Object.keys(ownershipClaims).length,
    };
    if (strict) outOfBand.structural.strict += 1;
    health.structural = result;
    if (strict && invalid.length) {
      throw createPrismError(
        ERROR_CODES.DOM_ROOT_DETACHED,
        `Prism render invariant failed: ${invalid.join(', ')}`,
        { roots: invalid },
      );
    }
    return result;
  };

  const verifyRuntimeProjection = () => {
    const projectedMarker = Boolean(
      document.body.dataset.prismObservationProjected === 'true'
      || document.body.dataset.observationProjected === 'true'
      || document.body.getAttribute('data-prism-observation-projected') === 'true'
    );
    const kernel = runtime?.kernel;
    const hasRoute = document.body.dataset.prismObservationProjected === 'true'
      || document.body.dataset.prismProjectionRoute != null
      || document.body.dataset.observationRoute != null;
    const counts = snapshot();
    const countProjection = Math.max(counts.projection, projectedMarker ? 1 : 0);
    const route = document.body.dataset.prismProjectionRoute || document.body.dataset.observationRoute || null;
    const routeMatch = Boolean(kernel?.state?.currentObservation)
      ? true
      : Boolean(route);
    const repositoryReady = Boolean(kernel?.state?.repositoryState || runtime?.kernel?.state?.repositoryState);
    const runtimeProjectionMatch = runtime ? Boolean(kernel?.state?.subjectId) && Boolean(kernel?.state?.currentObservation || route) : true;
    const valid = countProjection >= 1 && hasRoute && runtimeProjectionMatch && projectedMarker;
    const result = {
      valid,
      counts: {
        ...counts,
        projection: countProjection,
      },
      projectedRoute: document.body.dataset.prismProjectionRoute || null,
      projectedObservation: document.body.dataset.observationRoute || null,
      observation: document.body.dataset.sceneObservation || document.body.dataset.observation,
      hasRoute,
      route,
      routeMatch,
      runtimeCurrentObservation: kernel?.state?.currentObservation || null,
      repositoryReady,
      owner: 'runtime',
      marker: projectedMarker,
    };
    health.runtime = result;
    return result;
  };

  const verifyRendererMounts = () => {
    const counts = {
      shell: document.querySelectorAll('.observatory-shell').length,
      navigation: document.querySelectorAll('[data-observatory-navigation], .chapter-rail, .chapter-nav, #primary-navigation').length,
      observationRoot: document.querySelectorAll('[data-observation-root], [data-canonical-stage], [data-computeimage-renderer]').length,
      computeImageRoots: document.querySelectorAll('[data-computeimage-renderer], [data-computeimage-life]').length,
      canonicalJourney: document.querySelectorAll('[data-observatory-journey], [data-observation-root]').length,
    };
    const result = {
      valid: true,
      computeImageRoots: counts.computeImageRoots > 0,
      canonicalJourney: counts.canonicalJourney > 0,
      observatoryShell: counts.shell === 1,
      navigation: counts.navigation === 1,
    };
    result.counts = counts;
    if (counts.shell !== 1) {
      result.shellMultiplicity = counts.shell;
    }
    if (counts.navigation !== 1) {
      result.navigationMultiplicity = counts.navigation;
    }
    if (counts.observationRoot === 0) {
      result.observationRootMissing = true;
    }
    const invalid = Object.entries(result).filter(([key, value]) => key !== 'valid' && !value).map(([key]) => key);
    result.invalid = invalid;
    result.valid = invalid.length === 0;
    health.renderers = result;
    return result;
  };

  const detectExternalProjectionEnvironment = () => {
    const container = document.querySelector('#browser-mcp-container');
    const mcpNodes = container ? container.querySelectorAll('*').length : 0;
    const result = {
      present: Boolean(container),
      activeNodes: mcpNodes,
      bodyChildren: document.body?.children?.length || 0,
      bodyMutationRate: mutations.length,
    };
    health.external = result;
    return result;
  };

  const verifyOwnershipAging = () => {
    const invalid = [];
    for (const [owner, nodes] of Object.entries(ownershipClaims)) {
      nodes.forEach(node => {
        if (!node.isConnected) {
          invalid.push(`${owner}:${nodeIdentity(node) || 'node'}`);
        }
      });
    }
    if (!invalid.length) return { valid: true, missing: [] };
    return {
      valid: false,
      missing: invalid,
    };
  };

  const beginRenderFence = ({ phase = 'system', owner = 'system' } = {}) => {
    if (renderFence) {
      throw createPrismError(
        ERROR_CODES.DOM_RENDERING_LOCKED,
        'Prism render fence already active',
        { phase, owner },
      );
    }
    renderFence = true;
    outOfBand.lifecycle.fences += 1;
    recordChannel('lifecycle', 'render-fence-begin', { phase, owner });
    return {
      commit: () => {
        renderFence = false;
        recordChannel('lifecycle', 'render-fence-commit', { phase, owner });
      },
      rollback: () => {
        renderFence = false;
        recordChannel('lifecycle', 'render-fence-rollback', { phase, owner });
      },
    };
  };

  const verify = ({ strict = false } = {}) => {
    const structural = verifyStructuralRoots({ strict });
    const runtime = verifyRuntimeProjection();
    const renderers = verifyRendererMounts();
    const ownership = verifyOwnershipAging();
    const external = detectExternalProjectionEnvironment();
    const externalProjectionInconsistent = external.present && !runtime.valid;
    const report = {
      valid: structural.valid && renderers.valid && runtime.valid,
      structural,
      runtime,
      renderers,
      ownership,
      external,
      externalProjectionInconsistent,
    };
    record('render-verification', {
      valid: report.valid,
      structural: structural.valid,
      runtime: runtime.valid,
      renderers: renderers.valid,
      externalProjectionInconsistent,
      ownership: ownership.valid,
    });
    outOfBand.lifecycle.verification += 1;
    if (!ownership.valid && strict) {
      throw createPrismError(
        ERROR_CODES.OWNERSHIP_REGISTRY_GAP,
        `DOM ownership claim gap detected: ${ownership.missing.join(', ')}`,
        { ownership: ownership.missing },
      );
    }
    health.external = { ...health.external, externalProjectionInconsistent };
    diagnostics.snapshots.push(snapshot());
    return report;
  };

  const claim = (owner, selector, root = document) => {
    if (renderFence) {
      throw createPrismError(
        ERROR_CODES.DOM_RENDERING_LOCKED,
        'DOM claim attempted while render fence is active',
        { owner, selector },
      );
    }
    const base = typeof root === 'string' ? normalizeRootSelector(root) : root;
    const nodes = [...(root?.querySelectorAll ? base?.querySelectorAll?.(selector) : [])];
    const fallback = [];
    if (!nodes.length) {
      const direct = document.querySelectorAll(selector);
      direct.forEach(node => {
        if (!base || base.contains(node) || base === node) {
          fallback.push(node);
        }
      });
    }
    const targets = nodes.length ? nodes : fallback;
    const ownerSet = ownershipClaims[owner] || [];
    for (const node of targets) {
      const previous = owners.get(node);
      if (previous && previous !== owner) {
        outOfBand.ownership.conflicts += 1;
        outOfBand.errors = outOfBand.errors || [];
        throw createPrismError(
          ERROR_CODES.OWNERSHIP_CONFLICT,
          `DOM ownership conflict: ${selector} claimed by ${previous} and ${owner}`,
          { selector, owner, previousOwner: previous },
        );
      }
      owners.set(node, owner);
      markOwned(node, owner);
      ownerSet.push(node);
    }
    ownershipClaims[owner] = ownerSet;
    outOfBand.ownership.claimed = Object.values(ownershipClaims).reduce((total, nodesForOwner) => total + (nodesForOwner?.length || 0), 0);
    if (targets.length) {
      if (!rootClaims.has(owner)) {
        rootClaims.set(owner, new Set());
      }
      rootClaims.get(owner).add(selector);
      record('dom-claim', { owner, selector, count: targets.length });
      outOfBand.lifecycle.claims += 1;
    } else {
      record('dom-claim-empty', { owner, selector, count: 0 });
    }
    return targets;
  };

  const assertOwnership = (owner, node) => {
    if (renderFence) {
      throw createPrismError(
        ERROR_CODES.DOM_RENDERING_LOCKED,
        'DOM ownership assertion attempted while render fence is active',
        { owner },
      );
    }
    if (!node) throw new Error(`DOM ownership assertion failed: ${owner} received no node`);
    const previous = owners.get(node);
    if (previous && previous !== owner) {
      outOfBand.ownership.assertionFailures += 1;
      throw createPrismError(
        ERROR_CODES.OWNERSHIP_ASSERTION_FAILED,
        `DOM ownership conflict: ${previous} owns ${node.nodeName}, ${owner} attempted mutation`,
        { owner, previousOwner: previous, nodeName: node.nodeName, nodeId: node.id, selector: node.className },
      );
    }
    owners.set(node, owner);
    markOwned(node, owner);
    return true;
  };

const rootsForObservation = new Set(['body', '[data-observatory-shell]', '.observatory-shell', '.component-header', '.component-footer', '#prism-effects-shell', '#prism-effects-root', 'header.nav', 'main']);

  const isOwnedRootNode = node => {
    if (!node || node.nodeType !== Node.ELEMENT_NODE) return null;
    for (const [name, root] of Object.entries(rootReferences || {})) {
      if (!root) continue;
      if (root === node || root.contains(node)) {
        return name;
      }
    }
    return null;
  };

  const detachedRoots = () => {
    const rootFailures = {};
    if (!rootReferences) return rootFailures;
    Object.entries(rootReferences).forEach(([name, node]) => {
      if (!node || !node.isConnected) {
        rootFailures[name] = true;
      }
    });
    return rootFailures;
  };

  const observer = optionsSnapshot.observerEnabled ? new MutationObserver(records => {
    if (!records.length) return;
    const interesting = records.filter((mutation) => {
      if (!mutation || mutation.type !== 'childList') return false;
      if (isExternal(mutation.target)) return false;
      const touched = [mutation.target, ...mutation.addedNodes, ...mutation.removedNodes];
      return touched.some((node) => isOwnedRootNode(node) || isOwnedRootNode(node?.parentElement));
    });
    if (!interesting.length) return;

    const detached = detachedRoots();
      if (Object.keys(detached).length) {
        const detail = {
          at: Math.round(performance.now() - started),
          detached,
        };
        mutations.push(detail);
        diagnostics.mutations.push(detail);
        diagnostic('dom-root-detached', detail);
        record('dom-root-detached', { ...detached });
        outOfBand.lifecycle.mutations += 1;
        return;
      }

      const summary = {
        at: Math.round(performance.now() - started),
        observedMutations: records.length,
        ownedMutations: interesting.length,
      };
      mutations.push(summary);
      diagnostics.mutations.push(summary);
      diagnostic('dom-mutation-filtered', summary);
      outOfBand.lifecycle.mutations += 1;
    }) : null;

  const observeTargets = () => {
    const selectors = [...rootsForObservation];
    const roots = [];
    selectors.forEach((selector) => {
      const node = document.querySelector(selector);
      if (node && !roots.includes(node)) roots.push(node);
    });
    if (document.body && !roots.includes(document.body)) {
      roots.push(document.body);
    }
    return roots;
  };

  const start = () => {
    if (!document.body) return;
    if (!optionsSnapshot.observerEnabled) {
      rootReferences = resolveOwnedRoots();
      Object.keys(rootRegistry).forEach(key => {
        rootRegistry[key] = rootReferences[key] || null;
      });
      record('document-loaded', { after: snapshot() });
      claim('observatory-shell', '[data-observatory-shell], .component-header, header.nav');
      claim('observatory-shell', '[data-observation-root], [data-canonical-stage], [data-computeimage-renderer]');
      verifyStructuralRoots({ strict: false });
      return;
    }
    observeTargets().forEach((target) => {
      observer.observe(target, { childList: true, subtree: true });
    });
    rootReferences = resolveOwnedRoots();
    Object.keys(rootRegistry).forEach(key => {
      rootRegistry[key] = rootReferences[key] || null;
    });
    record('document-loaded', { after: snapshot() });
    claim('observatory-shell', '[data-observatory-shell], .component-header, header.nav');
    claim('observatory-shell', '[data-observation-root], [data-canonical-stage], [data-computeimage-renderer]');
    verifyStructuralRoots({ strict: false });
  };

  const exportDiagnostics = () => ({
    started: new Date(started).toISOString(),
    health: { ...health },
    timeline: [...timeline],
    mutations: [...mutations],
    diagnostics: {
      events: [...diagnostics.events],
      snapshots: [...diagnostics.snapshots],
      errors: [...(diagnostics.errors || [])],
      channels: {
        outOfBand,
      },
    },
    ownership: {
      owners: Object.entries(ownershipClaims).map(([owner, nodes]) => ({
        owner,
        selectors: [...new Set(rootClaims.get(owner) || [])],
        nodes: (nodes || []).map(node => nodeIdentity(node)).filter(Boolean),
      })),
      counts: {
        claimed: outOfBand.ownership.claimed,
        conflicts: outOfBand.ownership.conflicts,
        assertionFailures: outOfBand.ownership.assertionFailures,
      },
    },
  });

  const domRuntime = Object.freeze({
    claimNode,
    claim,
    attachRuntime: (nextRuntime) => {
      runtime = nextRuntime || runtime;
      return runtime;
    },
    assertOwnership,
    mark: (type, detail) => {
      const event = record(type, detail);
      diagnostics.timeline.push(event);
      return event;
    },
    snapshot,
    verify,
    verifyStructuralRoots,
    verifyRuntimeProjection,
    verifyRendererMounts,
    detectExternalProjectionEnvironment,
    timeline,
    mutations,
    health,
    debug: exportDiagnostics,
    start,
    exportDiagnostics,
    diagnostics: {
      outOfBand,
      timeline: () => [...diagnostics.events],
      snapshots: () => [...diagnostics.snapshots],
      events: () => [...diagnostics.events],
      mutations: () => [...diagnostics.mutations],
      record: diagnostic,
      ownership: () => ({
        counts: {
          claimed: outOfBand.ownership.claimed,
          conflicts: outOfBand.ownership.conflicts,
          assertionFailures: outOfBand.ownership.assertionFailures,
        },
      }),
      projections: () => ({ ...outOfBand.projections }),
      lifecycle: () => ({ ...outOfBand.lifecycle }),
      structural: () => ({ ...outOfBand.structural }),
      mark: diagnostic,
      setProjectionTelemetry: (name, value) => {
        outOfBand.projections[name] = value;
        if (name === 'attempt') outOfBand.projections.attempts += 1;
      },
    },
    dispose: () => observer?.disconnect?.(),
    started,
    beginRenderFence,
    requireConnectedRoot,
  });

  start();
  return domRuntime;
};

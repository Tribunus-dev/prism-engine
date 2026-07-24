import { PROJECTIONS, primaryProjection, projectionForLocation } from '../core/observation-projections.js';
import { CLAIM_CLASSES, EXISTENCE_STATES, KNOWLEDGE_STATES, validateClaim } from '../core/ontology.js';
import { TRANSFORMATIONS, validateTransformations } from '../core/transformations.js';
import { createRepositoryService } from './repository-service.js';
import { createRouteProjection } from './route-projection.js';
import { createPrismError, ERROR_CODES } from './errors.js';

/* Composition boundary for the Observatory. Side-effect systems remain
 * application orchestration is centralized here and receives its collaborators
 * explicitly so systems follow one explicit canonical path. */
export const createRuntime = ({
  kernel,
  domRuntime,
  registries,
  adapters,
  continuity,
  repository = createRepositoryService(),
  projection,
}) => {
  const runtimeProjection = projection || createRouteProjection({
    routeForLocation: projectionForLocation,
    primaryRoute: primaryProjection,
  });
  const normalizeKnowledgeState = (value) => Object.values(KNOWLEDGE_STATES).includes(value) ? value : KNOWLEDGE_STATES.OBSERVED;
  const normalizeExistenceState = (value) => Object.values(EXISTENCE_STATES).includes(value) ? value : EXISTENCE_STATES.ACTIVE;
  const normalizeBeliefState = (value) => Object.values(KNOWLEDGE_STATES).includes(value) ? value : KNOWLEDGE_STATES.OBSERVED;
  const resolveCanonicalSubjectId = (snapshot, claims) => (
    snapshot?.state?.subjectId
    || snapshot?.subjectId
    || claims.find(claim => typeof claim?.subjectId === 'string' && claim.subjectId.trim())?.subjectId
    || 'computational-subject:prism-model'
  );
  const buildCanonicalSubject = snapshot => {
    const nextClaims = Array.isArray(snapshot?.claims) ? snapshot.claims : [];
    const evidenceBoundary = (snapshot?.state?.evidenceBoundary
      || snapshot?.evidenceBoundary
      || 'repository evidence pending');

    return {
      id: resolveCanonicalSubjectId(snapshot, nextClaims),
      kind: 'ComputeImage',
      name: snapshot?.name || 'ComputeImage Runtime Subject',
      intent: snapshot?.intent || 'one-subject-canonical-journey',
      representations: snapshot?.representations || [],
      plans: snapshot?.plans || [],
      execution: snapshot?.execution || [],
      receipts: snapshot?.receipts || [],
      lifecycle: snapshot?.lifecycle || [{ phase: 'computeimage-seeded', timestamp: Date.now(), state: 'present' }],
      questions: snapshot?.questions || [],
      sourceRefs: Array.isArray(snapshot?.sourceRefs)
        ? snapshot.sourceRefs
        : nextClaims.flatMap(claim => claim.sourceRefs || []),
      evidenceBoundary,
      provenance: {
        source: snapshot?.provenance?.source || 'repository-state',
        boundary: snapshot?.provenance?.boundary || evidenceBoundary,
      },
      evidenceLevel: snapshot?.state?.evidenceLevel || snapshot?.evidenceLevel,
      knowledge: normalizeKnowledgeState(snapshot?.knowledge),
      belief: normalizeBeliefState(snapshot?.belief),
      existence: normalizeExistenceState(snapshot?.existence),
      objects: snapshot?.objects || {},
      capabilities: snapshot?.capabilities || [],
      claims: nextClaims,
      state: snapshot?.state || {},
    };
  };

  const runtime = {
    kernel,
    domRuntime,
    registries,
    repository,
    stateSubject: null,
    projection: runtimeProjection,
    currentProjection: null,
    currentRoute: null,
    subjectFromRepository: snapshot => buildCanonicalSubject(snapshot || kernel?.state?.repositoryState || {}),
    getCanonicalSubject: () => runtime.subjectFromRepository({
      ...(kernel?.state?.repositoryState || {}),
      claims: kernel?.state?.claims || [],
      capabilities: kernel?.state?.capabilities || [],
    }),
    applyRepositorySnapshot(snapshot = {}, mode = 'event') {
      const mergedState = mode === 'replace'
        ? snapshot?.state || snapshot || {}
        : {
          ...(kernel?.state?.repositoryState || {}),
          ...(snapshot?.state || snapshot || {}),
        };
      return runtime.refreshCanonicalProjection({
        state: mergedState,
        claims: Array.isArray(snapshot?.claims) ? snapshot.claims : kernel?.state?.claims || [],
        capabilities: Array.isArray(snapshot?.capabilities)
          ? snapshot.capabilities
          : kernel?.state?.capabilities || [],
      });
    },
    refreshCanonicalProjection(snapshot = {}) {
      const nextState = snapshot?.state && typeof snapshot.state === 'object' ? snapshot.state : {};
      const nextClaims = Array.isArray(snapshot?.claims) ? snapshot.claims : [];
      const nextCapabilities = Array.isArray(snapshot?.capabilities) ? snapshot.capabilities : [];
      kernel.state.repositoryState = nextState;
      kernel.state.claims = nextClaims;
      kernel.state.capabilities = nextCapabilities;
      runtime.stateSubject = runtime.subjectFromRepository({
        ...nextState,
        claims: nextClaims,
        capabilities: nextCapabilities,
      });
      kernel?.setSubject?.(runtime.stateSubject);
      runtime.client?.setRepository?.({
        state: nextState,
        claims: nextClaims,
        capabilities: nextCapabilities,
      });
      return runtime.stateSubject;
    },
    getCurrentRoute: () => runtime.currentRoute,
    getProjection: () => runtime.currentProjection,
    async start() {
      domRuntime?.mark('boot', { dependencies: ['kernel', 'domRuntime', 'registries', 'adapters', 'repository', 'projection'] });
      domRuntime?.mark('load-services');
      await adapters();
      domRuntime?.mark('services-loaded');
      const savedContinuity = continuity?.initial?.();
      if (savedContinuity) {
        kernel.state.continuity = { ...kernel.state.continuity, ...savedContinuity, visits: Number(savedContinuity.visits || 0) + 1 };
        if (savedContinuity.observerMode) kernel.state.observerMode = savedContinuity.observerMode;
        kernel.emit('continuity', kernel.state.continuity);
      }
      domRuntime?.mark('observation-graph-loaded');
      const route = runtimeProjection.project(window?.location?.pathname);
      runtime.currentProjection = route;
      runtime.currentRoute = route?.route || null;
      if (!route?.observation) {
        throw createPrismError(ERROR_CODES.ROUTE_PROJECTION_FAILED, 'Route projection produced an incomplete route', { route });
      }
      const repositorySnapshot = await repository.load().catch((cause) => {
        throw createPrismError(ERROR_CODES.REPOSITORY_SYNC_FAILED, 'Prism repository sync failed', {
          cause: cause?.message || String(cause || ''),
        });
      });
      if (!kernel?.syncFromRepository) {
        throw createPrismError(ERROR_CODES.RUNTIME_STATE_INVALID, 'Prism kernel synchronization adapter missing');
      }
      kernel.syncFromRepository({
        ...repositorySnapshot,
        claims: repositorySnapshot?.claims,
        capabilities: repositorySnapshot?.capabilities,
        state: repositorySnapshot?.state,
        subjectId: repositorySnapshot?.state?.subjectId,
      });
      runtime.refreshCanonicalProjection({
        state: repositorySnapshot?.state,
        claims: repositorySnapshot?.claims,
        capabilities: repositorySnapshot?.capabilities,
      });
      domRuntime?.mark('build-subject', { subject: runtime.stateSubject?.id });
      domRuntime?.mark('repository-loaded', { synchronized: Boolean(repositorySnapshot) });
      const invalid = validateTransformations(TRANSFORMATIONS).concat(
        Object.values(PROJECTIONS).flatMap(page => page.claims.map(claim => validateClaim(claim)))
      ).filter(Boolean);
      if (invalid.length) {
        throw createPrismError(ERROR_CODES.RUNTIME_STATE_INVALID, `Prism registry validation failed: ${invalid.join('; ')}`);
      }
      domRuntime?.mark('systems-registered');
      domRuntime?.mark('renderer-started');
      domRuntime?.mark('renderers-attached');
      kernel.record({ type: 'runtime-ready', observation: route.observation, visible: 'authoritative registries connected', transformed: 'page projected from shared state', hidden: 'renderer-only optical state' });
      domRuntime?.mark('first-render', { before: domRuntime.snapshot(), route: route.observation, repository: Boolean(repositorySnapshot) });
      domRuntime?.mark('observation-projected', { after: domRuntime.snapshot(), route: route.observation, repository: Boolean(repositorySnapshot) });
      domRuntime?.mark('run', { route: route.observation });
      return runtime;
    }
  };
  repository.subscribe(({ type, snapshot }) => {
    if (type === 'repository-ready') {
      runtime.applyRepositorySnapshot(snapshot, 'replace');
    } else if (type === 'capability-updated') {
      runtime.applyRepositorySnapshot(snapshot);
    } else if (type === 'claim-updated') {
      runtime.applyRepositorySnapshot(snapshot);
    } else if (type === 'evidence-updated') {
      runtime.applyRepositorySnapshot(snapshot);
    }
    kernel.emit(type, snapshot);
  });
  return runtime;
};

export const defaultRegistries = Object.freeze({ PROJECTIONS, CLAIM_CLASSES, KNOWLEDGE_STATES, TRANSFORMATIONS });

import { PROJECTIONS, primaryProjection, projectionForLocation } from '../core/observation-projections.js';
import { CLAIM_CLASSES, KNOWLEDGE_STATES, validateClaim } from '../core/ontology.js';
import { TRANSFORMATIONS, validateTransformations } from '../core/transformations.js';
import { createRepositoryService } from './repository-service.js';
import { createRouteProjection } from './route-projection.js';
import { createPrismError, ERROR_CODES } from './errors.js';

/* Composition boundary for the Observatory. Side-effect systems remain
 * application orchestration is centralized here and receives its collaborators
 * explicitly so systems follow one explicit canonical path. */
export const createRuntime = ({ kernel, domRuntime, registries, adapters, continuity, repository = createRepositoryService(), projection = createRouteProjection({ routeForLocation: projectionForLocation, primaryRoute: primaryProjection }) }) => {
  const runtime = {
    kernel,
    domRuntime,
    registries,
    repository,
    stateSubject: null,
    subjectFromRepository: (snapshot = kernel?.state?.repositoryState) => {
      if (!kernel?.ensureComputeImageSubject) return null;
      const nextClaims = Array.isArray(snapshot?.claims) ? snapshot.claims : [];
      const evidenceBoundary = (snapshot?.state?.evidenceBoundary
        || snapshot?.evidenceBoundary
        || 'repository evidence pending');
      return kernel.ensureComputeImageSubject({
        claims: nextClaims,
        sourceRefs: nextClaims.flatMap(claim => claim.sourceRefs || []),
        evidenceBoundary,
        provenance: {
          source: 'repository-state',
          boundary: evidenceBoundary,
        },
        evidenceLevel: snapshot?.state?.evidenceLevel || snapshot?.evidenceLevel,
      });
    },
    getCanonicalSubject: () => runtime.subjectFromRepository({
      ...(kernel?.state?.repositoryState || {}),
      claims: kernel?.state?.claims || [],
      capabilities: kernel?.state?.capabilities || [],
    }),
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
      domRuntime?.mark('build-subject', { subject: kernel?.subject?.id });
      const route = projection.project();
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
      runtime.stateSubject = runtime.getCanonicalSubject();
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
      kernel.state.repositoryState = snapshot.state;
      kernel.state.claims = snapshot.claims;
      runtime.stateSubject = runtime.subjectFromRepository(snapshot);
      runtime.client?.setRepository?.(snapshot);
    } else if (type === 'capability-updated') {
      kernel.state.repositoryState = { ...(kernel.state.repositoryState || {}), capabilities: snapshot.capabilities };
      runtime.stateSubject = runtime.subjectFromRepository({
        ...snapshot,
        claims: kernel.state.claims || snapshot.claims,
      });
      runtime.client?.setRepository?.({ state: kernel.state.repositoryState, claims: kernel.state.claims, capabilities: snapshot.capabilities });
    } else if (type === 'claim-updated') {
      kernel.state.claims = snapshot.claims;
      runtime.stateSubject = runtime.subjectFromRepository({
        ...snapshot,
        state: kernel.state.repositoryState,
      });
      runtime.client?.setRepository?.({ state: kernel.state.repositoryState, claims: snapshot.claims, capabilities: runtime.client.capabilities });
    } else if (type === 'evidence-updated') {
      kernel.state.repositoryState = snapshot.state;
      runtime.stateSubject = runtime.subjectFromRepository({
        ...snapshot,
        claims: kernel.state.claims || snapshot.claims,
      });
      runtime.client?.setRepository?.({ state: snapshot.state, claims: kernel.state.claims, capabilities: runtime.client.capabilities });
    }
    kernel.emit(type, snapshot);
  });
  return runtime;
};

export const defaultRegistries = Object.freeze({ PROJECTIONS, CLAIM_CLASSES, KNOWLEDGE_STATES, TRANSFORMATIONS });

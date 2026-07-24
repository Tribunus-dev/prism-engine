export const createRepositoryService = ({ fetchImpl = fetch } = {}) => {
  const listeners = new Set();
  const service = {
    state: null,
    claims: [],
    capabilities: [],
    subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    onReady(listener) { return service.subscribe(listener); },
    publish(type, payload) {
      const event = { type, snapshot: payload, timestamp: Date.now() };
      listeners.forEach(listener => listener(event));
      return event;
    },
    updateCapabilities(capabilities) {
      service.capabilities = capabilities || [];
      return service.publish('capability-updated', { capabilities: service.capabilities });
    },
    updateClaims(claims) {
      service.claims = claims || [];
      return service.publish('claim-updated', { claims: service.claims });
    },
    updateEvidence(evidence) {
      service.state = { ...(service.state || {}), evidence };
      return service.publish('evidence-updated', { state: service.state, evidence });
    },
    async load() {
      const state = await fetchImpl('repository-state.json')
        .then(response => response.ok ? response.json() : null);
      const normalizedState = state && typeof state === 'object' ? state : {};
      const normalizedClaims = Array.isArray(normalizedState.claims) ? normalizedState.claims : [];
      const normalizedCapabilities = Array.isArray(normalizedState.capabilities) ? normalizedState.capabilities : [];
      normalizedState.claims = normalizedClaims;
      normalizedState.capabilities = normalizedCapabilities;
      service.state = normalizedState;
      service.claims = normalizedClaims;
      service.capabilities = normalizedCapabilities;
      const snapshot = Object.freeze({
        state: normalizedState,
        claims: service.claims,
        capabilities: service.capabilities,
      });
      service.publish('repository-ready', snapshot);
      return snapshot;
    }
  };
  return service;
};

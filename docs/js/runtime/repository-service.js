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
      const [state, claims, capabilities] = await Promise.all([
        fetchImpl('repository-state.json').then(response => response.ok ? response.json() : null),
        fetchImpl('claims.generated.json').then(response => response.ok ? response.json() : null),
        fetchImpl('capabilities.generated.json').then(response => response.ok ? response.json() : null),
      ]);
      const normalizedClaims = Array.isArray(state?.claims)
        ? state.claims
        : Array.isArray(claims?.claims)
          ? claims.claims
          : [];
      const normalizedCapabilities = Array.isArray(state?.capabilities)
        ? state.capabilities
        : Array.isArray(capabilities?.capabilities)
          ? capabilities.capabilities
          : [];
      service.state = state;
      service.claims = normalizedClaims;
      service.capabilities = normalizedCapabilities;
      const snapshot = Object.freeze({ state, claims: service.claims, capabilities: service.capabilities });
      service.publish('repository-ready', snapshot);
      return snapshot;
    }
  };
  return service;
};

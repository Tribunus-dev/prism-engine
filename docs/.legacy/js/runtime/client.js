export const createObservatoryClient = ({ kernel }) => ({
  inspectSubject: () => kernel.inspectSubject(),
  inspectObject: kind => kernel.inspectObject(kind),
  observe: event => kernel.record(event),
  receipt: event => kernel.receipt(event),
  transform: transformation => kernel.transform(transformation),
  questions: () => kernel.questions(),
  on: (type, listener) => kernel.on(type, listener),
  setCurrentObservation: id => kernel.setCurrentObservation(id),
  setVisitorIntent: intent => kernel.setVisitorIntent(intent),
  repositoryState: null,
  claims: [],
  capabilities: [],
  setRepository(snapshot) {
    this.repositoryState = snapshot?.state || null;
    this.claims = snapshot?.claims || [];
    this.capabilities = snapshot?.capabilities || [];
    document.body.dataset.repositoryState = snapshot?.state && snapshot?.claims && snapshot?.capabilities ? 'synchronized' : 'partial';
    return snapshot;
  },
  inspectRepository() { return this.repositoryState; }
});

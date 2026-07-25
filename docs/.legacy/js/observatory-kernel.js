import { OBSERVER_MODES, OPTICAL_STATES, BELIEF_STATES, OBJECT_KINDS } from './core/vocabulary.js';

export const createKernel = ({ continuity } = {}) => {
  let subjectId = 'computational-subject:prism-model';
  const state = {
    subjectId,
    subject: null,
    currentObservation: null,
    observerMode: 'observer',
    opticalState: 'observation',
    observations: [],
    transformations: [],
    claims: [],
    receipts: [],
    visitorIntent: 'explore',
    disclosureLevel: 'intuition',
    repositoryState: null,
    history: [],
    continuity: {
      visits: 1,
      lastObservation: null,
      lastStage: null,
    },
  };

  const getSubject = () => {
    if (!state.subject) {
      state.subject = {
        id: subjectId,
        name: 'ComputeImage Runtime Subject',
        intent: null,
        representations: [],
        plans: [],
        execution: [],
        receipts: [],
        existence: 'active',
        knowledge: 'observed',
        belief: 'observed',
        questions: [],
        lifecycle: [{ phase: 'birth', timestamp: Date.now(), state: 'possible' }],
        history: [],
        relationships: [],
        sourceRefs: [],
        objects: Object.fromEntries(OBJECT_KINDS.map((kind) => [
          kind,
          {
            kind,
            subject: subjectId,
            knowledge: 'observed',
            belief: 'observed',
            existence: 'active',
            history: [],
            relationships: [],
          },
        ])),
      };
    }
    return state.subject;
  };

  const listeners = new Map();
  const emit = (type, payload) => (listeners.get(type) || []).forEach((listener) => listener(payload));

  const kernel = {
    state,
    get subject() {
      return getSubject();
    },
    modes: OBSERVER_MODES,
    opticalStates: OPTICAL_STATES,

    registerObservation(observation) {
      const entity = {
        id: `observation:${state.observations.length + 1}`,
        subject: subjectId,
        knowledgeState: 'observed',
        evidenceState: 'bounded',
        opticalState: state.opticalState,
        transitionState: 'observation',
        ...observation,
      };
      state.observations.push(entity);
      state.currentObservation = entity.id;
      emit('observation', entity);
      return entity;
    },

    on(type, listener) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(listener);
      return () => listeners.get(type)?.delete(listener);
    },

    emit(type, payload) {
      emit(type, payload);
      return payload;
    },

    setCurrentObservation(id) {
      if (!state.observations.some((item) => item.id === id)) return false;
      state.currentObservation = id;
      emit('observation', state.observations.find((item) => item.id === id));
      return true;
    },

    setVisitorIntent(intent) {
      state.visitorIntent = intent;
      emit('visitor-intent', intent);
      return intent;
    },

    setDisclosureLevel(level) {
      state.disclosureLevel = level;
      emit('disclosure', level);
      return level;
    },

    setMode(mode) {
      if (!OBSERVER_MODES.includes(mode)) return false;
      state.observerMode = mode;
      emit('observer-mode', mode);
      this.remember({ observerMode: mode });
      return true;
    },

    setSubjectId(nextId) {
      if (typeof nextId !== 'string' || !nextId.trim()) return false;
      const normalized = nextId.trim();
      if (subjectId === normalized) return true;
      subjectId = normalized;
      const subject = getSubject();
      subject.id = subjectId;
      subject.objects = Object.fromEntries(OBJECT_KINDS.map((kind) => [
        kind,
        {
          kind,
          subject: subjectId,
          knowledge: 'observed',
          belief: 'observed',
          existence: 'active',
          history: [],
          relationships: [],
        },
      ]));
      this.record({
        type: 'subject-id-updated',
        visible: 'repository subject synchronized',
        transformed: `subject id set to ${subjectId}`,
        hidden: 'transient startup metadata',
      });
      this.remember({ subjectId: subjectId });
      return true;
    },

    setSubject(nextSubject = {}) {
      const normalized = typeof nextSubject?.id === 'string' && nextSubject.id.trim() ? nextSubject.id.trim() : subjectId;
      this.setSubjectId(normalized);
      state.subject = {
        ...getSubject(),
        ...nextSubject,
        id: normalized,
        sourceRefs: Array.isArray(nextSubject?.sourceRefs) ? nextSubject.sourceRefs : getSubject().sourceRefs || [],
        capabilities: Array.isArray(nextSubject?.capabilities) ? nextSubject.capabilities : getSubject().capabilities || [],
      };
      return getSubject();
    },

    syncFromRepository(snapshot = {}) {
      const nextClaims = Array.isArray(snapshot?.claims) ? snapshot.claims : [];
      const nextCapabilities = Array.isArray(snapshot?.capabilities) ? snapshot?.capabilities : [];
      const nextSubjectId = (
        typeof snapshot?.subjectId === 'string' && snapshot.subjectId.trim()
          ? snapshot.subjectId.trim()
          : (nextClaims.find((claim) => typeof claim?.subjectId === 'string' && claim.subjectId.trim())?.subjectId || null)
      ) || subjectId;
      const nextRepositoryState = snapshot.state
        ? snapshot.state
        : { ...snapshot, schema: snapshot.schema || 'repository-state/v1', crates: snapshot.crates || [], docs: snapshot.docs || [] };
      this.setSubjectId(nextSubjectId);
      state.repositoryState = nextRepositoryState;
      state.claims = nextClaims;
      getSubject().capabilities = nextCapabilities;
      return {
        subjectId: nextSubjectId,
        repositoryState: nextRepositoryState,
        claims: nextClaims,
        capabilities: nextCapabilities,
      };
    },

    setOpticalState(next) {
      if (!OPTICAL_STATES.includes(next)) return false;
      state.opticalState = next;
      emit('optical-state', next);
      state.history.push({
        type: 'optical-transition',
        from: state.history.at(-1)?.to || 'observation',
        to: next,
        subject: subjectId,
      });
      return true;
    },

    record(event) {
      const subject = getSubject();
      const entry = {
        type: 'observation-event',
        subject: subjectId,
        timestamp: Date.now(),
        invariant: 'subject identity preserved',
        previousKnowledge: event.previousKnowledge || subject.knowledge,
        observation: event.observation || event.visible || 'observation focused',
        evidence: event.evidence || event.evidenceIncreased || 'unchanged',
        beliefDelta: event.beliefDelta || 'bounded',
        newKnowledge: event.newKnowledge || subject.knowledge,
        remainingUnknowns: event.remainingUnknowns || subject.questions || [],
        transformed: event.transformed || 'observation focused',
        visible: event.visible || 'selected surface',
        hidden: event.hidden || 'unselected surfaces',
        evidenceIncreased: event.evidenceIncreased || 'none',
        ...event,
      };
      state.history.push(entry);
      subject.history.push(entry);
      subject.lifecycle.push({
        phase: event.phase || event.type || 'observation',
        timestamp: entry.timestamp,
        state: event.newKnowledge || subject.existence,
        observation: entry.observation,
      });
      this.remember({
        lastObservation: event.visible || event.type || 'observation',
        lastStage: null,
      });
      return entry;
    },

    receipt(event) {
      const claimClass = event.claimClass || event.claim || (event.knowledgeSource === 'measured-observation' ? 'measured' : 'illustrative');
      if (!['illustrative', 'architectural', 'repository', 'compile-verified', 'measured'].includes(claimClass)) throw new Error(`Unknown claim class: ${claimClass}`);
      const subject = getSubject();
      const sourceRefs = Array.isArray(event.sourceRefs) ? event.sourceRefs.filter((reference) => typeof reference === 'string' && reference.length > 0) : [];
      if (claimClass === 'measured' && (!event.provenance || !event.constraints || event.knowledgeSource !== 'measured-observation' || !sourceRefs.length)) throw new Error('Measured receipts require measured evidence, provenance, constraints, and source references.');
      if (event.sourceRefs && sourceRefs.length !== event.sourceRefs.length) throw new Error('Receipt source references must be non-empty strings.');
      const receipt = {
        ...event,
        id: event.id || `receipt:${state.receipts.length + 1}`,
        subjectId: subjectId,
        subject: subjectId,
        observationId: state.currentObservation,
        transformationId: event.transformationId || null,
        boundary: event.boundary || 'observatory interaction',
        input: event.input || 'current computation',
        decision: event.decision || event.outcome || 'observation recorded',
        target: event.target || 'unbound',
        claimClass,
        evidenceScope: event.evidenceScope || event.knowledgeSource || 'illustrative',
        determinism: event.determinism || 'bounded',
        state: event.state || 'observed',
        sourceRefs,
        constraints: event.constraints || 'static observation',
        unknowns: event.unknowns || subject.questions || [],
        timestamp: Date.now(),
        knowledgeSource: event.knowledgeSource || 'illustrative',
        claim: claimClass,
        confidence: event.confidence || 'bounded',
        provenance: event.provenance || 'observatory interaction',
      };
      state.receipts.push(receipt);
      subject.receipts.push(receipt);
      emit('receipt', receipt);
      return receipt;
    },

    transform(transformation) {
      const errors = [];
      if (!transformation.identityPreserved) errors.push('identity cannot split without provenance');
      if (transformation.evidenceGained?.length && !transformation.postconditions?.length) errors.push('evidence cannot increase without observation');
      if ((transformation.evidenceLost?.length || 0) > (transformation.evidenceGained?.length || 0)) errors.push('a transformation cannot lose more evidence than it explains');
      if (errors.length) throw new Error(errors.join('; '));
      state.transformations.push({ ...transformation, timestamp: Date.now() });
      const receipt = this.receipt({
        observation: `${transformation.from} → ${transformation.to}`,
        outcome: transformation.postconditions.join(', '),
        knowledgeSource: 'architectural',
        claim: 'illustrative',
        confidence: 'bounded',
        provenance: 'transformation algebra',
        constraints: transformation.preconditions.join(', ') || 'none stated',
        transformationId: transformation.id,
      });
      this.record({
        type: 'transformation',
        transformed: `${transformation.from} → ${transformation.to}`,
        visible: transformation.postconditions.join(', '),
        evidenceIncreased: transformation.evidenceGained.join(', ') || 'none',
      });
      return { ...transformation, receipt };
    },

    knowledgeTransition(transition) {
      const entry = {
        subject: subjectId,
        priorKnowledge: transition.priorKnowledge || transition.priorBelief || 'unknown',
        question: transition.question || 'What should be observed?',
        priorBelief: transition.priorBelief || 'unknown',
        observation: transition.observation || 'observation recorded',
        conflict: transition.conflict || 'none stated',
        transformation: transition.transformation || transition.resolution || 'not yet resolved',
        resolution: transition.resolution || 'not yet resolved',
        evidence: transition.evidence || 'bounded',
        newMentalModel: transition.newMentalModel || 'model unchanged',
        confidence: transition.confidence || 'bounded',
        remainingUnknowns: transition.remainingUnknowns || transition.openQuestions || [],
        openQuestions: transition.openQuestions || [],
      };
      state.history.push({ type: 'knowledge-transition', ...entry });
      return entry;
    },

    setBelief(next, evidence = {}) {
      const subject = getSubject();
      if (!BELIEF_STATES.includes(next)) return false;
      const previous = subject.belief;
      if (BELIEF_STATES.indexOf(next) > BELIEF_STATES.indexOf(previous) && !evidence.observation && !evidence.receipt) return false;
      subject.belief = next;
      state.history.push({ type: 'belief-transition', subject: subjectId, from: previous, to: next, evidence });
      return true;
    },

    inspectObject(kind) {
      const subject = getSubject();
      const object = subject.objects[kind];
      if (!object) return null;
      return {
        identity: `${kind.toLowerCase()}:${subjectId}`,
        state: object.existence,
        knowledge: object.knowledge,
        belief: object.belief,
        evidence: object.evidence || 'bounded',
        history: object.history,
        relationships: object.relationships,
        capabilities: object.capabilities || [],
      };
    },

    inspectSubject() {
      const subject = getSubject();
      return {
        identity: subject.id,
        state: subject.existence,
        knowledge: subject.knowledge,
        belief: subject.belief,
        evidence: subject.receipts,
        claims: state.claims,
        repositoryState: state.repositoryState,
        history: subject.history,
        lifecycle: subject.lifecycle,
        relationships: subject.relationships,
        capabilities: subject.objects['Capability Surface']?.capabilities || [],
      };
    },

    questions() {
      return getSubject().questions;
    },

    assertConservation(event = {}) {
      const violations = [];
      if (event.identityChanged && !event.provenance) violations.push('identity cannot split without provenance');
      if (event.evidenceIncreased && !event.observation && !event.receipt) violations.push('evidence cannot increase without observation');
      if (event.claimStrengthened && !event.newEvidence) violations.push('claims cannot strengthen without new evidence');
      return { valid: violations.length === 0, violations };
    },

    remember(update = {}) {
      state.continuity = { ...state.continuity, ...update };
      continuity?.save?.({ ...state.continuity, observerMode: state.observerMode });
      emit('continuity', state.continuity);
      return state.continuity;
    },

    resetContinuity() {
      state.continuity = continuity?.reset?.() || { visits: 0, lastObservation: null, lastStage: null };
      emit('continuity', state.continuity);
      return state.continuity;
    },
  };

  return kernel;
};

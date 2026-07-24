
export const createObservationGraphSystem = () => {
  const start = (context) => {
    const domRuntime = context?.domRuntime;
    const kernel = context?.kernel;
    const owner = 'observation-graph';
    const scenes = {
      origin: {
        act: 'I',
        observation: 'identity',
        phase: 'intent',
        question: 'What is computation?',
        intent: 'An opaque model artifact enters the field.',
        next: 'refraction',
        claim: 'illustrative',
        knowledge: 'illustrative-example',
        existence: 'active',
        misconception: 'A model is just weights.',
        takeaway: 'A model already contains semantic intent.',
        cause: 'model enters the observatory',
        effect: 'semantic intent becomes discussable',
        consequence: 'the visitor can look for structure',
      },
      refraction: {
        act: 'I',
        observation: 'structure',
        phase: 'representation',
        question: 'What was already inside the model?',
        intent: 'Prism separates representation structure without destroying coherence.',
        next: 'representation',
        claim: 'architectural-derivation',
        knowledge: 'architectural-derivation',
        existence: 'active',
        misconception: 'Structure is added after the model is loaded.',
        takeaway: 'The prism reveals representation that was already present.',
        cause: 'intent is preserved',
        effect: 'semantic domains separate',
        consequence: 'the visitor can inspect relationships',
      },
      representation: {
        act: 'II',
        observation: 'structure',
        phase: 'representation',
        question: 'What can remain unified?',
        intent: 'A shared representation carries intent across domains.',
        next: 'compiler',
        claim: 'repository-verified',
        knowledge: 'repository-evidence',
        existence: 'active',
        misconception: 'Every subsystem needs its own semantic model.',
        takeaway: 'One canonical object can cross implementation boundaries.',
        cause: 'domains share a subject',
        effect: 'representations remain attributable',
        consequence: 'compiler and runtime can coordinate',
      },
      compiler: {
        act: 'II',
        observation: 'transformation',
        phase: 'plan',
        question: 'Which plan is legal?',
        intent: 'Candidates are searched against quality, resource, and target gates.',
        next: 'compute-image',
        claim: 'illustrative',
        knowledge: 'illustrative-example',
        existence: 'active',
        misconception: 'Compilation is only lowering.',
        takeaway: 'Compilation is semantic transformation and constrained search.',
        cause: 'representation meets constraints',
        effect: 'candidate plans are explored',
        consequence: 'one plan can be committed',
      },
      'compute-image': {
        act: 'III',
        observation: 'embodiment',
        phase: 'computeimage',
        question: 'What recombines?',
        intent: 'A sealed ComputeImage carries the deployment contract into execution.',
        next: 'scheduler',
        claim: 'repository-verified',
        knowledge: 'repository-evidence',
        existence: 'planned',
        misconception: 'A ComputeImage is another model format.',
        takeaway: 'A ComputeImage is an executable semantic artifact.',
        cause: 'a plan is admitted',
        effect: 'the subject is sealed as a ComputeImage',
        consequence: 'execution can observe one contract',
      },
      scheduler: {
        act: 'IV',
        observation: 'execution',
        phase: 'execution',
        question: 'Where does work belong?',
        intent: 'A serving requirement travels through explicit capability boundaries.',
        next: 'evidence',
        claim: 'compile-verified',
        knowledge: 'compile-verification',
        existence: 'partial',
        misconception: 'Scheduling is only hardware selection.',
        takeaway: 'Scheduling preserves intent across capability boundaries.',
        cause: 'execution observes capabilities',
        effect: 'work is placed and handed off',
        consequence: 'provider behavior becomes observable',
      },
      evidence: {
        act: 'V',
        observation: 'evidence',
        phase: 'receipt',
        question: 'What can be proven?',
        intent: 'Receipts preserve the boundary between a plan and an observed result.',
        next: 'fabric',
        claim: 'repository-verified',
        knowledge: 'repository-evidence',
        existence: 'active',
        misconception: 'Benchmarks prove correctness by themselves.',
        takeaway: 'Receipts expose the scope of claims.',
        cause: 'execution produces an outcome',
        effect: 'a receipt records provenance',
        consequence: 'claims can be inspected',
      },
      fabric: {
        act: 'VI',
        observation: 'scale',
        phase: 'fabric',
        question: 'How far can intent travel?',
        intent: 'The same semantic object expands across machines and providers.',
        next: 'origin',
        claim: 'research-direction',
        knowledge: 'research-direction',
        existence: 'deferred',
        misconception: 'Portability means every target behaves identically.',
        takeaway: 'Portability preserves intent while execution remains target-specific.',
        cause: 'one subject reaches a new provider',
        effect: 'the plan is distributed',
        consequence: 'scale becomes a research frontier',
      },
    };
    const pageScenes = {
      'index.html': 'origin',
      'field-guide.html': 'refraction',
      'architecture.html': 'representation',
      'demo.html': 'compiler',
      'general-compute.html': 'compute-image',
      'heterogeneous.html': 'scheduler',
      'roadmap.html': 'evidence',
      'prism-ml.html': 'fabric',
      'work-with-prism.html': 'fabric',
    };
    const page = context?.runtime?.getProjection?.()?.route
      || context?.runtime?.currentRoute
      || context?.route
      || 'index.html';
    const sceneId = pageScenes[page] || 'origin';
    const scene = scenes[sceneId];
    const canonicalSubject = context?.runtime?.getCanonicalSubject?.() || context?.runtime?.stateSubject;
    const objectId = canonicalSubject?.id || '';

    if (!scene) return { stop() {} };

    Object.assign(document.body.dataset, {
      scene: sceneId,
      sceneAct: scene.act,
      sceneClaim: scene.claim,
      sceneKnowledge: scene.knowledge,
      sceneExistence: scene.existence,
      sceneMisconception: scene.misconception,
      sceneTakeaway: scene.takeaway,
      sceneObservation: scene.observation,
      sceneQuestion: scene.question,
      scenePhase: scene.phase,
      sceneCause: scene.cause,
      sceneEffect: scene.effect,
      sceneConsequence: scene.consequence,
      priorBelief: scene.misconception,
      newMentalModel: scene.takeaway,
      openQuestion: scene.next,
      priorKnowledge: scene.misconception,
      mentalModelObservation: scene.effect,
      mentalModelTransformation: scene.takeaway,
      remainingUnknowns: scene.next,
      observationGraph: sceneId,
      computationalSubject: objectId,
      sceneIntent: scene.intent,
    });

    document.querySelectorAll('[data-scene-question]').forEach(node => {
      if (node === document.body || node === document.documentElement) return;
      node.textContent = scene.question;
    });
    document.querySelectorAll('[data-scene-object]').forEach(node => {
      if (node === document.body || node === document.documentElement) return;
      node.dataset.objectId = objectId;
    });

    const orientation = document.querySelector('.component-orientation');
    if (orientation) {
      const badge = document.createElement('span');
      badge.className = 'claim-badge';
      badge.dataset.claimClass = scene.claim;
      badge.textContent = scene.claim.replace('-', ' ');
      orientation.append(badge);
      domRuntime?.claimNode?.(owner, badge);
      domRuntime?.assertOwnership?.(owner, badge);
    }

    const observation = kernel?.registerObservation({
      instrument: scene.observation,
      phase: scene.phase,
      knowledgeState: scene.knowledge,
      evidenceState: scene.claim,
      opticalState: 'observation',
      transitionState: 'reveal',
      misconception: scene.misconception,
      takeaway: scene.takeaway,
    });

    if (kernel) {
      if (canonicalSubject) {
        canonicalSubject.knowledge = scene.knowledge;
        canonicalSubject.existence = scene.existence;
        canonicalSubject.intent = scene.intent;
        canonicalSubject.questions = [scene.next, scene.question];
      }
      const belief = scene.knowledge === 'repository-evidence' ? 'verified' : scene.knowledge === 'compile-verification' ? 'verified' : scene.knowledge === 'research-direction' ? 'hypothesized' : 'observed';
      kernel.setBelief(belief, { observation: scene.effect });
      kernel.record({
        type: 'observation-entered',
        visible: scene.observation,
        transformed: scene.effect,
        hidden: 'deeper implementation and repository layers',
      });
      kernel.knowledgeTransition({
        priorBelief: scene.misconception,
        priorKnowledge: scene.misconception,
        question: scene.question,
        observation: scene.effect,
        conflict: scene.misconception,
        transformation: scene.takeaway,
        resolution: scene.takeaway,
        newMentalModel: scene.takeaway,
        confidence: scene.knowledge,
        evidence: scene.claim,
        remainingUnknowns: [scene.next],
        openQuestions: [scene.next],
      });
      kernel.setCurrentObservation(observation?.id);
    }

    return { stop() {} };
  };

  return { start };
};

/* URL projections into the observation graph. URLs identify projections; the
 * observation subject and its evidence remain authoritative elsewhere. */
const projection = (route, observation, position, claims = []) => Object.freeze({ route, observation, position, claims });
export const PROJECTIONS = Object.freeze({
  introduction: projection('index.html', 'origin', 1, [{ id: 'claim:canonical-journey', class: 'architectural' }]),
  architecture: projection('architecture.html', 'representation', 2, [{ id: 'claim:ecs-native-compiler', class: 'repository' }]),
  compilation: projection('demo.html', 'compiler', 3, [{ id: 'claim:ecs-native-compiler', class: 'repository' }, { id: 'claim:canonical-journey', class: 'architectural' }]),
  execution: projection('heterogeneous.html', 'execution', 4, [{ id: 'claim:metal', class: 'repository' }, { id: 'claim:rocm', class: 'repository' }, { id: 'claim:xdna', class: 'repository' }]),
  evidence: projection('roadmap.html', 'evidence', 5, [{ id: 'claim:canonical-journey', class: 'architectural' }]),
  computeimages: projection('general-compute.html', 'computeimage', null, [{ id: 'claim:computeimage', class: 'repository' }]),
  research: projection('prism-ml.html', 'research', null, [{ id: 'claim:canonical-journey', class: 'architectural' }]),
  startHere: projection('field-guide.html', 'origin', null, [{ id: 'claim:canonical-journey', class: 'architectural' }]),
  participation: projection('work-with-prism.html', 'participation', null, [{ id: 'claim:canonical-journey', class: 'architectural' }])
});
export const projectionForLocation = pathname => Object.values(PROJECTIONS).find(item => pathname.endsWith(item.route)) || PROJECTIONS.introduction;
export const primaryProjection = projectionValue => projectionValue.position;

const menuButton=document.querySelector('.menu-toggle');
const primaryNavigation=document.querySelector('#primary-navigation');
if(menuButton&&primaryNavigation){
  const closeMenu=(returnFocus=false)=>{primaryNavigation.classList.remove('is-open');menuButton.setAttribute('aria-expanded','false');if(returnFocus)menuButton.focus();};
  menuButton.addEventListener('click',()=>{const open=primaryNavigation.classList.toggle('is-open');menuButton.setAttribute('aria-expanded',String(open));if(open)primaryNavigation.querySelector('a').focus();});
  primaryNavigation.querySelectorAll('a').forEach(link=>link.addEventListener('click',closeMenu));
  document.addEventListener('keydown',event=>{if(event.key==='Escape'&&primaryNavigation.classList.contains('is-open')){event.preventDefault();closeMenu(true)}});
}
const items=document.querySelectorAll('.reveal');
if('IntersectionObserver' in window&&!matchMedia('(prefers-reduced-motion: reduce)').matches){const o=new IntersectionObserver((es)=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('visible');o.unobserve(e.target)}}),{threshold:.12});items.forEach(i=>o.observe(i))}else items.forEach(i=>i.classList.add('visible'));
const journey=document.querySelector('.tensor-journey');const signature=document.querySelector('.signature-figure');if(journey&&signature)signature.parentNode.insertBefore(journey,signature);

const instrumentTabs=document.querySelectorAll('[data-instrument-tab]');
const instrumentPanels=document.querySelectorAll('[data-instrument-panel]');
instrumentTabs.forEach(tab=>tab.addEventListener('click',()=>{instrumentTabs.forEach(t=>{const active=t===tab;t.classList.toggle('is-active',active);t.setAttribute('aria-selected',String(active))});instrumentPanels.forEach(p=>p.classList.toggle('is-hidden',p.dataset.instrumentPanel!==tab.dataset.instrumentTab))}));
const stageCopy={source:['Source admitted','ILLUSTRATIVE TRACE','98.4%','HBM → L2','Metal / queue 0'],compiler:['Lowering + search','ILLUSTRATIVE SEARCH','97.8%','HBM → tile cache','Metal / queue 1'],image:['ComputeImage assembled','ILLUSTRATIVE SEAL','98.1%','HBM → L2','Metal / queue 0'],runtime:['Runtime lane scheduled','ILLUSTRATIVE ROUTE','99.1%','resident → streamed','Metal / queue 0']};
document.querySelectorAll('[data-stage]').forEach(node=>node.addEventListener('click',()=>{document.querySelectorAll('[data-stage]').forEach(n=>n.classList.toggle('is-current',n===node));const v=stageCopy[node.dataset.stage]||stageCopy.source;const title=document.querySelector('#instrument-stage-title');const status=document.querySelector('#instrument-status');if(title)title.textContent=v[0];if(status)status.textContent=v[1];['quality-value','residency-value','route-value'].forEach((id,i)=>{const el=document.getElementById(id);if(el)el.textContent=v[i+2]})}));
document.querySelectorAll('[data-candidate]').forEach(row=>row.addEventListener('click',()=>{document.querySelectorAll('[data-candidate]').forEach(r=>{r.classList.remove('is-survivor');r.querySelector('strong').textContent='candidate'});row.classList.add('is-survivor');row.querySelector('strong').textContent='survivor';const out=document.querySelector('#receipt-outcome');if(out)out.textContent=row.dataset.candidate==='ternary'?'gated':'validated'}));
const engineeringToggle=document.querySelector('#engineering-mode');
if(engineeringToggle)engineeringToggle.addEventListener('change',()=>document.querySelectorAll('.engineering-output,.engineering-metrics').forEach(el=>el.hidden=!engineeringToggle.checked));
(() => {
  const section = document.querySelector('#instruments');
  const grid = section?.querySelector('.instrument-grid');
  if (!section || !grid) return;
  const chapters = document.createElement('div');
  chapters.className = 'compiler-chapters';
  chapters.innerHTML = '<article class="compiler-chapter compiler-search-chapter"><div class="chapter-heading"><span class="tiny-label">CHAPTER 01 / REPRESENTATION SEARCH</span><small>Illustrative scroll state</small></div><h3>Candidate population narrows.</h3><p>As admission progresses, the search keeps only candidates that satisfy the quality and resource gates.</p><div class="candidate-population" role="img" aria-label="Illustrative candidate population shrinking during representation search"></div><div class="chapter-foot"><span>population</span><strong class="population-count">08 → 01</strong></div></article><article class="compiler-chapter compiler-execution-chapter"><div class="chapter-heading"><span class="tiny-label">CHAPTER 02 / EXECUTION PLANNING</span><small>Illustrative scroll state</small></div><h3>Work migrates across lanes.</h3><p>The planner moves work between CPU, GPU, and NPU lanes as the target-aware execution view becomes explicit.</p><div class="execution-lanes" role="img" aria-label="Illustrative work migrating across CPU GPU and NPU lanes"><div><span>CPU</span><i></i></div><div><span>GPU</span><i></i></div><div><span>NPU</span><i></i></div></div><div class="chapter-foot"><span>planning progress</span><strong class="lane-progress">00%</strong></div></article>';
  grid.parentElement.insertBefore(chapters, grid);
  const population = chapters.querySelector('.candidate-population');
  for (let index = 0; index < 8; index += 1) {
    const candidate = document.createElement('i');
    candidate.style.setProperty('--candidate-index', index);
    candidate.setAttribute('aria-hidden', 'true');
    population.append(candidate);
  }
  const lanes = [...chapters.querySelectorAll('.execution-lanes i')];
  const count = chapters.querySelector('.population-count');
  const progressLabel = chapters.querySelector('.lane-progress');
  let frame;
  const render = () => {
    frame = null;
    const rect = section.getBoundingClientRect();
    const progress = Math.max(0, Math.min(1, (window.innerHeight * .72 - rect.top) / Math.max(rect.height * .72, 1)));
    const survivors = Math.max(1, Math.ceil(8 - progress * 7));
    population.style.setProperty('--search-progress', progress);
    population.querySelectorAll('i').forEach((node, index) => node.classList.toggle('is-survivor', index < survivors));
    count.textContent = `${String(survivors).padStart(2, '0')} / 08`;
    lanes.forEach((lane, index) => {
      const phase = Math.max(0, Math.min(1, progress * 1.45 - index * .18));
      lane.style.setProperty('--lane-progress', phase);
      lane.classList.toggle('is-active', phase > .08 && phase < .95);
    });
    progressLabel.textContent = `${String(Math.round(progress * 100)).padStart(2, '0')}%`;
  };
  const requestRender = () => { if (!frame) frame = requestAnimationFrame(render); };
  window.addEventListener('scroll', requestRender, { passive: true });
  window.addEventListener('resize', requestRender);
  render();
})();

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
const stageCopy={source:['Source admitted','LIVE TRACE','98.4%','HBM → L2','Metal / queue 0'],compiler:['Lowering + search','SEARCHING','97.8%','HBM → tile cache','Metal / queue 1'],image:['ComputeImage assembled','SEALED','98.1%','HBM → L2','Metal / queue 0'],runtime:['Runtime lane scheduled','VALIDATED','99.1%','resident → streamed','Metal / queue 0']};
document.querySelectorAll('[data-stage]').forEach(node=>node.addEventListener('click',()=>{document.querySelectorAll('[data-stage]').forEach(n=>n.classList.toggle('is-current',n===node));const v=stageCopy[node.dataset.stage]||stageCopy.source;const title=document.querySelector('#instrument-stage-title');const status=document.querySelector('#instrument-status');if(title)title.textContent=v[0];if(status)status.textContent=v[1];['quality-value','residency-value','route-value'].forEach((id,i)=>{const el=document.getElementById(id);if(el)el.textContent=v[i+2]})}));
document.querySelectorAll('[data-candidate]').forEach(row=>row.addEventListener('click',()=>{document.querySelectorAll('[data-candidate]').forEach(r=>{r.classList.remove('is-survivor');r.querySelector('strong').textContent='candidate'});row.classList.add('is-survivor');row.querySelector('strong').textContent='survivor';const out=document.querySelector('#receipt-outcome');if(out)out.textContent=row.dataset.candidate==='ternary'?'gated':'validated'}));
const engineeringToggle=document.querySelector('#engineering-mode');
if(engineeringToggle)engineeringToggle.addEventListener('change',()=>document.querySelectorAll('.engineering-output,.engineering-metrics').forEach(el=>el.hidden=!engineeringToggle.checked));

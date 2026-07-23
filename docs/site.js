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

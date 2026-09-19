// Copyright 2026 Sreeram Anil — SPDX-License-Identifier: CC-BY-4.0
(function(){
  // theme toggle
  var root=document.documentElement;
  try{var t=localStorage.getItem('estkit-theme');if(t)root.setAttribute('data-theme',t);}catch(e){}
  window.toggleTheme=function(){
    var cur=root.getAttribute('data-theme');
    var next=cur==='dark'?'light':(cur==='light'?'dark':(matchMedia('(prefers-color-scheme:dark)').matches?'light':'dark'));
    root.setAttribute('data-theme',next);
    try{localStorage.setItem('estkit-theme',next);}catch(e){}
  };
})();
// sequential colour for a value in [lo,hi] -> one of 7 ramp steps (blue ramp)
function seqColor(v,lo,hi){
  if(v==null||isNaN(v))return'var(--surface-2)';
  var r=Math.max(0,Math.min(1,(Math.log(v+0.05)-Math.log(lo+0.05))/(Math.log(hi+0.05)-Math.log(lo+0.05))));
  var i=Math.round(r*6);
  return getComputedStyle(document.documentElement).getPropertyValue('--seq'+i)||'#ccc';
}
function inkFor(v,hi){ // dark text on light cells, light text on dark cells
  if(v==null||isNaN(v))return'var(--text-muted)';
  var r=Math.max(0,Math.min(1,(Math.log(v+0.05)-Math.log(0.02+0.05))/(Math.log(hi+0.05)-Math.log(0.02+0.05))));
  return r>0.6?'#f4f4f0':'#0b0b0b';
}

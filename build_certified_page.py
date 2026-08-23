"""Render card_data.json as the certified SWE-bench leaderboard page (HTML)."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
d = json.loads(Path("card_data.json").read_text(encoding="utf-8"))
J = d["J"]
data_js = json.dumps(d, ensure_ascii=False)

html = r"""<title>SWE-bench Verified, Certified</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@87.5,400;87.5,600;100,400;100,500;100,700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{
  --bg:#F4F6F8; --panel:#FFFFFF; --ink:#14181D; --muted:#5C6672; --line:#D9DEE4;
  --acc:#0B6B70; --acc-soft:#CFE6E7; --unk:#CBD2DA; --tier:#E9EDF1; --tier2:#F0F3F6;
  --pub:#9AA3AE;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0F1317; --panel:#151A20; --ink:#E7EAEE; --muted:#98A2B3; --line:#2A323C;
    --acc:#3FBFC4; --acc-soft:#16393B; --unk:#3A434E; --tier:#1A2128; --tier2:#161C23;
    --pub:#6B7683;
  }
}
:root[data-theme="dark"]{
  --bg:#0F1317; --panel:#151A20; --ink:#E7EAEE; --muted:#98A2B3; --line:#2A323C;
  --acc:#3FBFC4; --acc-soft:#16393B; --unk:#3A434E; --tier:#1A2128; --tier2:#161C23;
  --pub:#6B7683;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Archivo,"Helvetica Neue",Arial,sans-serif;font-size:15px;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 80px}
h1{font-family:Archivo;font-stretch:87.5%;font-weight:600;font-size:30px;letter-spacing:-.01em;margin:0 0 6px;text-wrap:balance}
h2{font-family:Archivo;font-stretch:87.5%;font-weight:600;font-size:18px;margin:40px 0 12px;letter-spacing:-.005em}
.sub{color:var(--muted);max-width:62ch;margin:0 0 28px}
.eyebrow{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:600}
.mono{font-family:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.card{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;background:var(--line);border:1px solid var(--line)}
.cell{background:var(--panel);padding:14px 16px}
.cell .k{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:600;margin-bottom:6px}
.cell .v{font-family:"JetBrains Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;font-size:22px;font-weight:600;line-height:1.15}
.cell .n{color:var(--muted);font-size:12.5px;margin-top:4px}
.legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12.5px;margin:8px 0 12px}
.legend span i{display:inline-block;width:14px;height:10px;vertical-align:-1px;margin-right:6px;border-radius:1px}
.chart{overflow-x:auto;border:1px solid var(--line);background:var(--panel)}
.rows{min-width:900px}
.row{display:grid;grid-template-columns:44px 300px 1fr 70px;align-items:center;height:22px;padding:0 10px;border-bottom:1px solid var(--line)}
.row:nth-child(odd){background:var(--tier2)}
.row.tierhead{background:var(--tier);height:26px;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:600;border-top:2px solid var(--line)}
.row .rk{color:var(--pub);font-size:12px}
.row .nm{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:13px}
.row .sc{text-align:right;font-size:12.5px}
.bar{position:relative;height:10px;background:transparent}
.bar .set{position:absolute;top:1px;height:8px;background:var(--acc-soft);border:1px solid var(--acc);border-radius:1px}
.bar .obs{position:absolute;top:0;width:2px;height:10px;background:var(--acc)}
.axis{display:grid;grid-template-columns:44px 300px 1fr 70px;padding:6px 10px;color:var(--muted);font-size:11px;border-bottom:1px solid var(--line)}
.axis .ticks{position:relative;height:14px}
.axis .ticks span{position:absolute;transform:translateX(-50%)}
.tl{overflow-x:auto;border:1px solid var(--line);background:var(--panel);padding:14px 16px}
.tlrow{display:grid;grid-template-columns:100px 1fr 64px 64px 60px 70px;gap:10px;align-items:center;font-size:12.5px;padding:4px 0;border-bottom:1px dashed var(--line);min-width:820px}
.tlrow.h{color:var(--muted);font-size:11px;letter-spacing:.1em;text-transform:uppercase;font-weight:600;border-bottom:1px solid var(--line)}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;letter-spacing:.04em}
.yes{background:var(--acc-soft);color:var(--acc);border:1px solid var(--acc)}
.no{background:transparent;color:var(--muted);border:1px solid var(--line)}
.note{color:var(--muted);font-size:13px;max-width:70ch;margin-top:10px}
footer{margin-top:48px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px}
@media (prefers-reduced-motion:no-preference){.bar .set{transition:width .2s ease}}
</style>
<div class="wrap">
  <div class="eyebrow">Leaderboard Reporting Standard · draft 0.1 · report card</div>
  <h1>SWE-bench Verified, as the evidence supports it</h1>
  <p class="sub">The same matrix the public leaderboard is printed from, shown with what it actually determines: simultaneous rank sets instead of positions, resolved tiers instead of 134 rows, and the bits of ordering the table prints without support.</p>

  <div class="card" id="card"></div>

  <h2>Rank sets, not ranks</h2>
  <div class="legend">
    <span><i style="background:var(--acc-soft);border:1px solid var(--acc)"></i>95 % simultaneous rank set</span>
    <span><i style="background:var(--acc);width:2px"></i>published position</span>
    <span><i style="background:var(--tier);border:1px solid var(--line)"></i>tier = set of mutually indistinguishable systems</span>
  </div>
  <div class="chart"><div class="rows" id="rows"></div></div>
  <p class="note">Every interval holds its system's true rank with 95 % simultaneous coverage — all 134 at once, not one at a time. Systems inside one tier cannot be ordered among themselves by this data. The grey number on the left is what the public leaderboard prints.</p>

  <h2>Every frontier advance, re-tested with the evidence of its day</h2>
  <div class="tl" id="tl"></div>
  <p class="note">“New state of the art” is a claim about two systems. <em>Pairwise</em> is McNemar’s exact test on the items they disagree on, two-sided at 5 %; <em>simultaneous</em> is the rank-set procedure over every system that existed on that date. Steps inside noise still add up: the current leader is separable from every leader up to June 2025 and from none after.</p>

  <footer id="foot"></footer>
</div>
<script>
const D = __DATA__;
const fmt = (x, d=1) => Number(x).toFixed(d);
const card = document.getElementById('card');
const cells = [
  ['R1 shape', D.J + ' × ' + D.n, 'systems × instances, binary'],
  ['R2 rank sets', D.tie1 + ' could be first', 'median set width ' + D.median_width + ' of ' + D.J],
  ['R3 entropy', 'H = ' + fmt(D.H,0) + ' bits', fmt(100*D.H/D.ceiling,1) + ' % of log₂(' + D.J + '!) — one of 2^' + fmt(D.H,0) + ' tables'],
  ['R4 top-10', fmt(D.H10,1) + ' / 21.8 bits', D.H10 > 21.5 ? 'complete antichain' : 'partly resolved'],
  ['R5 established', fmt(100*D.established,1) + ' %', 'of ordered pairs'],
  ['R6 tiers', D.tiers + ' of ' + D.J, 'resolved levels of printed positions; largest antichain ' + D.antichain],
];
for (const [k,v,n] of cells){ const c=document.createElement('div'); c.className='cell'; c.innerHTML='<div class="k">'+k+'</div><div class="v">'+v+'</div><div class="n">'+n+'</div>'; card.appendChild(c); }

const rows = document.getElementById('rows');
const ax = document.createElement('div'); ax.className='axis';
ax.innerHTML = '<div>pub.</div><div>system</div><div class="ticks">'+[1,20,40,60,80,100,120,134].map(t=>'<span style="left:'+(100*(t-1)/(D.J-1))+'%">'+t+'</span>').join('')+'</div><div style="text-align:right">score</div>';
rows.appendChild(ax);
let lastTier = 0;
for (const s of D.systems){
  if (s.tier !== lastTier){
    const th=document.createElement('div'); th.className='row tierhead';
    const members = D.systems.filter(z=>z.tier===s.tier).length;
    th.innerHTML='<div></div><div>tier '+s.tier+' — '+members+' system'+(members>1?'s':'')+', mutually indistinguishable</div><div></div><div></div>';
    rows.appendChild(th); lastTier=s.tier;
  }
  const r=document.createElement('div'); r.className='row';
  const l=100*(s.best-1)/(D.J-1), w=100*(s.worst-s.best)/(D.J-1), o=100*(s.rank-1)/(D.J-1);
  r.innerHTML='<div class="rk mono">'+s.rank+'</div><div class="nm" title="'+s.name+' ('+s.date+')">'+s.name+'</div>'+
    '<div class="bar"><div class="set" style="left:'+l+'%;width:'+Math.max(w,0.4)+'%"></div><div class="obs" style="left:'+o+'%"></div></div>'+
    '<div class="sc mono">'+fmt(s.score,3)+'</div>';
  rows.appendChild(r);
}

const tl=document.getElementById('tl');
const h=document.createElement('div'); h.className='tlrow h'; h.innerHTML='<div>date</div><div>new leader</div><div>gain</div><div>margin</div><div>exact p</div><div>separable</div>'; tl.appendChild(h);
for (const a of D.advances){
  const r=document.createElement('div'); r.className='tlrow';
  r.innerHTML='<div class="mono">'+a.date+'</div><div>'+a.name+'</div><div class="mono">+'+fmt(a.gain,1)+' pt</div><div class="mono">'+(a.margin>0?'+':'')+a.margin+' / '+a.disc+'</div><div class="mono">'+fmt(a.p,3)+'</div>'+
    '<div><span class="pill '+(a.pair?'yes':'no')+'">'+(a.sim?'both':(a.pair?'pairwise':'no'))+'</span></div>';
  tl.appendChild(r);
}
const npair=D.advances.filter(a=>a.pair).length, nsim=D.advances.filter(a=>a.sim).length;
document.getElementById('foot').innerHTML='Of '+D.advances.length+' frontier advances, '+npair+' were pairwise-separable from the leader they displaced and '+nsim+' under the simultaneous criterion. Computed from the public SWE-bench Verified matrix by the reference implementation of the Leaderboard Reporting Standard (draft 0.1), 2026-08-23. Every number here can be recomputed from the matrix.';
</script>
"""
Path("certified_swebench.html").write_text(html.replace("__DATA__", data_js), encoding="utf-8", newline="\n")
print("wrote certified_swebench.html", len(html) // 1024, "KB template +", len(data_js) // 1024, "KB data")

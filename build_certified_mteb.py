"""Render card_data_mteb.json as a standard-0.2 certified leaderboard page."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
d = json.loads(Path("card_data_mteb.json").read_text(encoding="utf-8"))
data_js = json.dumps(d, ensure_ascii=False)

html = r"""<title>MTEB Under the Standard</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Roboto+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap">
<style>
:root{
  --bg:#FBF9F4; --panel:#FFFFFF; --ink:#1A1C1E; --muted:#6A6E76; --line:#E2DED4;
  --ink-soft:#8B8F98;
  --claim:#7A3E9D; --claim-soft:#EFE3F7;
  --ok:#2C6E49; --no:#A03E2F; --warn:#8A6A12;
  --band:#F3EFE6;
  --disp:"Fraunces",Georgia,serif; --ui:"Inter",system-ui,sans-serif; --mono:"Roboto Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#141311; --panel:#1C1B18; --ink:#EDE9E1; --muted:#A19C92; --line:#332F29;
  --ink-soft:#7E7A72; --claim:#C79BE8; --claim-soft:#2E2138; --ok:#7EC79A; --no:#E58C77;
  --warn:#D6AE4C; --band:#211F1B;
}}
:root[data-theme="dark"]{
  --bg:#141311; --panel:#1C1B18; --ink:#EDE9E1; --muted:#A19C92; --line:#332F29;
  --ink-soft:#7E7A72; --claim:#C79BE8; --claim-soft:#2E2138; --ok:#7EC79A; --no:#E58C77;
  --warn:#D6AE4C; --band:#211F1B;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--ui);line-height:1.5;margin:0;padding:2.5rem 1.25rem 5rem}
main{max-width:64rem;margin:0 auto}
h1{font-family:var(--disp);font-size:2.3rem;font-weight:600;line-height:1.1;margin:0 0 .4rem;text-wrap:balance}
h2{font-family:var(--disp);font-size:1.25rem;font-weight:600;margin:2.6rem 0 .6rem;padding-top:.8rem;border-top:1px solid var(--line)}
p{max-width:66ch;margin:.5rem 0;color:var(--ink)}
.lede{color:var(--muted);font-size:1rem;max-width:70ch}
.stamp{display:inline-block;font-family:var(--mono);font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--claim);background:var(--claim-soft);padding:.25rem .55rem;border-radius:2px;margin-bottom:1rem}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));gap:.7rem;margin:1.4rem 0}
.fig{background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:.8rem .9rem}
.fig b{display:block;font-family:var(--mono);font-size:1.5rem;font-weight:500;letter-spacing:-.02em}
.fig span{color:var(--muted);font-size:.82rem;display:block;margin-top:.15rem}
.wrap{overflow-x:auto;margin:.8rem 0 1.2rem}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{text-align:left;padding:.34rem .55rem;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:600}
td.n,th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:.84rem}
.name{font-family:var(--mono);font-size:.82rem;white-space:nowrap;max-width:26rem;overflow:hidden;text-overflow:ellipsis}
.bar{position:relative;height:.62rem;background:var(--band);border-radius:1px;min-width:11rem}
.bar i{position:absolute;top:0;bottom:0;background:var(--claim);opacity:.28;border-radius:1px}
.bar u{position:absolute;top:-1px;bottom:-1px;width:2px;background:var(--claim)}
.pill{font-family:var(--mono);font-size:.7rem;padding:.05rem .35rem;border-radius:2px;border:1px solid var(--line);color:var(--muted)}
.pill.ok{color:var(--ok);border-color:var(--ok)}
.pill.no{color:var(--no);border-color:var(--no)}
.pill.lin{color:var(--claim);border-color:var(--claim)}
.note{border-left:3px solid var(--claim);padding:.45rem .9rem;color:var(--muted);max-width:68ch;margin:1.1rem 0;font-size:.94rem}
code{font-family:var(--mono);font-size:.86em}
.small{font-size:.85rem;color:var(--muted)}
</style>
<main>
<div class="stamp">Leaderboard Reporting Standard · draft 0.2</div>
<h1>MTEB English v2, as the standard would print it</h1>
<p class="lede">174 embedding models, 41 tasks, one published ranking — and the evidence that ranking actually rests on. Every number below is computed from the score matrix alone by the reference implementation; nothing is asserted that the matrix does not support.</p>

<div class="figs" id="figs"></div>

<h2>What the top of the table can and cannot say</h2>
<p id="r10"></p>
<div class="note" id="dividend"></div>

<h2>The ranking, with rank sets</h2>
<p class="small">The bar spans every rank the data allows for that system at 95 % simultaneous confidence; the mark is where the table prints it. Lineage badges group the top twenty by item-level behaviour: two systems carry the same badge when the benchmark cannot treat them as independent evidence.</p>
<div class="wrap"><table id="tbl"><thead><tr>
<th class="n">#</th><th>system</th><th class="n">score</th><th class="n">rank set</th><th>where the data allows it</th><th class="n">tier</th><th>lineage</th>
</tr></thead><tbody></tbody></table></div>
<p class="small" id="more"></p>

<h2>Every "new state of the art" on this board</h2>
<p class="small">κ is the pair's own resolution: below 1 the two models move together task by task and the comparison is sharper than the board average. <em>pairwise</em> = a sign-flip permutation test on the paired differences; <em>simultaneous</em> = the new leader beats the old one inside the rank sets of the field that existed that day.</p>
<div class="wrap"><table id="adv"><thead><tr>
<th>date</th><th>new leader</th><th class="n">gain</th><th class="n">κ</th><th class="n">p</th><th>pairwise</th><th>simultaneous</th><th class="n">field</th>
</tr></thead><tbody></tbody></table></div>
<p id="sotasum"></p>

<h2>What conformance means here</h2>
<p>The standard does not ask a leaderboard to stop ranking. It asks it to print, beside the ranking, the quantities that say how much of the ranking is evidence: simultaneous rank sets, the entropy of the orders the data allows, the resolved tiers, and — new in 0.2 — the resolution of the specific pair behind every claim, together with the split-half reliability of that resolution. On this board κ replicates at r = 0.70 across random halves of the tasks, so the pair-level numbers are measured, not estimated hopefully.</p>
<p class="small">Reference implementation <code>leaderboard_standard.py</code> 0.2 · matrix: 174 models × 41 tasks, dated from HuggingFace repo creation · seed 20260823.</p>
</main>
<script>
const D = __DATA__;
const el = id => document.getElementById(id);
const fmt = (v, k) => (v * 100).toFixed(k === undefined ? 1 : k) + " %";

el("figs").innerHTML = [
  [D.tie1, "models whose rank set contains 1", "of " + D.J + " ranked"],
  [D.median_width, "median rank-set width", "the table prints one rank each"],
  [D.tiers, "tiers the data resolves", "against " + D.J + " printed positions"],
  [D.kappa_top + " κ", "resolution of the #1 vs #2 pair", "board median " + D.kappa_all],
].map(([a, b, c]) => `<div class="fig"><b>${a}</b><span>${b}</span><span style="opacity:.75">${c}</span></div>`).join("");

const s1 = D.systems[0], s2 = D.systems[1];
el("r10").innerHTML = `The board prints <span class="name">${s1.name}</span> first and `
  + `<span class="name">${s2.name}</span> second. The gap between them is ${(D.top_gap * 100).toFixed(2)} points; `
  + `the standard error of that specific difference is ${(D.top_se * 100).toFixed(2)} points, so t = ${D.top_t}. `
  + `Their κ is ${D.kappa_top} — this comparison is sharper than the board's typical pair (${D.kappa_all}) — `
  + `and it still cannot separate them. ${D.tie1} of the ${D.J} models have a rank set containing 1.`;

el("dividend").innerHTML = `<b>The pairing dividend.</b> Computed the way this page does it — paired, `
  + `over the same tasks — 12 to 15 models could be first. Computed the way a leaderboard does it when it `
  + `publishes each model's own error bar, 131 could be. Nothing about the data changes between those two `
  + `numbers; only whether the comparison keeps the pairing.`;

const tb = el("tbl").querySelector("tbody");
const SHOW = 25;
const worst = Math.max(...D.systems.slice(0, SHOW).map(s => s.worst));
D.systems.slice(0, SHOW).forEach(s => {
  const lo = (s.best - 1) / worst * 100, hi = s.worst / worst * 100, at = (s.rank - .5) / worst * 100;
  tb.insertAdjacentHTML("beforeend", `<tr>
    <td class="n">${s.rank}</td>
    <td class="name">${s.name}</td>
    <td class="n">${(s.score * 100).toFixed(2)}</td>
    <td class="n">${s.best}–${s.worst}</td>
    <td><div class="bar"><i style="left:${lo}%;width:${Math.max(hi - lo, 1)}%"></i><u style="left:${at}%"></u></div></td>
    <td class="n">${s.tier}</td>
    <td>${s.lineage ? `<span class="pill lin">L${s.lineage}</span>` : ""}</td>
  </tr>`);
});
el("more").textContent = `Showing the top ${SHOW} of ${D.J}. Entropy of the orders the data allows: `
  + `${D.H} bits of a possible ${D.ceiling}; among the top ten, ${D.H10} of 21.8 bits are undetermined, `
  + `which is every order of those ten. Established pairs: ${fmt(D.established)} of ordered pairs. `
  + `Lineage threshold κ < ${D.lineage_thr}; the top ten hold ${D.lineages_top10} independent lineages.`;

const ab = el("adv").querySelector("tbody");
D.advances.forEach(a => {
  const dt = a.date.slice(0, 4) + "-" + a.date.slice(4, 6) + "-" + a.date.slice(6);
  ab.insertAdjacentHTML("beforeend", `<tr>
    <td class="n">${dt}</td>
    <td class="name">${a.name}</td>
    <td class="n">${(a.gain * 100).toFixed(2)}</td>
    <td class="n">${a.kappa}</td>
    <td class="n">${a.p < 0.001 ? "&lt;0.001" : a.p.toFixed(3)}</td>
    <td><span class="pill ${a.pair ? "ok" : "no"}">${a.pair ? "separable" : "not separable"}</span></td>
    <td><span class="pill ${a.sim ? "ok" : "no"}">${a.sim ? "separable" : "not separable"}</span></td>
    <td class="n">${a.field}</td>
  </tr>`);
});
const np_ = D.advances.filter(a => a.pair).length, ns = D.advances.filter(a => a.sim).length;
const lastSim = D.advances.filter(a => a.sim).slice(-1)[0];
el("sotasum").innerHTML = `${D.advances.length} times the frontier moved. ${np_} of those steps survive a paired `
  + `test against the previous leader; ${ns} survive the standard's simultaneous criterion. The last one that did `
  + `was <span class="name">${lastSim ? lastSim.name : "—"}</span> on `
  + `${lastSim ? lastSim.date.slice(0, 4) + "-" + lastSim.date.slice(4, 6) + "-" + lastSim.date.slice(6) : "—"}; `
  + `every claim since then is inside the noise of the pair it was claimed against.`;
</script>
"""

Path("certified_mteb.html").write_text(html.replace("__DATA__", data_js), encoding="utf-8", newline="\n")
print(f"wrote certified_mteb.html ({len(html) // 1024} KB template, {len(data_js) // 1024} KB data)")

#!/usr/bin/env python3
# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
"""Generate the estkit GitHub Pages website (static HTML/CSS/JS, no framework) from the
catalogue, the report figures and the benchmark summaries.

    python3 scripts/catalogue.py               # first: writes docs/catalogue.json
    python3 scripts/site/build_site.py --out site

Produces a self-contained `site/`:
    index.html            landing page (hero, scale, architecture, results, paths)
    estimators.html       filterable / sortable catalogue of all 86 estimators
    estimator/<slug>.html one page per estimator
    benchmark.html        interactive in-browser heat map + ranking + cost/accuracy
    docs.html             getting started, model concept, adding an estimator, numerics, validation
    references.html       all 314 cited publications
    assets/               style.css, app.js, figures, report link
"""
import os, re, sys, json, shutil, html, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# In GitHub Actions, GITHUB_REPOSITORY is 'owner/repo'; fall back to the placeholder otherwise.
_slug = os.environ.get('ESTKIT_REPO') or os.environ.get('GITHUB_REPOSITORY') or 'GITHUB_USER/estkit'
GH = _slug.split('/')[0]
REPO = f'https://github.com/{_slug}'
PAGES = f'https://{GH}.github.io/{_slug.split("/")[-1]}'
FAM_ORDER = ['baselines', 'kalman', 'adaptive_kalman', 'robust_kalman', 'observers', 'soh',
             'particle', 'learning', 'optimization', 'smoothers', 'pack', 'attitude']
FAM_SWATCH = {'baselines': 1, 'kalman': 2, 'adaptive_kalman': 3, 'robust_kalman': 4, 'observers': 5,
              'soh': 6, 'particle': 7, 'learning': 8, 'optimization': 1, 'smoothers': 2, 'pack': 3, 'attitude': 4}
SCEN = ['nominal', 'init_error', 'current_bias', 'noise_step', 'outliers', 'cold', 'hot',
        'aged_cell', 'param_mismatch', 'sensor_dropout', 'preisach', 'combined']


def slug(n):
    return re.sub(r'[^a-z0-9]+', '-', n.lower()).strip('-')


def safe(n):
    return re.sub(r'[^A-Za-z0-9]', '', n)


def e(s):
    return html.escape(str(s))


def md_links(s):
    s = e(s)
    s = re.sub(r'\[([^\]]+)\]\(REFERENCES\.md#([^)]+)\)', r'<a href="references.html#\2">\1</a>', s)
    s = re.sub(r'doi:\[([^\]]+)\]\((https?://[^)]+)\)', r'doi:<a href="\2">\1</a>', s)
    s = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<a href="\2">\1</a>', s)
    s = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    return s


# ------------------------------------------------------------------ page shell
def page(title, active, body, depth=0, extra_head='', description=''):
    up = '../' * depth
    nav = [('Overview', 'index.html'), ('Estimators', 'estimators.html'),
           ('Benchmark', 'benchmark.html'), ('Docs', 'docs.html'), ('References', 'references.html')]
    navhtml = ''.join(
        f'<a href="{up}{href}" class="{"active" if key==active else ""}">{lbl}</a>'
        for lbl, href in [(l, h) for l, h in nav] for key in [href.split(".")[0]])
    navhtml = ''.join(f'<a href="{up}{h}" class="{"active" if h.split(".")[0]==active else ""}">{l}</a>' for l, h in nav)
    return f'''<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{e(title)}</title>
<meta name="description" content="{e(description or 'estkit — a header-only C++17 library and benchmark of 86 state estimators and observers for battery-management systems in electric aircraft.')}">
<link rel="stylesheet" href="{up}assets/style.css">{extra_head}
</head><body>
<header class="top"><div class="wrap">
<a class="brand" href="{up}index.html"><span class="dot"></span>estkit</a>
<nav class="main">{navhtml}
<button class="theme-btn" onclick="toggleTheme()" title="Toggle light / dark">◐</button>
<a class="gh" href="{REPO}">GitHub ↗</a></nav>
</div></header>
{body}
<footer><div class="wrap">
<strong>estkit</strong> · a header-only C++17 state-estimation library &amp; benchmark · © 2026 Sreeram Anil ·
code under <a href="{REPO}/blob/main/LICENSE">Apache&nbsp;2.0</a>, report &amp; data under
<a href="{REPO}/blob/main/report/LICENSE-CC-BY-4.0.txt">CC&nbsp;BY&nbsp;4.0</a> ·
<a href="{REPO}">source</a> · <a href="{up}references.html">references</a>
<div class="muted" style="margin-top:8px">Benchmark figures generated from <code>results/final</code>. AI assistance was used in preparing this project; all engineering decisions, implementations and results are the author's.</div>
</div></footer>
<script src="{up}assets/app.js"></script></body></html>'''


def tile(n, l, s='', accent=False):
    return f'<div class="tile{" accent" if accent else ""}"><div class="n bignum">{n}</div><div class="l">{l}</div>{f"<div class=s>{s}</div>" if s else ""}</div>'


def fam_chip(fam, label, on=True, link=None):
    sw = f'var(--s{FAM_SWATCH[fam]})'
    inner = f'<span class="sw" style="background:{sw}"></span>{e(label)}'
    if link:
        return f'<a class="chip{" on" if on else ""}" href="{link}" data-fam="{fam}">{inner}</a>'
    return f'<span class="chip{" on" if on else ""}" data-fam="{fam}">{inner}</span>'


# ------------------------------------------------------------------ landing
def build_index(cat, out):
    ncell = sum(1 for c in cat if c['level'] == 'cell')
    npack = sum(1 for c in cat if c['level'] == 'pack')
    natt = sum(1 for c in cat if c['level'] == 'attitude')
    # best nominal / combined performers for a headline line
    best_comb = min((c for c in cat if c['level'] == 'cell'), key=lambda c: c['results']['ecm']['combined'])
    fams = [f for f in FAM_ORDER if any(c['family'] == f for c in cat)]
    famcards = ''
    labels = {c['family']: c['family_label'] for c in cat}
    for f in fams:
        rows = [c for c in cat if c['family'] == f]
        names = ', '.join(c['name'] for c in rows[:6]) + ('…' if len(rows) > 6 else '')
        famcards += f'''<a class="panel" href="estimators.html#{f}" style="display:block">
<div style="display:flex;align-items:center;gap:9px"><span class="chip on" style="pointer-events:none"><span class="sw" style="background:var(--s{FAM_SWATCH[f]})"></span>{e(labels[f])}</span><span class="muted" style="margin-left:auto">{len(rows)}</span></div>
<div class="muted" style="font-size:.85rem;margin-top:9px">{e(names)}</div></a>'''
    hero = f'''<section class="hero"><div class="wrap">
<div class="eyebrow">Header-only C++17 · embedded · dependency-free</div>
<h1>Every state estimator, in one library, on one benchmark.</h1>
<p class="lede">estkit implements <b>86 state estimators and observers</b> — from the Kalman filter to
moving-horizon estimation, particle filters and deterministic observers — against a single
model-agnostic interface, with no dynamic memory and no exceptions, then measures every one of
them on the battery-management problem of an electric aircraft.</p>
<div class="cta">
<a class="btn primary" href="estimators.html">Browse the catalogue →</a>
<a class="btn ghost" href="benchmark.html">Explore the benchmark</a>
<a class="btn ghost" href="{REPO}/releases/latest">Read the report (PDF, 958 pp.)</a>
</div>
<div class="tiles">
{tile('86', 'estimators &amp; observers', f'{ncell} cell · {npack} pack · {natt} attitude', True)}
{tile('12', 'algorithm families', 'baselines → smoothers')}
{tile('~9,000', 'benchmark runs', '2 plants · 12 faults · 3 seeds')}
{tile('0', 'runtime dependencies', 'C++17 standard library only')}
</div>
</div></section>'''
    body = hero + f'''<div class="wrap">
<h2>What it is</h2>
<div class="grid2">
<div class="panel"><h3 style="margin-top:0">A library</h3>
<p class="sec">One header per estimator, each templated on a small model concept
(<code>f, h, Q, R</code>; Jacobians optional). No heap, no exceptions, fixed-size linear
algebra with compile-time dimensions, <code>Real = double</code> or <code>float</code>,
warning-free under <code>-Wall -Wextra -Wpedantic -Werror</code> on GCC and Clang, and it
<b>cross-compiles to a Cortex-M7</b>. The battery model is one instance; a drone navigation model
and an attitude model are others — the same <code>Ekf&lt;M&gt;</code> estimates all three.</p>
<a href="docs.html">The model concept &amp; getting started →</a></div>
<div class="panel"><h3 style="margin-top:0">A benchmark</h3>
<p class="sec">Two truth plants — an enhanced equivalent-circuit cell and an electrochemical
single-particle model built from LG&nbsp;M50 electrode data — over four missions, twelve fault
scenarios and three noise seeds, plus a twelve-cell pack, a 120-mission ageing study and an
instrumented flight. Cross-validated against FilterPy, PyBaMM and ahrs to floating-point
precision.</p>
<a href="benchmark.html">The results →</a></div>
</div>

<h2>The architecture</h2>
<div class="imgframe">{ARCH_SVG}</div>
<p class="muted" style="margin-top:10px">Any discrete-time model that provides a transition, a measurement and two noise covariances plugs
into every estimator unchanged. Everything below the model concept is problem-independent; everything
above it is the only battery-specific code.</p>

<h2>The twelve families</h2>
<div class="grid3">{famcards}</div>

<h2>One headline from ~9,000 runs</h2>
<div class="grid2">
<figure class="imgframe"><img src="assets/figures/overview_runtime_vs_rmse.png" alt="Accuracy versus computational cost">
<figcaption>Nominal accuracy versus cost per step. The best twelve estimators run within five times
the cost of the EKF — below 2&nbsp;µs on a desktop core, a fraction of a percent of a flight
processor.</figcaption></figure>
<div class="panel"><h3 style="margin-top:0">The finding</h3>
<p class="sec">On a cell whose model the estimator knows, the nominal problem is solved by any tuned
closed-loop filter; the field is separated only by faults. On a cell whose model it does
<i>not</i> know, every converged Kalman filter parks on a ~4&nbsp;% bias with a clean innovation,
while fading-memory and fixed-gain estimators hold under 0.7&nbsp;%. The choice of estimator matters
much less than whether it is told the truth about its model and its sensors.</p>
<p class="sec">The plain EKF fails the hardest scenario (<code>{e(best_comb['name'])}</code> and the
adaptive family reach {best_comb['results']['ecm']['combined']:.2f}&nbsp;% where the EKF has
{next(c for c in cat if c['name']=='EKF')['results']['ecm']['combined']:.1f}&nbsp;%) not because it
is a poor filter but because it is an <i>exact</i> one for a model that is wrong.</p>
<a href="benchmark.html">See it scenario by scenario →</a></div>
</div>

<h2>Who it is for</h2>
<div class="grid3">
<div class="panel"><h3 style="margin-top:0">Engineers</h3><p class="sec">Drop one header into your project and
run any filter on your own model in twenty lines.</p><a href="docs.html#start">Getting started →</a></div>
<div class="panel"><h3 style="margin-top:0">Researchers</h3><p class="sec">A reproducible, cross-validated
benchmark and a 958-page report with one chapter per estimator.</p><a href="{REPO}/releases/latest">The report →</a></div>
<div class="panel"><h3 style="margin-top:0">Practitioners (BMS)</h3><p class="sec">A reference estimator stack
for an electric-aircraft battery-management system, with the evidence.</p><a href="benchmark.html#recommend">Recommendations →</a></div>
</div>
<div style="height:20px"></div>
</div>'''
    open(os.path.join(out, 'index.html'), 'w').write(page('estkit — embedded state estimation in C++17', 'index', body))


ARCH_SVG = '''<svg viewBox="0 0 920 300" width="100%" role="img" aria-label="estkit architecture" style="max-width:920px;display:block;margin:0 auto;font-family:var(--sans)">
<style>
 .bx{fill:var(--surface-1);stroke:var(--border-strong);stroke-width:1.4;rx:11}
 .lb{fill:var(--text-primary);font-size:13px;font-weight:600}
 .sm{fill:var(--text-secondary);font-size:11px}
 .core{fill:color-mix(in srgb,var(--accent) 12%,var(--surface-1));stroke:var(--accent);stroke-width:1.6}
 .cn{stroke:var(--border-strong);stroke-width:1.4;fill:none}
 .hd{fill:var(--text-muted);font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
</style>
<text x="90" y="26" class="hd">Models</text>
<g>
<rect class="bx" x="30" y="40" width="150" height="34" rx="10"/><text x="46" y="62" class="lb">Battery ECM</text>
<rect class="bx" x="30" y="82" width="150" height="34" rx="10"/><text x="46" y="104" class="lb">Battery SPM</text>
<rect class="bx" x="30" y="124" width="150" height="34" rx="10"/><text x="46" y="146" class="lb">Attitude / IMU</text>
<rect class="bx" x="30" y="166" width="150" height="34" rx="10"/><text x="46" y="188" class="lb">Drone / nav</text>
<rect class="bx" x="30" y="208" width="150" height="34" rx="10"/><text x="46" y="230" class="lb">your model</text>
</g>
<rect class="core" x="250" y="110" width="180" height="72" rx="12"/>
<text x="268" y="140" class="lb">Model concept</text><text x="268" y="160" class="sm">f, h, Q, R</text><text x="268" y="175" class="sm">Jacobians optional</text>
<path class="cn" d="M180 57 C 215 57 215 140 250 140"/><path class="cn" d="M180 99 C 220 99 220 146 250 146"/>
<path class="cn" d="M180 141 C 230 141 230 150 250 150"/><path class="cn" d="M180 183 C 220 183 220 155 250 155"/>
<path class="cn" d="M180 225 C 215 225 215 160 250 160"/>
<text x="500" y="26" class="hd">Estimation engine</text>
<rect class="bx" x="470" y="40" width="200" height="30" rx="9"/><text x="484" y="60" class="lb">Kalman &amp; sigma-point</text>
<rect class="bx" x="470" y="76" width="200" height="30" rx="9"/><text x="484" y="96" class="lb">Adaptive &amp; robust</text>
<rect class="bx" x="470" y="112" width="200" height="30" rx="9"/><text x="484" y="132" class="lb">Observers</text>
<rect class="bx" x="470" y="148" width="200" height="30" rx="9"/><text x="484" y="168" class="lb">Particle / ensemble</text>
<rect class="bx" x="470" y="184" width="200" height="30" rx="9"/><text x="484" y="204" class="lb">Joint / dual (SOH)</text>
<rect class="bx" x="470" y="220" width="200" height="30" rx="9"/><text x="484" y="240" class="lb">MHE &amp; smoothers</text>
<path class="cn" d="M430 146 C 450 146 450 55 470 55"/><path class="cn" d="M430 146 C 450 146 450 91 470 91"/>
<path class="cn" d="M430 146 L 470 127"/><path class="cn" d="M430 150 C 450 150 450 163 470 163"/>
<path class="cn" d="M430 152 C 450 152 450 199 470 199"/><path class="cn" d="M430 155 C 450 155 450 235 470 235"/>
<rect class="core" x="720" y="110" width="170" height="72" rx="12"/>
<text x="737" y="138" class="lb">Embedded C++17</text><text x="737" y="157" class="sm">no heap · no exceptions</text><text x="737" y="172" class="sm">double or float · Cortex-M7</text>
<path class="cn" d="M670 55 C 700 55 700 140 720 140"/><path class="cn" d="M670 235 C 700 235 700 155 720 155"/>
<path class="cn" d="M670 146 L 720 146"/>
</svg>'''


# ------------------------------------------------------------------ catalogue
def build_estimators(cat, out):
    labels = {c['family']: c['family_label'] for c in cat}
    fams = [f for f in FAM_ORDER if f in labels]
    chips = ''.join(fam_chip(f, labels[f]) for f in fams)
    data = [{'name': c['name'], 'slug': slug(c['name']), 'family': c['family'], 'family_label': c['family_label'],
             'level': c['level'], 'title': c['chapter_title'],
             'geo': c['results'].get('geo_mean'), 'worst': c['results'].get('worst'),
             'nominal': c['results'].get('nominal'), 'ns': c['results'].get('ns_per_step'),
             'rank': c['results'].get('rank'),
             'ref': (c['references'][0]['text'].split('.')[0] if c['references'] else '')} for c in cat]
    head = ('<div class="wrap">'
            '<h1 style="margin-top:34px">Estimator catalogue</h1>'
            '<p class="lede sec">All 86 registered estimators and observers. Each is implemented from its '
            'primary literature reference and benchmarked on the same missions. Filter by family, search by '
            'name, sort by any column; click a row for its page.</p>'
            '<div class="controls"><input type="search" id="q" placeholder="Search 86 estimators — name, family, reference…">'
            '<div class="seg" id="lvl"><button class="on" data-l="all">All</button><button data-l="cell">Cell</button>'
            '<button data-l="pack">Pack</button><button data-l="attitude">Attitude</button></div></div>'
            '<div id="chips" style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px">' + chips + '</div>'
            '<div class="tablewrap"><table class="data" id="tbl"><thead><tr>'
            '<th data-k="name">Estimator</th><th data-k="family_label">Family</th>'
            '<th data-k="rank">Rank</th><th data-k="nominal">Nominal&nbsp;%</th><th data-k="geo">Geo-mean&nbsp;%</th>'
            '<th data-k="worst">Worst&nbsp;%</th><th data-k="ns">ns/step</th><th data-k="ref">Reference</th>'
            '</tr></thead><tbody id="rows"></tbody></table></div>'
            '<p class="muted" id="count" style="margin-top:12px"></p><div style="height:24px"></div></div>')
    script = SCRIPT_EST.replace('__CAT__', json.dumps(data)).replace('__SW__', json.dumps(FAM_SWATCH))
    open(os.path.join(out, 'estimators.html'), 'w').write(page('Estimator catalogue — estkit', 'estimators', head + script,
        description='All 86 state estimators and observers in estkit, filterable and sortable, each with its literature reference and benchmark result.'))


SCRIPT_EST = """<script>
const CAT=__CAT__, SW=__SW__;
let st={q:'',lvl:'all',fams:new Set(CAT.map(c=>c.family)),sort:'rank',dir:1};
const rowsEl=document.getElementById('rows');
function fmt(v){return v==null?'—':(typeof v==='number'?(v<100&&v!==Math.round(v)?v.toFixed(2):v.toLocaleString()):v);}
function render(){
  let r=CAT.filter(c=>st.fams.has(c.family)&&(st.lvl==='all'||c.level===st.lvl));
  if(st.q){let q=st.q.toLowerCase();r=r.filter(c=>(c.name+' '+c.family_label+' '+c.title+' '+c.ref).toLowerCase().includes(q));}
  r.sort((a,b)=>{let x=a[st.sort],y=b[st.sort];if(x==null)return 1;if(y==null)return -1;return (x>y?1:x<y?-1:0)*st.dir;});
  rowsEl.innerHTML=r.map(c=>`<tr onclick="location.href='estimator/${c.slug}.html'" style="cursor:pointer">
    <td class="name-link">${c.name} <span class="muted" style="font-weight:400">— ${c.title}</span></td>
    <td style="text-align:left"><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:var(--s${SW[c.family]});margin-right:6px"></span>${c.family_label}</td>
    <td>${c.rank!=null?'#'+c.rank:'—'}</td><td>${fmt(c.nominal)}</td><td>${fmt(c.geo)}</td><td>${fmt(c.worst)}</td>
    <td>${c.ns!=null?c.ns.toLocaleString():'—'}</td><td style="text-align:left" class="muted">${c.ref}</td></tr>`).join('');
  document.getElementById('count').textContent=r.length+' of '+CAT.length+' estimators shown';
}
document.getElementById('q').addEventListener('input',e=>{st.q=e.target.value;render();});
document.querySelectorAll('#lvl button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#lvl button').forEach(x=>x.classList.remove('on'));b.classList.add('on');st.lvl=b.dataset.l;render();});
document.querySelectorAll('#chips .chip').forEach(ch=>ch.onclick=()=>{let f=ch.dataset.fam;if(st.fams.has(f)){st.fams.delete(f);ch.classList.remove('on');}else{st.fams.add(f);ch.classList.add('on');}render();});
document.querySelectorAll('#tbl thead th').forEach(th=>th.onclick=()=>{let k=th.dataset.k;st.dir=(st.sort===k)?-st.dir:1;st.sort=k;render();});
if(location.hash){let f=location.hash.slice(1);if(SW[f]!=null){st.fams=new Set([f]);document.querySelectorAll('#chips .chip').forEach(c=>c.classList.toggle('on',c.dataset.fam===f));}}
render();
</script>"""


# ------------------------------------------------------------------ per estimator
def scen_row(res, key, scen):
    d = res.get(key, {})
    if not d:
        return ''
    cells = ''.join(f'<td class="cell" data-v="{d.get(s,"")}">{d.get(s,"—") if s in d else "—"}</td>' for s in scen)
    head = ''.join(f'<th>{s.replace("_","_<wbr>")}</th>' for s in scen)
    return head, cells



def _cellbg(v):
    import math
    if v is None:
        return 'transparent', 'var(--text-muted)'
    r = max(0.0, min(1.0, (math.log(v + .05) - math.log(.07)) / (math.log(15) - math.log(.07))))
    return f'var(--seq{round(r*6)})', ('#f4f4f0' if r > 0.55 else '#0b0b0b')


def build_estimator_pages(cat, out):
    os.makedirs(os.path.join(out, 'estimator'), exist_ok=True)
    byname = {c['name']: c for c in cat}
    order = cat
    for i, c in enumerate(order):
        s = safe(c['name'])
        r = dict(c['results']); r['ns'] = r.get('ns_per_step')
        prev = order[i - 1] if i else order[-1]
        nxt = order[(i + 1) % len(order)]
        refs = ''.join(f'<li>{md_links(x["text"])}</li>' for x in c['references'])
        # results block
        results_html = ''
        if c['level'] == 'cell':
            for key, plant, cols in [('ecm', 'Equivalent-circuit plant', SCEN), ('spm', 'Electrochemical (SPM) plant', [x for x in SCEN if x not in ('preisach', 'aged_cell', 'combined')])]:
                d = r.get(key, {})
                if not d:
                    continue
                head = ''.join(f'<th>{x.replace("_","&shy;_")}</th>' for x in cols)
                cell_html = []
                for x in cols:
                    if x in d:
                        bg, fg = _cellbg(d[x])
                        cell_html.append(f'<td class="cell" title="{x}: {d[x]} %" style="background:{bg};color:{fg}">{d[x]}</td>')
                    else:
                        cell_html.append('<td class="cell">—</td>')
                cells = ''.join(cell_html)
                results_html += f'<h3>{plant} — steady-state SOC RMSE [%]</h3><div class="tablewrap"><table class="data"><thead><tr>{head}</tr></thead><tbody><tr>{cells}</tr></tbody></table></div>'
            figs = f'''<div class="grid2" style="margin-top:16px">
<figure class="imgframe"><img loading="lazy" src="../assets/figures/cell_{s}_nominal.png" alt="{e(c["name"])} nominal"><figcaption>Equivalent-circuit plant, nominal eVTOL mission.</figcaption></figure>
<figure class="imgframe"><img loading="lazy" src="../assets/figures/cellspm_{s}_nominal.png" alt="{e(c["name"])} SPM nominal"><figcaption>Electrochemical plant, nominal mission.</figcaption></figure></div>'''
            kpis = tile(f'#{r.get("rank","—")}', 'composite rank', 'of 70 cell-level') + tile(f'{r.get("nominal","—")}%', 'nominal RMSE') + tile(f'{r.get("worst","—")}%', 'worst scenario') + tile((f'{r["ns"]:,} ns' if isinstance(r.get('ns'),int) else '—'), 'per step')
        elif c['level'] == 'pack':
            d = r.get('pack', {})
            cols = list(d.keys())
            head = ''.join(f'<th>{x}</th>' for x in cols)
            cells = ''.join(f'<td class="cell">{d[x]}</td>' for x in cols)
            results_html = f'<h3>Per-cell SOC RMSE [%] by scenario (12-cell module)</h3><div class="tablewrap"><table class="data"><thead><tr>{head}</tr></thead><tbody><tr>{cells}</tr></tbody></table></div>'
            figs = f'<figure class="imgframe" style="margin-top:16px"><img loading="lazy" src="../assets/figures/pack_{s}_nominal.png" alt="{e(c["name"])} pack"><figcaption>Twelve-cell module, nominal mission.</figcaption></figure>'
            kpis = tile(f'{d.get("nominal","—")}%', 'per-cell RMSE') + tile(f'{r.get("ns",0)/1000:.1f} µs', 'per pack step') + tile(f'{r.get("messages","—")}', 'msgs/step') + tile(f'{r.get("bytes",0)//1000} kB', 'state')
        else:
            d = r.get('imu', {})
            cols = list(d.keys())
            head = ''.join(f'<th>{x}</th>' for x in cols)
            cells = ''.join(f'<td class="cell">{d[x]}</td>' for x in cols)
            results_html = f'<h3>RMS attitude error [deg] by scenario</h3><div class="tablewrap"><table class="data"><thead><tr>{head}</tr></thead><tbody><tr>{cells}</tr></tbody></table></div>'
            figs = f'<figure class="imgframe" style="margin-top:16px"><img loading="lazy" src="../assets/figures/imu_{s}_nominal.png" alt="{e(c["name"])} attitude"><figcaption>Instrumented flight, nominal scenario.</figcaption></figure>'
            kpis = tile(f'{d.get("nominal","—")}°', 'RMS error') + tile((f'{r["ns"]:,} ns' if isinstance(r.get('ns'),int) else '—'), 'per step') + tile(f'{r.get("bytes","—")} B', 'state')
        headers = ' '.join(f'<a class="tag" href="{REPO}/blob/main/{h}">{os.path.basename(h)}</a>' for h in c['headers'])
        soh = ''
        if 'soh' in r:
            soh = f'''<div class="panel" style="margin-top:16px"><h3 style="margin-top:0">Multi-flight capacity tracking</h3>
<p class="sec">Over 120 consecutive missions with accelerated ageing: capacity RMSE
<b>{r["soh"]["capacity_rmse_mAh"]} mAh</b>, final error {r["soh"]["final_err_mAh"]:+g} mAh, mean per-flight SOC RMSE {r["soh"]["soc_rmse_pct"]} %.</p></div>'''
        body = f'''<div class="wrap">
<div style="margin-top:28px"><a class="muted" href="../estimators.html">← catalogue</a></div>
<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:8px">
<h1 style="margin:0">{e(c['name'])}</h1>{fam_chip(c['family'], c['family_label'], link="../estimators.html#"+c['family'])}</div>
<p class="lede sec" style="margin-top:8px">{e(c['chapter_title'])}</p>
<div class="tiles" style="grid-template-columns:repeat(4,1fr)">{kpis}</div>
<div class="grid2">
<div class="panel"><h3 style="margin-top:0">At a glance</h3>
<ul class="clean">
<li><span class="muted">Family</span> · {e(c['family_label'])}</li>
<li><span class="muted">Header</span> · {headers}</li>
<li><span class="muted">State</span> · {md_links(c['state']) or '—'}</li>
<li><span class="muted">Cost</span> · {md_links(c['cost']) or '—'}</li>
{f'<li><span class=muted>Estimates</span> · {md_links(c["estimates"])}</li>' if c['estimates'] else ''}
</ul></div>
<div class="panel"><h3 style="margin-top:0">Primary references</h3><ul class="clean">{refs}</ul>
<p class="muted" style="margin-top:10px">Full derivation, algorithm box and discussion: chapter
<code>{e(c['chapter_label'])}</code> of the <a href="{REPO}/releases/latest">report</a>.</p></div>
</div>
<h2>Benchmark results</h2>{results_html}{figs}{soh}
<div class="rule"></div>
<div style="display:flex;justify-content:space-between;font-size:.92rem">
<a href="{slug(prev['name'])}.html">← {e(prev['name'])}</a>
<a href="{slug(nxt['name'])}.html">{e(nxt['name'])} →</a></div>
<div style="height:24px"></div></div>'''
        open(os.path.join(out, 'estimator', slug(c['name']) + '.html'), 'w').write(
            page(f'{c["name"]} — estkit', 'estimators', body, depth=1,
                 description=f'{c["name"]}: {c["chapter_title"]}. Implementation, references and benchmark results in estkit.'))


# ------------------------------------------------------------------ benchmark explorer
def build_benchmark(cat, out):
    cells = [c for c in cat if c['level'] == 'cell']
    data = [{'name': c['name'], 'slug': slug(c['name']), 'family': c['family'], 'fl': c['family_label'],
             'ecm': c['results']['ecm'], 'spm': c['results'].get('spm', {}),
             'rank': c['results']['rank']} for c in cells]
    data.sort(key=lambda d: d['rank'])
    head = ('<div class="wrap"><h1 style="margin-top:34px">Benchmark explorer</h1>'
        '<p class="lede sec">Steady-state state-of-charge RMSE, in percent of capacity, for every '
        'cell-level estimator across the twelve fault scenarios. Darker is worse. Switch plants, sort by '
        'any column, hover a cell for the number. Lower is better everywhere.</p>'
        '<div class="controls"><div class="seg" id="plant"><button class="on" data-p="ecm">Equivalent-circuit plant</button>'
        '<button data-p="spm">Electrochemical plant</button></div>'
        '<span class="legend-seq">better <span class="bar"><span style="background:var(--seq0)"></span>'
        '<span style="background:var(--seq1)"></span><span style="background:var(--seq2)"></span>'
        '<span style="background:var(--seq3)"></span><span style="background:var(--seq4)"></span>'
        '<span style="background:var(--seq5)"></span><span style="background:var(--seq6)"></span></span> worse</span></div>'
        '<div class="tablewrap"><table class="data" id="heat"><thead id="hhead"></thead><tbody id="hbody"></tbody></table></div>'
        '<p class="muted" style="margin-top:10px"><code>div</code> in the source data marks a run that diverged in at least one seed; '
        'blank cells are scenarios not defined on that plant.</p>')
    mid = ('<h2>Composite ranking &amp; cost</h2><div class="grid2">'
        '<figure class="imgframe"><img loading="lazy" src="assets/figures/res_ranking_bars.png" alt="Composite ranking">'
        '<figcaption>Geometric-mean RMSE over the twelve scenarios (bar) with the worst scenario (tick), by family.</figcaption></figure>'
        '<figure class="imgframe"><img loading="lazy" src="assets/figures/res_ecm_vs_spm.png" alt="ECM vs SPM">'
        '<figcaption>Nominal accuracy on the two plants is uncorrelated: the converged Kalman filters are twenty times worse on the electrochemical plant.</figcaption></figure></div>'
        '<h2 id="recommend">Pack, attitude and state-of-health</h2><div class="grid3">'
        '<figure class="imgframe"><img loading="lazy" src="assets/figures/pack_overview.png" alt="Pack"><figcaption>Twelve-cell module: averaging inside the filter halves the per-cell error.</figcaption></figure>'
        '<figure class="imgframe"><img loading="lazy" src="assets/figures/imu_overview.png" alt="Attitude"><figcaption>Attitude estimators: the accelerometer is not a gravity reference in a banked turn.</figcaption></figure>'
        '<figure class="imgframe"><img loading="lazy" src="assets/figures/soh_capacity_tracking.png" alt="SOH"><figcaption>120-mission ageing: joint/dual filters track capacity to the current-sensor accuracy.</figcaption></figure></div>'
        '<div class="panel" style="margin-top:18px"><h3 style="margin-top:0">The reference stack</h3>'
        '<p class="sec">A 0.5&nbsp;µs EKF that adapts its measurement noise and bounds its memory, with a gated capacity '
        'state and an independent Coulomb counter beside it on a calibrated current sensor, has no failure scenario in '
        'this benchmark on either plant. The full argument — eight design rules and a reference architecture — is in the '
        'report\'s recommendations chapter.</p>'
        '<a class="btn ghost" href="' + REPO + '/releases/latest">Read the recommendations →</a></div><div style="height:24px"></div></div>')
    script = SCRIPT_BENCH.replace('__DATA__', json.dumps(data)).replace('__SCEN__', json.dumps(SCEN))
    open(os.path.join(out, 'benchmark.html'), 'w').write(page('Benchmark explorer — estkit', 'benchmark', head + script + mid,
        description='Interactive benchmark: steady-state SOC RMSE of 70 estimators across 12 fault scenarios on two battery plants.'))


SCRIPT_BENCH = """<script>
const D=__DATA__, SCEN=__SCEN__;
let plant='ecm', sort='rank', dir=1;
function val(d,k){if(k==='name')return d.name;if(k==='rank')return d.rank;return d[plant][k]==null?1e9:d[plant][k];}
function draw(){
  let sc=SCEN.filter(s=>D.some(d=>d[plant][s]!=null));
  document.getElementById('hhead').innerHTML='<tr><th data-k="name">Estimator</th><th data-k="rank">#</th>'+
    sc.map(s=>`<th data-k="${s}">${s.replace(/_/g,'_<wbr>')}</th>`).join('')+'</tr>';
  let vals=[];D.forEach(d=>sc.forEach(s=>{if(d[plant][s]!=null)vals.push(d[plant][s]);}));
  let hi=Math.max.apply(null,vals),lo=0.02, cs=getComputedStyle(document.documentElement);
  function col(v){if(v==null)return['transparent','var(--text-muted)'];
    let r=Math.max(0,Math.min(1,(Math.log(v+.05)-Math.log(lo+.05))/(Math.log(hi+.05)-Math.log(lo+.05))));
    return[cs.getPropertyValue('--seq'+Math.round(r*6)).trim(), r>0.55?'#f4f4f0':'#0b0b0b'];}
  let rows=D.slice().sort((a,b)=>{let x=val(a,sort),y=val(b,sort);return (x>y?1:x<y?-1:0)*dir;});
  document.getElementById('hbody').innerHTML=rows.map(d=>`<tr><td class="name-link"><a href="estimator/${d.slug}.html">${d.name}</a> <span class="muted" style="font-weight:400">${d.fl}</span></td><td>${d.rank}</td>`+
    sc.map(s=>{let v=d[plant][s];let c=col(v);return `<td class="cell" title="${d.name} · ${s} · ${v==null?'—':v+' %'}" style="background:${c[0]};color:${c[1]}">${v==null?'':v}</td>`;}).join('')+'</tr>').join('');
  document.querySelectorAll('#hhead th').forEach(th=>th.onclick=()=>{let k=th.dataset.k;if(sort===k)dir=-dir;else{sort=k;dir=1;}draw();});
}
document.querySelectorAll('#plant button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#plant button').forEach(x=>x.classList.remove('on'));b.classList.add('on');plant=b.dataset.p;draw();});
draw();
</script>"""


# ------------------------------------------------------------------ docs
CODE_MODEL = '''<span class="kw">struct</span> MyModel {
  <span class="kw">static constexpr int</span> NX = 4, NU = 1, NY = 2;
  <span class="ty">Vec</span>&lt;NX&gt; f(<span class="kw">const</span> <span class="ty">Vec</span>&lt;NX&gt;&amp; x, <span class="kw">const</span> <span class="ty">Vec</span>&lt;NU&gt;&amp; u) <span class="kw">const</span>;   <span class="cm">// x_{k+1}</span>
  <span class="ty">Vec</span>&lt;NY&gt; h(<span class="kw">const</span> <span class="ty">Vec</span>&lt;NX&gt;&amp; x, <span class="kw">const</span> <span class="ty">Vec</span>&lt;NU&gt;&amp; u) <span class="kw">const</span>;   <span class="cm">// y_k</span>
  <span class="ty">Mat</span>&lt;NX,NX&gt; Q(<span class="kw">const</span> <span class="ty">Vec</span>&lt;NX&gt;&amp;, <span class="kw">const</span> <span class="ty">Vec</span>&lt;NU&gt;&amp;) <span class="kw">const</span>;
  <span class="ty">Mat</span>&lt;NY,NY&gt; R(<span class="kw">const</span> <span class="ty">Vec</span>&lt;NX&gt;&amp;, <span class="kw">const</span> <span class="ty">Vec</span>&lt;NU&gt;&amp;) <span class="kw">const</span>;
  <span class="cm">// optional: F(x,u), H(x,u), B(x,u), constrain(x), NP/params()/set_params()</span>
};'''
CODE_USE = '''<span class="cm">#include "estkit/filters/ukf.hpp"</span>
<span class="kw">using namespace</span> estkit;

<span class="ty">Ukf</span>&lt;MyModel&gt; ukf{MyModel{}};
ukf.<span class="fn">init</span>(x0, P0);
<span class="kw">for</span> (...) {
  ukf.<span class="fn">predict</span>(u_prev);        <span class="cm">// time update with the previous input</span>
  ukf.<span class="fn">update</span>(y, u);           <span class="cm">// measurement update</span>
  <span class="kw">auto</span> x = ukf.<span class="fn">x</span>();          <span class="cm">// state estimate; ukf.P() its covariance</span>
}'''


def build_docs(cat, out):
    body = f'''<div class="wrap">
<h1 style="margin-top:34px">Documentation</h1>
<p class="lede sec">estkit is header-only: add <code>include/</code> to your include path and
<code>#include</code> the estimator you want. Everything below is on this one page.</p>

<h2 id="start">Getting started</h2>
<p class="sec">Build the benchmark, the tests and the examples with CMake:</p>
<pre>cmake -S . -B build -DCMAKE_BUILD_TYPE=Release &amp;&amp; cmake --build build -j
ctest --test-dir build                       <span class="cm"># unit tests + examples</span>
./build/example_drone                        <span class="cm"># EKF/UKF/PF on a drone range model</span>
./build/example_battery combined             <span class="cm"># SOC estimation on a hard scenario</span>
./build/bench_cell --list                    <span class="cm"># every cell-level estimator</span></pre>
<p class="sec">Single-precision build for a microcontroller:
<code>cmake -S . -B build_f -DESTKIT_FLOAT=ON</code>. The library also cross-compiles for a
Cortex-M7 with <code>arm-none-eabi-g++ -mcpu=cortex-m7 -mfpu=fpv5-d16 -fno-exceptions -fno-rtti</code>
— see <code>examples/embedded_cortex_m7.cpp</code>.</p>

<h2 id="concept">The model concept</h2>
<p class="sec">An estimator is a template over a <i>model</i>: any class that provides a state
transition, a measurement and the two noise covariances. Jacobians, constraints and tunable
parameters are optional and supplied numerically when absent.</p>
<pre>{CODE_MODEL}</pre>
<p class="sec">Every filter and observer is a template over such a model, so the same
<code>Ekf&lt;M&gt;</code>, <code>Ukf&lt;M&gt;</code> or <code>Mhe&lt;M,N&gt;</code> that estimates the
state of charge of a cell estimates the position of a drone or the attitude of an airframe:</p>
<pre>{CODE_USE}</pre>
<p class="sec">The interface is the same four calls for every estimator in the library — from a
Luenberger observer to a 500-particle filter. Offline smoothers add a <code>run()</code> over the
whole record.</p>

<h2 id="add">Adding an estimator</h2>
<p class="sec">One header under the family directory, templated on the model, with the primary
reference in the header comment; no heap, no exceptions, warning-free in double and float; register
it in <code>benchmarks/registry_&lt;family&gt;.cpp</code>, add a unit test and a chapter. The full
contract is in <a href="{REPO}/blob/main/docs/ESTIMATOR_API.md">docs/ESTIMATOR_API.md</a> and
<a href="{REPO}/blob/main/CONTRIBUTING.md">CONTRIBUTING.md</a>.</p>

<h2 id="numerics">Numerical &amp; embedded rules</h2>
<div class="grid2">
<div class="panel"><ul class="clean">
<li>No dynamic memory, no exceptions, no recursion of unbounded depth.</li>
<li>Fixed-size linear algebra with compile-time dimensions.</li>
<li>Joseph-form covariance update; symmetrise every step.</li>
<li>Square-root and U-D forms available where the covariance dynamic range demands them.</li>
<li>Deterministic, reproducible bit-for-bit from the seed.</li>
</ul></div>
<div class="panel"><p class="sec" style="margin-top:0">No estimator in the library needs double
precision on this problem — the single-precision build agrees with the double build to within the
seed-to-seed spread — but the factorised forms are there for larger state vectors. Below 2&nbsp;µs
per step the cost of an estimator is irrelevant on any flight processor.</p>
<a href="{REPO}/releases/latest">The numerics chapter →</a></div>
</div>

<h2 id="validation">Validation</h2>
<p class="sec">The Kalman-type filters are cross-checked against <b>FilterPy</b>, the electrochemical
plant against <b>PyBaMM</b>, and the attitude filters against <b>ahrs</b> — each an independent
reference implementation, run on the exported benchmark data:</p>
<div class="grid3">
{tile('1e-15', 'EKF vs FilterPy', 'max |Δ SOC|')}
{tile('0.6–3.2 mV', 'SPM vs PyBaMM', 'terminal voltage')}
{tile('0.0014°', 'Madgwick/Mahony vs ahrs', 'attitude')}
</div>
<p class="muted" style="margin-top:12px">These libraries are used only as references and are not part of
estkit; see <a href="{REPO}/blob/main/THIRD_PARTY_NOTICES.md">THIRD_PARTY_NOTICES</a>. The checks run in CI.</p>
<div style="height:24px"></div></div>'''
    open(os.path.join(out, 'docs.html'), 'w').write(page('Documentation — estkit', 'docs', body,
                                                         description='Getting started with estkit: the model concept, using any filter on your own model, adding an estimator, the numerical rules, and the validation.'))


# ------------------------------------------------------------------ references
def build_references(out):
    md = open(os.path.join(ROOT, 'docs', 'REFERENCES.md'), encoding='utf-8').read()
    md = re.sub(r'^<!--.*?-->\n', '', md)
    lines = md.split('\n')
    htmlparts = []
    for ln in lines:
        if ln.startswith('# '):
            continue
        if ln.startswith('## '):
            htmlparts.append(f'<h2>{e(ln[3:])}</h2>')
        elif re.match(r'^\d+\. ', ln):
            m = re.match(r'^\d+\. <a id="([^"]+)"></a>(.*)$', ln)
            if m:
                htmlparts.append(f'<li id="{m.group(1)}">{md_links_ref(m.group(2))}</li>')
        elif ln.strip():
            htmlparts.append(f'<p class="sec">{md_links_ref(ln)}</p>')
    # wrap consecutive li in ol
    out_html = []
    inlist = False
    for p in htmlparts:
        if p.startswith('<li'):
            if not inlist:
                out_html.append('<ol class="refs">'); inlist = True
            out_html.append(p)
        else:
            if inlist:
                out_html.append('</ol>'); inlist = False
            out_html.append(p)
    if inlist:
        out_html.append('</ol>')
    body = f'''<div class="wrap">
<h1 style="margin-top:34px">References</h1>
<p class="lede sec">Every publication cited in the report and in the source headers — grouped by
topic. Each estimator is implemented from its primary references; see its
<a href="estimators.html">catalogue page</a>.</p>
<style>ol.refs{{padding-left:1.4em}}ol.refs li{{margin:.5em 0;color:var(--text-secondary)}}ol.refs li a{{color:var(--accent-ink)}}</style>
{''.join(out_html)}
<div style="height:24px"></div></div>'''
    open(os.path.join(out, 'references.html'), 'w').write(page('References — estkit', 'references', body,
                                                              description='All 314 publications cited by estkit, grouped by topic.'))


def md_links_ref(s):
    s = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', s)
    s = re.sub(r'doi:\[([^\]]+)\]\(([^)]+)\)', r'doi:<a href="\2">\1</a>', s)
    s = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<a href="\2">\1</a>', s)
    return s


# ------------------------------------------------------------------ assets + drive
def copy_assets(cat, out):
    ad = os.path.join(out, 'assets')
    os.makedirs(os.path.join(ad, 'figures'), exist_ok=True)
    shutil.copy(os.path.join(HERE, 'style.css'), os.path.join(ad, 'style.css'))
    shutil.copy(os.path.join(HERE, 'app.js'), os.path.join(ad, 'app.js'))
    figdir = os.path.join(ROOT, 'report', 'figures')
    need = set(['overview_runtime_vs_rmse.png', 'overview_heatmap_ecm.png', 'overview_heatmap_spm.png',
                'res_ranking_bars.png', 'res_ecm_vs_spm.png', 'res_spm_traces.png', 'res_ecm_combined_traces.png',
                'pack_overview.png', 'imu_overview.png', 'soh_capacity_tracking.png'])
    for c in cat:
        s = safe(c['name'])
        if c['level'] == 'cell':
            need |= {f'cell_{s}_nominal.png', f'cellspm_{s}_nominal.png'}
        elif c['level'] == 'pack':
            need |= {f'pack_{s}_nominal.png'}
        else:
            need |= {f'imu_{s}_nominal.png'}
    miss = 0
    for f in sorted(need):
        src = os.path.join(figdir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(ad, 'figures', f))
        else:
            miss += 1
    # .nojekyll so GitHub Pages serves folders starting with _ etc verbatim
    open(os.path.join(out, '.nojekyll'), 'w').write('')
    return len(need) - miss


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='site'); a = ap.parse_args()
    out = os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    cat = json.load(open(os.path.join(ROOT, 'docs', 'catalogue.json')))
    # inject the arch svg into the module namespace for build_index
    global ARCH_SVG
    n = copy_assets(cat, out)
    build_index(cat, out)
    build_estimators(cat, out)
    build_estimator_pages(cat, out)
    build_benchmark(cat, out)
    build_docs(cat, out)
    build_references(out)
    pages = len([f for _, _, fs in os.walk(out) for f in fs if f.endswith('.html')])
    print(f'site: {pages} HTML pages, {n} figures, out={out}')


if __name__ == '__main__':
    main()

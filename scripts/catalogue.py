#!/usr/bin/env python3
# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
"""Build the machine-readable estimator catalogue and the reference list from the sources of truth:
the report chapters (At-a-glance tables), the bibliography files, the benchmark summaries and the
registries.

    python3 scripts/catalogue.py [--results results/final] [--out docs]

Writes
  docs/catalogue.json   one record per registered estimator (86): name, family, chapter, header,
                        primary references (resolved), state/cost lines, benchmark figures of merit
  docs/CATALOGUE.md     the same as a Markdown table (embedded in the README)
  docs/REFERENCES.md    every bibliography entry (314), grouped by family file, formatted
  docs/bib.json         the parsed bibliography (used by the website generator)
"""
import os, re, sys, json, glob, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILY_LABEL = {'baselines': 'Baselines', 'kalman': 'Classical & sigma-point Kalman', 'adaptive_kalman': 'Adaptive Kalman',
                'robust_kalman': 'Robust & embedded Kalman', 'observers': 'Deterministic observers', 'soh': 'Joint / dual state–parameter (SOH)',
                'particle': 'Particle, ensemble & Gaussian-sum', 'learning': 'Data-driven hybrids', 'optimization': 'Moving-horizon estimation',
                'smoothers': 'Smoothers (offline)', 'pack': 'Pack-level & distributed', 'attitude': 'Attitude (AHRS)'}
FAMILY_ORDER = list(FAMILY_LABEL)
BIB_GROUP = {'refs': 'Core (battery modelling, estimation theory, standards)', 'refs_additions_front': 'Foundations, datasets, methodology, library',
             'refs_additions_kalman': 'Classical & sigma-point Kalman filters', 'refs_additions_adaptive_kalman': 'Adaptive Kalman filters',
             'refs_additions_robust_kalman': 'Robust & embedded Kalman filters', 'refs_additions_observers': 'Deterministic observers',
             'refs_additions_soh': 'Joint / dual state–parameter estimation', 'refs_additions_particle': 'Particle, ensemble & Gaussian-sum filters',
             'refs_additions_learning': 'Data-driven hybrids', 'refs_additions_optimization': 'Moving-horizon estimation',
             'refs_additions_smoothers': 'Smoothers', 'refs_additions_pack': 'Pack-level estimation', 'refs_additions_attitude': 'Attitude estimation'}


# ----------------------------------------------------------------------------- bibliography
def parse_bib(path):
    s = open(path, encoding='utf-8').read()
    out = []
    i = 0
    while True:
        m = re.search(r'@(\w+)\s*\{', s[i:])
        if not m:
            break
        start = i + m.end()
        typ = m.group(1).lower()
        if typ in ('comment', 'preamble', 'string'):
            i = start; continue
        depth, j = 1, start
        while depth and j < len(s):
            if s[j] == '{': depth += 1
            elif s[j] == '}': depth -= 1
            j += 1
        body = s[start:j - 1]
        i = j
        key, _, fields = body.partition(',')
        rec = {'type': typ, 'key': key.strip()}
        k = 0
        while k < len(fields):
            m2 = re.match(r'\s*(\w+)\s*=\s*', fields[k:])
            if not m2:
                break
            name = m2.group(1).lower(); k += m2.end()
            if fields[k] == '{':
                depth, l = 1, k + 1
                while depth and l < len(fields):
                    if fields[l] == '{': depth += 1
                    elif fields[l] == '}': depth -= 1
                    l += 1
                val = fields[k + 1:l - 1]; k = l
            elif fields[k] == '"':
                l = fields.index('"', k + 1); val = fields[k + 1:l]; k = l + 1
            else:
                m3 = re.match(r'[^,]*', fields[k:]); val = m3.group(0); k += m3.end()
            rec[name] = clean(val)
            m4 = re.match(r'\s*,', fields[k:])
            if m4: k += m4.end()
            else: break
        out.append(rec)
    return out


def clean(v):
    v = re.sub(r'\s+', ' ', v.strip())
    v = v.replace('---', '—').replace('--', '–').replace('~', ' ').replace('\\&', '&')
    v = re.sub(r"\{\\'([a-zA-Z])\}", lambda m: m.group(1), v)
    v = re.sub(r'\\"\{?([a-zA-Z])\}?', lambda m: {'a': 'ä', 'o': 'ö', 'u': 'ü', 'A': 'Ä', 'O': 'Ö', 'U': 'Ü', 'e': 'ë', 'i': 'ï'}.get(m.group(1), m.group(1)), v)
    v = re.sub(r"\\'\{?([a-zA-Z])\}?", lambda m: {'e': 'é', 'a': 'á', 'o': 'ó', 'i': 'í', 'u': 'ú', 'c': 'ć', 'n': 'ń', 'y': 'ý'}.get(m.group(1), m.group(1)), v)
    v = re.sub(r"\\`\{?([a-zA-Z])\}?", lambda m: {'e': 'è', 'a': 'à'}.get(m.group(1), m.group(1)), v)
    v = re.sub(r'\\\^\{?([a-zA-Z])\}?', lambda m: {'e': 'ê', 'a': 'â', 'o': 'ô'}.get(m.group(1), m.group(1)), v)
    v = re.sub(r'\\v\{?([a-zA-Z])\}?', lambda m: {'c': 'č', 's': 'š', 'z': 'ž', 'C': 'Č', 'S': 'Š'}.get(m.group(1), m.group(1)), v)
    v = re.sub(r'\\c\{?([a-zA-Z])\}?', lambda m: {'c': 'ç'}.get(m.group(1), m.group(1)), v)
    v = v.replace('{\\o}', 'ø').replace('\\o', 'ø').replace('{\\ss}', 'ß').replace('\\ss', 'ß').replace('{\\aa}', 'å').replace('\\aa', 'å')
    v = re.sub(r'\\(emph|textit|textbf|mathrm|text)\{([^}]*)\}', r'\2', v)
    v = re.sub(r'\$([^$]*)\$', r'\1', v)
    v = v.replace('\\infty', '∞').replace('\\ell', 'ℓ').replace('_', '')
    v = re.sub(r'[{}]', '', v)
    return v.strip()


def authors(rec):
    a = rec.get('author', rec.get('editor', ''))
    parts = [p.strip() for p in re.split(r'\s+and\s+', a) if p.strip()]
    names = []
    for p in parts:
        if ',' in p:
            last, first = [x.strip() for x in p.split(',', 1)]
        else:
            toks = p.split(); last, first = toks[-1], ' '.join(toks[:-1])
        initials = ' '.join(t[0] + '.' for t in re.split(r'[\s.-]+', first) if t)
        names.append((last + (', ' + initials if initials else '')))
    if len(names) > 6:
        return ', '.join(names[:6]) + ' et al.'
    return ', '.join(names[:-1]) + (' and ' if len(names) > 1 else '') + names[-1] if names else ''


def fmt(rec):
    t = rec['type']; a = authors(rec); y = rec.get('year', 'n.d.'); title = rec.get('title', '')
    s = f'{a} ({y}). {title}.'
    if t == 'article':
        s += f" *{rec.get('journal', '')}*"
        if rec.get('volume'): s += f" {rec['volume']}"
        if rec.get('number'): s += f"({rec['number']})"
        if rec.get('pages'): s += f", {rec['pages']}"
        s += '.'
    elif t in ('inproceedings', 'incollection', 'conference'):
        s += f" In *{rec.get('booktitle', '')}*"
        if rec.get('pages'): s += f", pp. {rec['pages']}"
        s += '.'
    elif t == 'book':
        s += f" {rec.get('publisher', '')}"
        if rec.get('edition'): s += f", {rec['edition']} ed."
        s += '.'
    elif t in ('phdthesis', 'mastersthesis'):
        s += f" {'PhD thesis' if t == 'phdthesis' else 'Master thesis'}, {rec.get('school', '')}."
    elif t == 'techreport':
        s += f" {rec.get('institution', '')}"
        if rec.get('number'): s += f", {rec['number']}"
        s += '.'
    else:
        if rec.get('howpublished'): s += f" {rec['howpublished']}."
        elif rec.get('publisher'): s += f" {rec['publisher']}."
    if rec.get('doi'):
        s += f" doi:[{rec['doi']}](https://doi.org/{rec['doi']})"
    elif rec.get('url'):
        s += f" [{rec['url']}]({rec['url']})"
    return s


# ----------------------------------------------------------------------------- chapters
def detex(v):
    v = re.sub(r'\\texttt\{([^}]*)\}', r'`\1`', v)
    v = re.sub(r'\\code\{([^}]*)\}', r'`\1`', v)
    v = re.sub(r'\\cref\{[^}]*\}|Chapter~\\ref\{[^}]*\}', 'its chapter', v)
    v = v.replace('\\_', '_').replace('\\%', '%').replace('~', ' ').replace('\\,', ' ').replace('---', '—').replace('--', '–')
    v = re.sub(r'\\SI\{([^}]*)\}\{([^}]*)\}', lambda m: m.group(1) + ' ' + m.group(2).replace('\\micro\\second', 'µs').replace('\\nano\\second', 'ns').replace('\\percent', '%').replace('\\', ''), v)
    v = re.sub(r'\\num\{([^}]*)\}', r'\1', v)
    v = re.sub(r'\\(mathcal|mathrm|text|textbf|emph)\{([^}]*)\}', r'\2', v)
    v = v.replace('\\approx', '≈').replace('\\times', '×').replace('\\mu', 'µ').replace('\\cdot', '·').replace('\\ldots', '…')
    v = re.sub(r'\\(?:[a-zA-Z]+)', '', v)
    v = re.sub(r'[{}$]', '', v)
    return re.sub(r'\s+', ' ', v).strip()


def parse_chapters():
    ch = []
    for p in sorted(glob.glob(os.path.join(ROOT, 'report/chapters/*/*.tex'))):
        fam = p.split(os.sep)[-2]
        if fam in ('front', 'appendix', 'results'):
            continue
        s = open(p, encoding='utf-8').read()
        m = re.search(r'\\chapter\{(.*?)\}\s*\\label\{(ch:[^}]+)\}', s, re.S)
        g = re.search(r'\\section\*\{At a glance\}(.*?)\\end\{tabular\}', s, re.S)
        if not (m and g):
            continue
        d = {}
        for line in g.group(1).split('\\\\'):
            if '&' in line:
                k, v = line.split('&', 1)
                k = k.strip().split('\n')[-1].strip()
                d[k] = v.strip()
        refs = re.findall(r'\\cite[pt]?\{([^}]*)\}', d.get('Primary references', d.get('Primary reference', '')))
        keys = [k.strip() for r in refs for k in r.split(',')]
        names = re.findall(r'\\texttt\{([^}]*)\}', d.get('Registered name', d.get('Registered names', '')))
        headers = re.findall(r'\\texttt\{(include/estkit/[^}]*)\}', d.get('Header', d.get('Headers', '')))
        ch.append({'file': os.path.relpath(p, ROOT), 'family_dir': fam, 'title': detex(m.group(1)), 'label': m.group(2),
                   'registered': [n.replace('\\_', '_') for n in names], 'headers': [h.replace('\\_', '_') for h in headers],
                   'state': detex(d.get('State dimension', d.get('State', ''))), 'cost': detex(d.get('Cost per step', d.get('Cost', ''))),
                   'estimates': detex(d.get('Estimates', d.get('Structure', ''))), 'refs': keys})
    return ch


# ----------------------------------------------------------------------------- results
def results(resdir):
    out = {}
    e = pd.read_csv(os.path.join(resdir, 'summary_ecm.csv')); e = e[e.profile == 'evtol']
    g = e.groupby(['estimator', 'scenario']).rmse_ss.mean().unstack() * 100
    geo = np.exp(np.log(g.clip(lower=1e-3)).mean(axis=1)); worst = g.max(axis=1)
    cost = e[e.scenario == 'nominal'].groupby('estimator').agg(ns=('ns_per_step', 'median'), by=('bytes', 'first'))
    fam = e.groupby('estimator')['group'].first()
    rank = geo.sort_values().index.tolist()
    s = pd.read_csv(os.path.join(resdir, 'summary_spm.csv'))
    gs = s.groupby(['estimator', 'scenario']).rmse_ss.mean().unstack() * 100
    for est in g.index:
        out[est] = {'family': fam[est], 'level': 'cell', 'ecm': {c: round(float(g.loc[est, c]), 2) for c in g.columns},
                    'spm': {c: round(float(gs.loc[est, c]), 2) for c in gs.columns} if est in gs.index else {},
                    'geo_mean': round(float(geo[est]), 2), 'worst': round(float(worst[est]), 2), 'nominal': round(float(g.loc[est, 'nominal']), 2),
                    'ns_per_step': int(round(cost.loc[est, 'ns'])), 'bytes': int(cost.loc[est, 'by']), 'rank': rank.index(est) + 1}
    p = pd.read_csv(os.path.join(resdir, 'summary_pack.csv'))
    gp = p.groupby(['estimator', 'scenario']).rmse_cells.mean().unstack() * 100
    cp = p[p.scenario == 'nominal'].groupby('estimator').agg(ns=('ns_per_step', 'median'), by=('bytes', 'first'), msg=('messages', 'first'))
    for est in gp.index:
        out[est] = {'family': 'pack', 'level': 'pack', 'pack': {c: round(float(gp.loc[est, c]), 2) for c in gp.columns},
                    'nominal': round(float(gp.loc[est, 'nominal']), 2), 'ns_per_step': int(round(cp.loc[est, 'ns'])), 'bytes': int(cp.loc[est, 'by']), 'messages': int(cp.loc[est, 'msg'])}
    i = pd.read_csv(os.path.join(resdir, 'summary_imu.csv'))
    gi = i.groupby(['estimator', 'scenario']).rms_angle_deg.mean().unstack()
    ci = i[i.scenario == 'nominal'].groupby('estimator').agg(ns=('ns_per_step', 'median'), by=('bytes', 'first'))
    for est in gi.index:
        out[est] = {'family': 'attitude', 'level': 'attitude', 'imu': {c: round(float(gi.loc[est, c]), 2) for c in gi.columns},
                    'nominal': round(float(gi.loc[est, 'nominal']), 2), 'ns_per_step': int(round(ci.loc[est, 'ns'])), 'bytes': int(ci.loc[est, 'by'])}
    soh = os.path.join(resdir, 'summary_soh.csv')
    if os.path.exists(soh):
        for _, r in pd.read_csv(soh).iterrows():
            if r.estimator in out:
                out[r.estimator]['soh'] = {'capacity_rmse_mAh': round(float(r.capacity_rmse_Ah * 1e3), 1), 'final_err_mAh': round(float(r.final_capacity_err_Ah * 1e3), 1), 'soc_rmse_pct': round(float(r.mean_soc_rmse * 100), 2)}
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', default='results/final'); ap.add_argument('--out', default='docs')
    a = ap.parse_args()
    bib = {}
    groups = []
    for p in sorted(glob.glob(os.path.join(ROOT, 'report/*.bib'))):
        stem = os.path.basename(p)[:-4]
        recs = parse_bib(p)
        groups.append((stem, recs))
        for r in recs:
            bib.setdefault(r['key'], r | {'file': stem})
    chapters = parse_chapters()
    res = results(os.path.join(ROOT, a.results))
    by_name = {}
    for c in chapters:
        for n in c['registered']:
            by_name[n] = c
    # attitude gated variants live in the chapter of the published algorithm
    for n in list(res):
        if n not in by_name and n.endswith('-Gated'):
            by_name[n] = by_name[n[:-6]]
    cat = []
    missing = []
    for n, r in res.items():
        c = by_name.get(n)
        if c is None:
            missing.append(n); continue
        rec = {'name': n, 'family': r['family'], 'family_label': FAMILY_LABEL.get(r['family'], r['family']), 'level': r['level'],
               'chapter_title': c['title'], 'chapter_label': c['label'], 'chapter_file': c['file'], 'headers': c['headers'],
               'state': c['state'], 'cost': c['cost'], 'estimates': c['estimates'],
               'references': [{'key': k, 'text': fmt(bib[k]), 'doi': bib[k].get('doi', '')} for k in c['refs'] if k in bib],
               'results': {k: v for k, v in r.items() if k not in ('family', 'level')}}
        cat.append(rec)
    cat.sort(key=lambda x: (FAMILY_ORDER.index(x['family']), x['results'].get('rank', 0), x['name']))
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    json.dump(cat, open(os.path.join(ROOT, a.out, 'catalogue.json'), 'w'), indent=1)
    json.dump({k: v for k, v in bib.items()}, open(os.path.join(ROOT, a.out, 'bib.json'), 'w'), indent=1)

    # CATALOGUE.md
    L = ['<!-- Copyright 2026 Sreeram Anil — SPDX-License-Identifier: CC-BY-4.0 -->',
         '# Estimator catalogue', '',
         f'{len(cat)} registered estimators: {sum(1 for c in cat if c["level"]=="cell")} cell-level, {sum(1 for c in cat if c["level"]=="pack")} pack-level, '
         f'{sum(1 for c in cat if c["level"]=="attitude")} attitude. Generated by `scripts/catalogue.py` from the report chapters, the bibliography and `results/final`; '
         'the figures of merit are the steady-state SOC RMSE in % of capacity (geometric mean over the 12 ECM scenarios / worst scenario / nominal), '
         'the pack per-cell RMSE, or the attitude RMS error in degrees, and the measured cost per step on the benchmark machine.', '']
    for famkey in FAMILY_ORDER:
        rows = [c for c in cat if c['family'] == famkey]
        if not rows: continue
        L.append(f'## {FAMILY_LABEL[famkey]} ({len(rows)})'); L.append('')
        if rows[0]['level'] == 'cell':
            L.append('| Estimator | Header | Primary reference(s) | geo-mean / worst / nominal [%] | ns/step |'); L.append('|---|---|---|---|---|')
        elif rows[0]['level'] == 'pack':
            L.append('| Estimator | Header | Primary reference(s) | per-cell RMSE nominal / weak_cell [%] | µs/step |'); L.append('|---|---|---|---|---|')
        else:
            L.append('| Estimator | Header | Primary reference(s) | RMS error nominal / gyro_bias [deg] | ns/step |'); L.append('|---|---|---|---|---|')
        for c in rows:
            hdr = ', '.join(f'[`{os.path.basename(h)}`]({h})' for h in c['headers'][:2]) or '—'
            refs = '; '.join(short_ref(bib[r['key']]) for r in c['references'][:3]) or '—'
            r = c['results']
            if c['level'] == 'cell':
                fig = f"{r['geo_mean']:.2f} / {r['worst']:.2f} / {r['nominal']:.2f}"; cost = f"{r['ns_per_step']}"
            elif c['level'] == 'pack':
                fig = f"{r['pack']['nominal']:.2f} / {r['pack']['weak_cell']:.2f}"; cost = f"{r['ns_per_step']/1000:.1f}"
            else:
                fig = f"{r['imu']['nominal']:.2f} / {r['imu']['gyro_bias']:.2f}"; cost = f"{r['ns_per_step']}"
            L.append(f"| **{c['name']}** — {c['chapter_title']} | {hdr} | {refs} | {fig} | {cost} |")
        L.append('')
    open(os.path.join(ROOT, a.out, 'CATALOGUE.md'), 'w').write('\n'.join(L))

    # REFERENCES.md
    R = ['<!-- Copyright 2026 Sreeram Anil — SPDX-License-Identifier: CC-BY-4.0 -->', '# References', '',
         f'All {len(bib)} publications cited in the report and in the source headers, grouped by the bibliography file they live in '
         '(`report/*.bib`). Every estimator in estkit is implemented from the primary references listed in its chapter and in '
         '[CATALOGUE.md](CATALOGUE.md). DOIs were transcribed by the author and should be verified before formal reuse.', '']
    n = 0
    for stem, recs in groups:
        recs = [r for r in recs if bib[r['key']]['file'] == stem]
        if not recs: continue
        R.append(f'## {BIB_GROUP.get(stem, stem)} ({len(recs)})'); R.append('')
        for r in sorted(recs, key=lambda x: (authors(x).lower(), x.get('year', ''))):
            n += 1
            R.append(f'{n}. <a id="{r["key"]}"></a>{fmt(r)}')
        R.append('')
    open(os.path.join(ROOT, a.out, 'REFERENCES.md'), 'w').write('\n'.join(R))
    print(f'catalogue: {len(cat)} estimators; references: {n}; unmatched estimators: {missing}')


def short_ref(rec):
    a = rec.get('author', rec.get('editor', ''))
    parts = [p.strip() for p in re.split(r'\s+and\s+', a) if p.strip()]
    lasts = [(p.split(',')[0] if ',' in p else p.split()[-1]).strip() for p in parts]
    if len(lasts) == 1: first = lasts[0]
    elif len(lasts) == 2: first = lasts[0] + ' & ' + lasts[1]
    else: first = lasts[0] + ' et al.'
    return f"[{first} ({rec.get('year', 'n.d.')})](REFERENCES.md#{rec['key']})"


if __name__ == '__main__':
    main()

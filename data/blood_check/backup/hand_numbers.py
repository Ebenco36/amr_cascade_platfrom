"""Numbers typed into the manuscripts on 2026-10-20 that have no \\runnum key.
Recomputes each from the final run's combined E. coli edge report and checks that
the text still prints it. Run from the project root:
    .venv/bin/python manuscript_edit_log/2026-10-20/edges/hand_numbers.py [--run-root export_archives/hpc_final]
Exit 1 if any value differs or is missing from the text."""
import argparse, pathlib, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument('--run-root', default='export_archives/hpc_final'); a = ap.parse_args()
e = pd.read_parquet(pathlib.Path(a.run_root) / 'data/artifacts/cascade/combined/organisms/escherichia_coli/edge_report.parquet')
v = e[e.validation_status.isin(['robust', 'supported'])].copy()
fold = np.exp(np.abs(np.log(v.escalation_ratio)))
def pair(up, down):
    r = e[(e.upstream_antibiotic == up) & (e.downstream_antibiotic == down)]
    assert len(r) == 1, (up, down); return r.iloc[0]
def pct(x): return f'{100 * x:.1f}\\%'
values = {
  'validated patterns at least 1.5-fold from ER 1': str(int((fold >= 1.5).sum())),
  '  of which escalation': str(int(((fold >= 1.5) & (v.cascade_direction == 'escalation')).sum())),
  '  of which suppression': str(int(((fold >= 1.5) & (v.cascade_direction == 'suppression')).sum())),
  'validated patterns at least 2-fold from ER 1': str(int((fold >= 2).sum())),
}
examples = [('GENTAMICIN', 'AMIKACIN', 2), ('CEFTRIAXON', 'CEFTAZIDIM', 2), ('CEFEPIM', 'IMIPENEM', 2), ('CEFTRIAXON', 'FOSFOMYCIN', 1), ('AZTREONAM', 'PIPERACILLIN/TAZOBACTAM', 2)]
for up, down, d in examples:
    r = pair(up, down)
    tag = f'{up} -> {down}'
    values[f'{tag}: label'] = r.validation_status
    values[f'{tag}: p_R'] = pct(r.resistant_downstream_test_probability)
    values[f'{tag}: p_S'] = pct(r.susceptible_downstream_test_probability)
    values[f'{tag}: ER'] = f'{r.escalation_ratio:.{d}f}'
    values[f'{tag}: fold'] = f'{float(np.exp(abs(np.log(r.escalation_ratio)))):.2f}'
    values[f'{tag}: sites agreeing'] = f'{r.site_direction_agreement_rate:.3f} of {int(r.site_replication_n)}'
for k, val in values.items(): print(f'{k:60s} {val}')
# Strings that must appear in the text (file, string)
TEXT = {
 'paper1_detection_validation/results_only.tex': ['\\runnum{validation.validated} validated patterns, ' + values['validated patterns at least 1.5-fold from ER 1'] + ' differed from ER $=1$ by at least 1.5-fold (ER $\\ge1.5$ or $\\le0.67$; ' + values['  of which escalation'] + ' escalation- and ' + values['  of which suppression'] + ' suppression-direction patterns) and ' + values['validated patterns at least 2-fold from ER 1'] + ' by at least 2-fold'],
 'paper1_detection_validation/discussion_only.tex': [
   values['GENTAMICIN -> AMIKACIN: p_R'] + ' of gentamicin-resistant episodes but ' + values['GENTAMICIN -> AMIKACIN: p_S'],
   '(ER ' + values['GENTAMICIN -> AMIKACIN: ER'] + ', with all three sites agreeing in direction)',
   values['CEFTRIAXON -> CEFTAZIDIM: p_R'] + ' versus ' + values['CEFTRIAXON -> CEFTAZIDIM: p_S'],
   values['CEFEPIM -> IMIPENEM: p_R'] + ' versus ' + values['CEFEPIM -> IMIPENEM: p_S'],
   values['CEFTRIAXON -> FOSFOMYCIN: p_R'] + ' of ceftriaxone-resistant and ' + values['CEFTRIAXON -> FOSFOMYCIN: p_S'],
   values['AZTREONAM -> PIPERACILLIN/TAZOBACTAM: p_R'] + ' of aztreonam-resistant but ' + values['AZTREONAM -> PIPERACILLIN/TAZOBACTAM: p_S'],
 ],
}
# --- added 2026-10-20 (agreement by strength, concentration, within-site check) ---
s11 = pd.read_csv(pathlib.Path(a.run_root) / 'outputs/tables/combined/organisms/escherichia_coli/supplementary/table_s11_cross_site_replication.csv')
s11['strong'] = np.exp(np.abs(np.log(s11.combined_escalation_ratio))) >= 1.5
def allag(meas, direction):
    m = s11[(s11[f'{meas}_sites_with_estimate'] >= 2) & s11.strong & s11.cascade_direction.eq(direction)]
    k = int(m[f'{meas}_sites_agreeing'].eq(m[f'{meas}_sites_with_estimate']).sum())
    return f'{k} of {len(m)} ({100 * k / len(m):.1f}\\%)'
ss = s11[(s11.cascade_direction == 'suppression') & s11.strong & (s11.independent_sites_with_estimate >= 2)]
def agrees(col): return ss[col].lt(1)
only = ss[ss.independent_sites_agreeing.eq(1)]
strong_esc = v[(fold >= 1.5) & v.cascade_direction.eq('escalation')]
top = strong_esc.downstream_antibiotic.value_counts()
ws = pd.read_csv(pathlib.Path(__file__).resolve().parents[1] / 'within_site/within_site_check.csv')
wsf = ws.fold >= 1.5
TEXT['paper1_detection_validation/results_only.tex'] += [
  f"for {allag('pooled', 'escalation')} under the pooled measure and {allag('independent', 'escalation')} under the independent measure, compared with {allag('pooled', 'suppression').split(' (')[0]} ({allag('pooled', 'suppression').split(' (')[1]} and {allag('independent', 'suppression').split(' (')[0]} ({allag('independent', 'suppression').split(' (')[1]} for suppression",
  f"ECU Health shared the combined direction for all {len(ss)} of these suppression-direction patterns, Stanford for {int(agrees('er_armd').sum())} and UT Southwestern for {int(agrees('er_armd_utsw').sum())} of {int(ss.er_armd_utsw.notna().sum())}, and for {len(only)} of them ECU Health was the only such site",
  f"fosfomycin ({top['FOSFOMYCIN']}), cefotetan ({top['CEFOTETAN']}), piperacillin ({top['PIPERACILLIN']}) or doxycycline ({top['DOXYCYCLIN']})",
  f"{top[['FOSFOMYCIN', 'CEFOTETAN', 'PIPERACILLIN', 'DOXYCYCLIN']].sum()} of the {len(strong_esc)} escalation-direction patterns".replace('55 of', 'Fifty-five of'),
  f"the ratio stratified by site and era kept that of {int(ws.mh_site_era_same_direction.sum())}",
  f"{int((wsf & ws.cascade_direction.eq('escalation') & ws.mh_site_era_same_direction).sum())} of the {int((wsf & ws.cascade_direction.eq('escalation')).sum())} escalation-direction patterns at least 1.5-fold from ER $=1$ kept their direction and {int((wsf & ws.cascade_direction.eq('escalation') & (ws.mh_site_era_fold >= 1.5)).sum())} remained",
  f"{int((wsf & ws.cascade_direction.eq('suppression') & ws.mh_site_era_same_direction).sum())} kept their direction and {int((wsf & ws.cascade_direction.eq('suppression') & (ws.mh_site_era_fold >= 1.5)).sum())} remained",
]
assert ss.er_armd_ecuh.lt(1).all() and only.er_armd_ecuh.lt(1).all() and ws.mh_site_same_direction.all()
TEXT['paper1_detection_validation/discussion_only.tex'] += [f"downstream drug in {top['FOSFOMYCIN']} strong escalation-direction patterns"]
# --- added 2026-10-20: paper 2 threshold classification ---
PS = pathlib.Path(a.run_root) / 'data/artifacts/prevalence_shift/combined/organisms/escherichia_coli'
tp = pd.read_parquet(PS / 'prevalence_mnar_tipping_points.parquet'); sh = pd.read_parquet(PS / 'prevalence_shift.parquet').set_index('drug')
th = {}
for cut in (10, 20, 30):
    x = tp[tp.decision_threshold.mul(100).round().eq(cut)].set_index('drug')
    th[cut] = dict(below=sh.index[sh.prevalence_upper_bound_pct < cut].tolist(), above=sh.index[sh.prevalence_lower_bound_pct >= cut].tolist(),
                   across=int(((sh.prevalence_lower_bound_pct < cut) & (sh.prevalence_upper_bound_pct >= cut)).sum()),
                   cross=sorted(x.index[x.crossing_status.eq('crosses_threshold')]), oneside=int(x.crossing_status.isin(['always_below_threshold', 'always_above_threshold']).sum()),
                   opposite=x.index[(x.naive_prevalence_pct >= cut) != (x.prevalence_at_lambda0_pct >= cut)].tolist())
    print(f'paper 2, {cut}% threshold:', th[cut])
t20 = th[20]
assert sorted(t20['below']) == ['ERTAPENEM', 'PIPERACILLIN/TAZOBACTAM'] and t20['above'] == ['CIPROFLOXACIN'] and t20['opposite'] == ['MOXIFLOXACIN']
assert t20['cross'] == ['AZTREONAM', 'CEFALOTIN', 'CEFOTETAN', 'CEFOXITIN', 'LEVOFLOXACIN', 'MINOCYCLIN', 'MOXIFLOXACIN']
mox = sh.loc['MOXIFLOXACIN']; mx = (f'{mox.naive_prevalence_pct:.1f}\\%', f'{mox.mnar_lambda0_prevalence_pct:.1f}\\%')
n3 = len(t20['below']) + len(t20['above'])
P2 = 'paper2_prevalence_surveillance/'; JV = P2 + 'journal_versions/'
TEXT[P2 + 'results_only.tex'] = [f"across it for {t20['across']}", f"for {len(t20['cross'])} antibiotics (aztreonam", f"one side of it for {t20['oneside']}",
    f"naive prevalence ({mx[0]}) and the no-tilt estimate ({mx[1]})",
    f"the bounds placed no antibiotic below and {len(th[10]['above'])} above it, and the curve crossed it for {len(th[10]['cross'])}; at 30\\%, the bounds placed {len(th[30]['below'])} below and none above, and the curve crossed it for {len(th[30]['cross'])}"]
assert not th[10]['below'] and not th[30]['above']
TEXT[P2 + 'abstract_body.tex'] = [f"placed only {n3} of these antibiotics", f"for {len(t20['cross'])}, and for moxifloxacin naive prevalence was {mx[0]} but the no-tilt estimate {mx[1]}"]
TEXT[P2 + 'discussion_only.tex'] = [f"placed only {n3} of the", f"For {len(t20['cross'])} antibiotics, whether", f"no-tilt estimate ({mx[1]})", f"naive prevalence ({mx[0]})"]
TEXT[JV + 'CMI/abstract_only.tex'] = [f"placed only {n3} antibiotics", f"For {len(t20['cross'])} antibiotics, exceeding 20\\%", ]  # 2026-11-08: the moxifloxacin clause gave way to the masking result in the CMI abstract
TEXT[JV + 'CMI/results_only.tex'] = [f"across it for the other {t20['across']} antibiotics", f"crossed 20\\% for {len(t20['cross'])} (aztreonam", f"tested-only prevalence ({mx[0]}) and the no-tilt estimate ({mx[1]})", f"placed none below and {len(th[10]['above'])} above, and estimates crossed it for {len(th[10]['cross'])}"]
TEXT[JV + 'CMI/discussion_only.tex'] = [f"placed only {n3} of the", f"assumed tilt for {len(t20['cross'])} antibiotics"]
TEXT[JV + 'Eurosurveillance/abstract_only.tex'] = [f"placing only {n3} antibiotics", f"assumed MNAR tilt for {len(t20['cross'])} antibiotics"]
TEXT[JV + 'Eurosurveillance/results_only.tex'] = [f"across it for the other {t20['across']} antibiotics", f"crossed 20\\% for {len(t20['cross'])} antibiotics", f"tested-only prevalence ({mx[0]}) and the no-tilt estimate ({mx[1]})",
    f"placed no antibiotic below and {len(th[10]['above'])} above the threshold, and the estimate crossed it for {len(th[10]['cross'])}; at 30\\%, the bounds placed {len(th[30]['below'])} below and none above, and the estimate crossed it for {len(th[30]['cross'])}"]
TEXT[JV + 'Eurosurveillance/discussion_only.tex'] = [f"placed only {n3} of the", f"assumed tilt for {len(t20['cross'])} antibiotics"]
TEXT[JV + 'Eurosurveillance/key_public_health_message_only.tex'] = [f"For {t20['across']} of the"]
# --- added 2026-10-21: paper 1 abstract ---
TEXT['paper1_detection_validation/abstract_body.tex'] = [
  f"and {values['validated patterns at least 1.5-fold from ER 1']} differed at least 1.5-fold",
  f"amikacin was observed in {values['GENTAMICIN -> AMIKACIN: p_R']} of gentamicin-resistant and {values['GENTAMICIN -> AMIKACIN: p_S']} of gentamicin-susceptible episodes"]
# --- added 2026-10-21: support values (replaces the earlier example strings) ---
def c(x): return f'{x:,.0f}'.replace(',', '{,}')
def ex(up, down):
    r = pair(up, down); kR = round(r.resistant_downstream_test_probability * r.resistant_support_n); kS = round(r.susceptible_downstream_test_probability * r.susceptible_support_n)
    return r, int(kR), int(kS)
sup = v.total_support_n; rs = v.resistant_support_n; ss_ = v.susceptible_support_n
TEXT['paper1_detection_validation/results_only.tex'] += [
  f"median of {c(sup.median())} episode-pair rows (interquartile range {c(sup.quantile(.25))}--{c(sup.quantile(.75))}), with medians of {c(rs.median())} episodes in the resistant-upstream branch and {c(ss_.median())} in the susceptible-upstream branch; {int((rs < 100).sum())} had fewer than 100"]
for d, lab in (('escalation', 'the largest from '), ('suppression', '(')):
    x = v[v.cascade_direction == d]; r = x.loc[x.escalation_ratio.idxmax() if d == 'escalation' else x.escalation_ratio.idxmin()]
    _, kR, kS = ex(r.upstream_antibiotic, r.downstream_antibiotic)
    TEXT['paper1_detection_validation/results_only.tex'].append(f"{lab}{kR} of {c(r.resistant_support_n)} resistant-upstream and {c(kS)} of {c(r.susceptible_support_n)} susceptible-upstream episodes")
g, gkR, gkS = ex('GENTAMICIN', 'AMIKACIN'); ct, _, _ = ex('CEFTRIAXON', 'CEFTAZIDIM'); ci, _, _ = ex('CEFEPIM', 'IMIPENEM')
fo, fkR, fkS = ex('CEFTRIAXON', 'FOSFOMYCIN'); az, _, _ = ex('AZTREONAM', 'PIPERACILLIN/TAZOBACTAM')
TEXT['paper1_detection_validation/discussion_only.tex'] = [
  f"{pct(g.resistant_downstream_test_probability)} of {c(g.resistant_support_n)} gentamicin-resistant episodes but {pct(g.susceptible_downstream_test_probability)} of {c(g.susceptible_support_n)} gentamicin-susceptible ones",
  '(ER ' + values['GENTAMICIN -> AMIKACIN: ER'] + ', with all three sites agreeing in direction)',
  f"({pct(ct.resistant_downstream_test_probability)} of {c(ct.resistant_support_n)} ceftriaxone-resistant versus {pct(ct.susceptible_downstream_test_probability)} of {c(ct.susceptible_support_n)} ceftriaxone-susceptible episodes)",
  f"({pct(ci.resistant_downstream_test_probability)} of {c(ci.resistant_support_n)} cefepime-resistant versus {pct(ci.susceptible_downstream_test_probability)} of {c(ci.susceptible_support_n)} cefepime-susceptible episodes)",
  f"{fkR} of {c(fo.resistant_support_n)} ceftriaxone-resistant ({pct(fo.resistant_downstream_test_probability)}) and {fkS} of {c(fo.susceptible_support_n)} ceftriaxone-susceptible episodes ({pct(fo.susceptible_downstream_test_probability)})",
  f"{pct(az.resistant_downstream_test_probability)} of {c(az.resistant_support_n)} aztreonam-resistant but {pct(az.susceptible_downstream_test_probability)} of {c(az.susceptible_support_n)} aztreonam-susceptible episodes",
  f"downstream drug in {top['FOSFOMYCIN']} strong escalation-direction patterns"]
TEXT['paper1_detection_validation/abstract_body.tex'] = [
  f"and {values['validated patterns at least 1.5-fold from ER 1']} differed at least 1.5-fold",
  f"amikacin was observed in {pct(g.resistant_downstream_test_probability)} of {c(g.resistant_support_n)} gentamicin-resistant and {pct(g.susceptible_downstream_test_probability)} of {c(g.susceptible_support_n)} gentamicin-susceptible episodes"]
# --- added 2026-10-21 (later): abstract rebuilt around ceftriaxone->fosfomycin and aztreonam->piperacillin/tazobactam ---
def p2(x): return f'{100 * x:.2f}\\%'
fo_ratio = fo.escalation_ratio  # the paper's continuity-corrected ER (59.2); the raw ratio is 59.5
TEXT['paper1_detection_validation/abstract_body.tex'] = [
  f"; {values['validated patterns at least 1.5-fold from ER 1']} differed at least 1.5-fold, in both directions",
  f"Fosfomycin was observed {fo_ratio:.0f} times as often in ceftriaxone-resistant as in ceftriaxone-susceptible episodes ({pct(fo.resistant_downstream_test_probability)} versus {p2(fo.susceptible_downstream_test_probability)})",
  f"observed in {pct(az.susceptible_downstream_test_probability)} of aztreonam-susceptible episodes, was missing for a third of aztreonam-resistant ones"]
assert 0.30 < 1 - az.resistant_downstream_test_probability < 0.37, 'a third'
assert (fold >= 1.5).sum() and ((fold >= 1.5) & v.cascade_direction.eq('suppression')).sum()
TEXT['paper1_detection_validation/results_only.tex'].append(
  # 2026-11-08: the two examples named and split into two sentences (Results-only pass)
  f"fosfomycin was observed in {fkR} of {c(fo.resistant_support_n)} ceftriaxone-resistant ({pct(fo.resistant_downstream_test_probability)}) and {fkS} of {c(fo.susceptible_support_n)} ceftriaxone-susceptible episodes ({p2(fo.susceptible_downstream_test_probability)}; ER {fo.escalation_ratio:.1f})")
TEXT['paper1_detection_validation/results_only.tex'].append(
  f"piperacillin--tazobactam was observed in {c(round(az.resistant_downstream_test_probability * az.resistant_support_n))} of {c(az.resistant_support_n)} aztreonam-resistant ({pct(az.resistant_downstream_test_probability)}) and {c(round(az.susceptible_downstream_test_probability * az.susceptible_support_n))} of {c(az.susceptible_support_n)} aztreonam-susceptible episodes ({pct(az.susceptible_downstream_test_probability)}; ER {az.escalation_ratio:.2f})")
wsx = ws.set_index(['upstream_antibiotic', 'downstream_antibiotic'])
assert wsx.loc[('CEFTRIAXON', 'FOSFOMYCIN'), 'mh_site_era'] > 1 and wsx.loc[('AZTREONAM', 'PIPERACILLIN/TAZOBACTAM'), 'mh_site_era'] < 1
assert fo.validation_status == 'robust' and az.validation_status == 'robust'
TEXT['paper1_detection_validation/discussion_only.tex'] = [x.replace(f"({pct(fo.susceptible_downstream_test_probability)})", f"({p2(fo.susceptible_downstream_test_probability)})") for x in TEXT['paper1_detection_validation/discussion_only.tex']]
# --- added 2026-10-22: Discussion rewritten from the author's draft (wording of the example sentences changed) ---
TEXT['paper1_detection_validation/discussion_only.tex'] = [
  f"Amikacin was observed in {pct(g.resistant_downstream_test_probability)} of {c(g.resistant_support_n)} gentamicin-resistant episodes but {pct(g.susceptible_downstream_test_probability)} of {c(g.susceptible_support_n)} gentamicin-susceptible episodes (ER {g.escalation_ratio:.2f}), with all three sites agreeing in direction",
  f"({pct(ct.resistant_downstream_test_probability)} of {c(ct.resistant_support_n)} ceftriaxone-resistant versus {pct(ct.susceptible_downstream_test_probability)} of {c(ct.susceptible_support_n)} ceftriaxone-susceptible episodes)",
  f"({pct(ci.resistant_downstream_test_probability)} of {c(ci.resistant_support_n)} cefepime-resistant versus {pct(ci.susceptible_downstream_test_probability)} of {c(ci.susceptible_support_n)} cefepime-susceptible episodes)",
  f"observed in {fkR} of {c(fo.resistant_support_n)} ceftriaxone-resistant episodes ({pct(fo.resistant_downstream_test_probability)}) but only {fkS} of {c(fo.susceptible_support_n)} ceftriaxone-susceptible episodes ({p2(fo.susceptible_downstream_test_probability)})",
  f"{pct(az.resistant_downstream_test_probability)} of {c(az.resistant_support_n)} aztreonam-resistant but {pct(az.susceptible_downstream_test_probability)} of {c(az.susceptible_support_n)} aztreonam-susceptible episodes",
  f"downstream antibiotic in {top['FOSFOMYCIN']} strong escalation-direction patterns, which need not represent {top['FOSFOMYCIN']} distinct"]
_s11 = s11.set_index(['upstream_antibiotic', 'downstream_antibiotic'])
assert all(pair(u, d).validation_status == 'robust' for u, d in (('GENTAMICIN', 'AMIKACIN'), ('CEFTRIAXON', 'CEFTAZIDIM'), ('CEFEPIM', 'IMIPENEM'), ('CEFTRIAXON', 'FOSFOMYCIN'), ('AZTREONAM', 'PIPERACILLIN/TAZOBACTAM'))), 'all five robust'
assert _s11.loc[('GENTAMICIN', 'AMIKACIN'), 'pooled_sites_agreeing'] == 3 == _s11.loc[('GENTAMICIN', 'AMIKACIN'), 'independent_sites_agreeing'], 'all three sites agree'
# --- added 2026-10-27: adjusted-OR extremes (Results) and the modest-magnitude count (Discussion); these depend on
# the adjusted models, so the covariate re-run may change them
_orr = pd.to_numeric(v.adjusted_odds_ratio, errors='coerce'); _est = v[_orr.notna()]; _o = _orr[_orr.notna()]
_big = _est[(_o >= 10) | (_o <= 0.1)]
_ext = ((_big.resistant_downstream_test_probability >= 0.95) | (_big.resistant_downstream_test_probability <= 0.05)
        | (_big.susceptible_downstream_test_probability >= 0.95) | (_big.susceptible_downstream_test_probability <= 0.05))
_cx = pair('CEFOTAXIM', 'CEFTAZIDIM')
TEXT['paper1_detection_validation/results_only.tex'] += [
  f"Of the {len(_big)} adjusted ORs at least 10-fold from 1, {int(_ext.sum())} belonged to patterns",
  f"observed in {pct(_cx.resistant_downstream_test_probability)} of cefotaxime-resistant and {pct(_cx.susceptible_downstream_test_probability)} of cefotaxime-susceptible episodes (ER {_cx.escalation_ratio:.2f}, adjusted OR {_cx.adjusted_odds_ratio:.1f})"]
TEXT['paper1_detection_validation/discussion_only.tex'] += [f"{len(v) - int((fold >= 1.5).sum())} of the {len(v)} differed from ER $=1$ by less than 1.5-fold"]
# --- added 2026-10-27 (robust label): how the robust patterns met the replication criterion, recomputed from the
# validation results with the classifier's own rule (cascade_validation_analyzer._classify_validation; thresholds from
# configs/base/cascade.yaml). Validation is not touched by the covariate re-run, so these do not move.
_vr = pd.read_parquet(pathlib.Path(a.run_root) / 'data/artifacts/cascade/combined/organisms/escherichia_coli/validation_results.parquet')
def _site_rep(r):
    if pd.notna(r.site_i_squared) and pd.notna(r.site_pooled_se) and r.site_pooled_se > 0 and pd.notna(r.site_pooled_log_er):
        same = (r.observed_escalation_ratio > 1 and r.site_pooled_log_er > 0) or (r.observed_escalation_ratio < 1 and r.site_pooled_log_er < 0)
        return r.site_replication_n >= 2 and r.site_i_squared <= 50.0 and abs(r.site_pooled_log_er / r.site_pooled_se) >= 1.96 and same
    return r.site_replication_n >= 2 and pd.notna(r.site_direction_agreement_rate) and r.site_direction_agreement_rate >= 0.67
_rb = _vr[_vr.validation_status == 'robust']
_s = _rb.apply(_site_rep, axis=1).astype(bool); _tm = _rb.temporal_direction_agreement.fillna(False).astype(bool)
assert int((~_s & ~_tm).sum()) == 0  # every robust pattern met one of the two routes
TEXT['paper1_detection_validation/results_only.tex'] += [
  f"Of the \\texttt{{robust}} patterns, {int((_tm & ~_s).sum())} met the replication criterion through temporal agreement alone, {int((_s & ~_tm).sum())} through site-level consistency alone and {int((_s & _tm).sum())} through both."]
# --- added 2026-10-21: ESCMID Global 2027 abstracts (LaTeX, conference_abstracts/ESCMID_Global_2027/) ---
import json
grid = json.loads((PS / 'prevalence_shift_summary.json').read_text())['mnar_lambda_grid']
ESC = 'conference_abstracts/ESCMID_Global_2027/'
TEXT[ESC + 'paper1_abstract.tex'] = [
  # 2026-10-29: clearer abstract; retain checks derived from the underlying counts.
  f"Fosfomycin was observed in {pct(fo.resistant_downstream_test_probability)} of ceftriaxone-resistant versus {p2(fo.susceptible_downstream_test_probability)} of ceftriaxone-susceptible episodes",
  f"piperacillin--tazobactam, observed in {pct(az.resistant_downstream_test_probability)} of aztreonam-resistant versus {pct(az.susceptible_downstream_test_probability)} of aztreonam-susceptible episodes",
  f"After site-and-era stratification, {int(ws.mh_site_era_same_direction.sum())}/{len(ws)} retained their direction",
  f"Within this eligible pair population, ceftriaxone-resistant episodes constituted {100 * fo.resistant_support_n / (fo.resistant_support_n + fo.susceptible_support_n):.1f}\\% but supplied "
  f"{100 * fo.resistant_tested_n / (fo.resistant_tested_n + fo.susceptible_tested_n):.1f}\\% of observed fosfomycin results",
  ]  # the held-out percentages are checked in held_out_site/check_numbers.py
# "all" kept direction within sites
assert bool(ws.mh_site_same_direction.all())
# "strong suppression-direction patterns were largely ECU Health-specific": ECU Health shares the direction of all, the other two sites of fewer than half
assert int(agrees('er_armd').sum()) < len(ss) / 2 and int(agrees('er_armd_utsw').sum()) < ss.er_armd_utsw.notna().sum() / 2
_WORDS = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
_tet = pd.read_parquet(PS / 'prevalence_shift.parquet').set_index('drug').loc['TETRACYCLIN']
TEXT[ESC + 'paper2_abstract.tex'] = [
  f"over \\({min(grid):g}\\) to \\({max(grid):g}\\)",
  f"Bounds were wholly below 20\\% for ertapenem and piperacillin--tazobactam, wholly above for ciprofloxacin, and crossed it for {t20['across']} antibiotics",
  f"resolved threshold classification for only {len(t20['below']) + len(t20['above'])}/{len(sh)} antibiotics",
  f"Sensitivity curves crossed 20\\% for {_WORDS.get(len(t20['cross']), len(t20['cross']))} antibiotics.",
  f"For tetracycline, episodes with an observed validated resistance trigger supplied {100 * _tet.cascade_trigger_fraction:.1f}\\% of observed results and had "
  f"{_tet.cascade_prevalence_pct:.1f}\\% resistance versus {_tet.independent_prevalence_pct:.1f}\\% in other observed episodes, indicating that tested-only prevalence ({_tet.naive_prevalence_pct:.1f}\\%) depends on which episodes are tested"]
_fos = sh.loc['FOSFOMYCIN']
_fdiag = pd.read_csv(pathlib.Path(a.run_root) / 'outputs/tables/combined/organisms/escherichia_coli/table_k_prevalence_shift_diagnostics.csv')
_fdiag = _fdiag[_fdiag.Drug.astype(str).str.upper().eq('FOSFOMYCIN')].iloc[0]
TEXT[ESC + 'paper2_abstract.tex'] += [
  f"Fosfomycin, observed in {100 * _fdiag['Observed n'] / _fdiag['Eligible n']:.2f}\\% of eligible episodes, had tested-only resistance of {_fos.naive_prevalence_pct:.1f}\\% "
  f"but bounds of {_fos.prevalence_lower_bound_pct:.2f}--{_fos.prevalence_upper_bound_pct:.1f}\\%"]
# "the association persisted within site and era": ceftriaxone -> fosfomycin keeps its direction and stays at least 1.5-fold
_cf = ws[ws.upstream_antibiotic.eq('CEFTRIAXON') & ws.downstream_antibiotic.eq('FOSFOMYCIN')].iloc[0]
assert bool(_cf.mh_site_era_same_direction) and _cf.mh_site_era_fold >= 1.5
# 2026-10-27: the author's wording of both ESCMID abstracts (backup of the earlier strings in 2026-10-27/escmid/);
# the names in brackets must be the antibiotics the counts come from
assert sorted(t20['below']) == ['ERTAPENEM', 'PIPERACILLIN/TAZOBACTAM'] and t20['above'] == ['CIPROFLOXACIN']
# --- added 2026-10-27 (Supplement, ridge-penalty note): "changes of direction occur mainly where the adjusted odds
# ratio is close to 1". Read here as: more than half of the patterns whose adjusted OR changes side of 1 across
# lambda in {1, 10, 100} have an OR within 0.8-1.25 at lambda = 10. Depends on the adjusted models (covariate re-run).
import numpy as np
_rp = pd.read_parquet(pathlib.Path(a.run_root) / 'data/artifacts/cascade/combined/organisms/escherichia_coli/ridge_penalty_sensitivity.parquet')
_w = _rp.pivot_table(index=['upstream_antibiotic', 'downstream_antibiotic'], columns='ridge_penalty', values='adjusted_odds_ratio').dropna()
_flip = _w[np.sign(np.log(_w)).nunique(axis=1) > 1]
_near = int(_flip[10.0].between(0.8, 1.25).sum())
print(f'ridge note: {len(_flip)} direction changes, {_near} with OR within 0.8-1.25 at lambda 10')
RIDGE_CLAIM_OK = len(_flip) == 0 or _near > len(_flip) / 2
# --- added 2026-10-28: ESKAPE species sentence (paper 1 Results) and masking wording (paper 2 Discussions)
_sp = pd.read_csv(pathlib.Path('manuscript_edit_log/2026-10-21/eskape_section/eskape_species_summary.csv'))
_sp = _sp[_sp.scope.eq('combined') & _sp.analysed].set_index('organism')
_top = _sp.validated_edges.sort_values(ascending=False)
assert list(_top.index[:3]) == ['KLEBSIELLA PNEUMONIAE', 'PSEUDOMONAS AERUGINOSA', 'STAPHYLOCOCCUS AUREUS']
_all3 = _sp[_sp.sites.eq('3 of 3')]
_none = _sp[_sp.validated_edges.eq(0)]
assert set(_none.index) == {'ENTEROBACTER', 'ENTEROBACTER HORMAECHEI', 'ENTEROBACTER GERGOVIAE'}
TEXT['paper1_detection_validation/results_only.tex'] += [
  f"\\textit{{Klebsiella pneumoniae}} ({int(_top.iloc[0])} validated patterns), \\textit{{Pseudomonas aeruginosa}} ({int(_top.iloc[1])}) and \\textit{{Staphylococcus aureus}} ({int(_top.iloc[2])}) contributed most",
  f"these and {len(_all3) - 3} further labels had validated patterns in the own analysis of each of the three sites".replace("5 further", "five further"),
  f"({int(_none.culture_episode_rows['ENTEROBACTER'])}, {int(_none.culture_episode_rows['ENTEROBACTER HORMAECHEI'])} and {int(_none.culture_episode_rows['ENTEROBACTER GERGOVIAE'])} culture episodes)"]
assert set(_all3.index) >= set(_top.index[:3])
_mk = pd.read_csv(pathlib.Path(a.run_root) / 'outputs/tables/combined/organisms/escherichia_coli/masking_experiment/masking_results.csv')
_tr = _mk[_mk.mechanism.eq('transplanted')]
_il = _tr.implied_lambda.dropna()
MASKING_WORDING_OK = (_il.gt(0).sum() > len(_il) / 2) and (_tr.mnar_lambda0_error_pp.gt(0).sum() > len(_tr) / 2)
print(f"masking wording: implied lambda > 0 in {int(_il.gt(0).sum())} of {len(_il)} ('mostly positive'); no-tilt above truth in {int(_tr.mnar_lambda0_error_pp.gt(0).sum())} of {len(_tr)} ('usually too high')")
# --- added 2026-10-28 (year-by-year course): wording claims without a \runnum key. Read from the run's
# temporal_regimes tables (scripts/run_temporal_regimes.py).
_T = pathlib.Path(a.run_root) / 'outputs/tables/combined/organisms/escherichia_coli/temporal_regimes'
_tn = json.loads((_T / 'temporal_regime_numbers.json').read_text())
_ty = pd.read_csv(_T / 'temporal_regime_years.csv')
_st = _tn['strong']
_half = _st['switching'] / _st['site_series_followed']
_cl = {s: _st['change_years_by_site'][s]['changes_within_1_of_mode'] / _st['change_years_by_site'][s]['changes'] for s in ('armd', 'armd_ecuh', 'armd_utsw')}
_weak_never = _tn['weak']['never'] / _tn['weak']['site_series_followed']
_after = []
for _site in ('armd', 'armd_ecuh'):
    _last = int(_tn['examples']['gentamicin_amikacin'][_site]['last_present_year'])
    _g = _ty[_ty.upstream_antibiotic.eq('GENTAMICIN') & _ty.downstream_antibiotic.eq('AMIKACIN') & _ty.source_site.eq(_site) & _ty.evaluable & _ty.year.gt(_last)]
    _after.append(float(min(_g.pR.min(), _g.pS.min())) if len(_g) else float('nan'))
print(f"year-by-year: switching share {_half:.2f} ('about half'); clustered within 1 year {_cl} ('Stanford and ECU Health clustered'); "
      f"weak never {_weak_never:.2f} ('mostly never present'); amikacin after the change, lowest share {_after} ('nearly all')")
TEMPORAL_WORDING_OK = (0.4 <= _half <= 0.6 and _cl['armd'] >= 0.7 and _cl['armd_ecuh'] >= 0.7 and _cl['armd_utsw'] < min(_cl['armd'], _cl['armd_ecuh'])
                       and _weak_never > 0.5 and all(x >= 0.95 for x in _after))
TEXT['paper1_detection_validation/discussion_only.tex'] += [
  "about half of the strong patterns' site series were present in some years and absent or reversed in others. At Stanford and ECU Health the changes clustered around a single calendar year"]
TEXT['paper1_detection_validation/results_only.tex'] += ["in later years, amikacin was observed in nearly all episodes at these two sites whatever the gentamicin result"]
TEXT['paper1_detection_validation/supplementary.tex'] = ["were mostly classified as never present"]
# --- added 2026-10-28 (evening): paper 1 Discussion, strong suppression-direction patterns within site and era
_st = ws[ws.fold >= 1.5].copy(); _st['kept'] = np.log(_st.mh_site_era_fold) / np.log(_st.fold)
_k = _st.groupby('cascade_direction').kept.median()
_n = lambda d: (int((_st.cascade_direction.eq(d) & _st.mh_site_era_same_direction & (_st.mh_site_era_fold >= 1.5)).sum()), int(_st.cascade_direction.eq(d).sum()))
print(f"within site and era: median share of the log ER kept, escalation {_k['escalation']:.2f}, suppression {_k['suppression']:.2f} ('most of their strength reflects differences between periods')")
SUPPRESSION_CLAIM_OK = _k['suppression'] < 0.5 <= _k['escalation']
TEXT['paper1_detection_validation/discussion_only.tex'] += [
  f"within site alone, {int((_st.cascade_direction.eq('suppression') & _st.mh_site_same_direction & (_st.mh_site_fold >= 1.5)).sum())} of {_n('suppression')[1]} strong suppression-direction patterns remained at least 1.5-fold from ER $=1$, "
  f"but within site and era only {_n('suppression')[0]} did, against {_n('escalation')[0]} of {_n('escalation')[1]} escalation-direction patterns"]
# --- added 2026-10-28 (final pass): new typed results in paper 1 and paper 2 (base) ---
import sys as _sys; _sys.path.insert(0, 'src')
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver
_res = AntibioticClassificationResolver(pathlib.Path('data/antibiotic_classification_complete.csv'))
_rank = {'Access': 1, 'Watch': 2, 'Reserve': 3}
_vr = pd.read_parquet(pathlib.Path(a.run_root) / 'data/artifacts/cascade/combined/organisms/escherichia_coli/validation_results.parquet')
_vr['step'] = np.sign(_vr.downstream_antibiotic.map(lambda x: _rank[_res.resolve(x).aware_category]) - _vr.upstream_antibiotic.map(lambda x: _rank[_res.resolve(x).aware_category]))
_vr['esc'] = _vr.observed_escalation_ratio > 1
_val = _vr[_vr.validation_status.isin(['robust', 'supported'])]
def _ud(frame, esc): g = frame[frame.esc.eq(esc)]; return int(g.step.eq(1).sum()), int(g.step.eq(-1).sum())
(_eu, _ed), (_su, _sd), (_ceu, _ced), (_csu, _csd) = _ud(_val, True), _ud(_val, False), _ud(_vr, True), _ud(_vr, False)
_fos_pair = fo; _cf2 = ws[ws.upstream_antibiotic.eq('CEFTRIAXON') & ws.downstream_antibiotic.eq('FOSFOMYCIN')].iloc[0]
_share_R = 100 * _fos_pair.resistant_support_n / (_fos_pair.resistant_support_n + _fos_pair.susceptible_support_n)
_share_obs = 100 * _fos_pair.resistant_tested_n / (_fos_pair.resistant_tested_n + _fos_pair.susceptible_tested_n)
TEXT['paper1_detection_validation/abstract_body.tex'] = [
  f"120, in both directions, differed at least 1.5-fold".replace('120', values['validated patterns at least 1.5-fold from ER 1']),
  f"Ceftriaxone-resistant episodes, {_share_R:.1f}\\% of those eligible, supplied {_share_obs:.1f}\\% of fosfomycin results",
  f"piperacillin--tazobactam was missing for a third of aztreonam-resistant but {100 * (1 - az.susceptible_downstream_test_probability):.1f}\\% of aztreonam-susceptible episodes"]
TEXT['paper1_detection_validation/results_only.tex'] += [
  f"Ceftriaxone-resistant episodes made up {_share_R:.1f}\\% of the episodes eligible for the fosfomycin pattern but supplied {_share_obs:.1f}\\% of its observed fosfomycin results; its site--era-stratified ratio was {_cf2.mh_site_era:.1f}",
  f"for {_eu} escalation-direction patterns and in a lower one for {_ed}, against {_su} and {_sd} suppression-direction patterns",
  f"the corresponding counts were {_ceu} and {_ced} for escalation-direction and {_csu} and {_csd} for suppression-direction candidates"]
# Discussion wording: "about three times as often"; "four downstream antibiotics" hold most strong escalation patterns; "about half" observed
assert 2.5 <= _eu / _ed <= 3.5 and top.head(4).sum() > len(strong_esc) / 2
import re as _re
_obs_all = float(_re.search(r"amrnum:observation_coverage\.all_sites\.observed_share\|1\\endcsname\{([0-9.]+)\}", pathlib.Path('paper1_detection_validation/generated_numbers.tex').read_text()).group(1))
assert 0.45 <= _obs_all <= 0.55, f"'only about half of eligible results were observed' (paper 1 Discussion) no longer holds: {_obs_all}"
# paper 2 (base): bounds contrast, concentration vs enrichment, masking cefazolin
_d = pd.read_csv(pathlib.Path(a.run_root) / 'outputs/tables/combined/organisms/escherichia_coli/table_k_prevalence_shift_diagnostics.csv').set_index('Drug')
def _bin(drug): r = _d.loc[drug]; return 100 * r['Binary-evaluable tested n'] / r['Eligible n']
_cip = sh.loc['CIPROFLOXACIN']; _fs = sh.loc['FOSFOMYCIN']; _tt = sh.loc['TETRACYCLIN']
_fd = _d.loc['FOSFOMYCIN']
TEXT[P2 + 'results_only.tex'] = TEXT.get(P2 + 'results_only.tex', []) + [
  f"Ciprofloxacin, with such a result in {_bin('CIPROFLOXACIN'):.1f}\\% of eligible opportunities, had bounds of {_cip.prevalence_lower_bound_pct:.1f}--{_cip.prevalence_upper_bound_pct:.1f}\\%, "
  f"whereas fosfomycin, with one in {_bin('FOSFOMYCIN'):.2f}\\%, had a naive prevalence of {_fs.naive_prevalence_pct:.1f}\\% but bounds of {_fs.prevalence_lower_bound_pct:.2f}--{_fs.prevalence_upper_bound_pct:.1f}\\%",
  f"For tetracycline, {100 * _tt.cascade_trigger_fraction:.1f}\\% of observed results came from cascade-triggered episodes, in which resistance was {_tt.cascade_prevalence_pct:.1f}\\%, "
  f"against {_tt.independent_prevalence_pct:.1f}\\% in non-triggered observed episodes ($\\rho^{{obs}}$ {_tt.independent_prevalence_pct / _tt.cascade_prevalence_pct:.2f})",
  f"Fosfomycin results were even more concentrated ($\\kappa$ {100 * _fs.cascade_trigger_fraction:.1f}\\%), but only {int(_fd['No-trigger tested n'])} non-triggered results, "
  f"{_WORDS[int(round(_fs.independent_prevalence_pct / 100 * _fd['No-trigger tested n']))]} of them resistant, were available for comparison"]
_mk2 = pd.read_csv(pathlib.Path(a.run_root) / 'outputs/tables/combined/organisms/escherichia_coli/masking_experiment/masking_results.csv')
_mk2 = _mk2[_mk2.mechanism.eq('transplanted')].assign(err=lambda t: t.mnar_lambda0_error_pp.abs())
_big = _mk2[_mk2.err > 10].sort_values('err')
_top2 = _big.tail(2)
CEFAZOLIN_CLAIM_OK = len(_big) == 3 and _big.truth_antibiotic.eq('CEFAZOLIN').all()
TEXT[P2 + 'results_only.tex'] += [
  f"the two largest ({_top2.err.iloc[0]:.1f} and {_top2.err.iloc[1]:.1f} points) arose when {100 * _top2.kept_share.iloc[0]:.0f}\\% and {100 * _top2.kept_share.iloc[1]:.0f}\\% of its results were kept"]
TEXT[P2 + 'discussion_only.tex'] = TEXT.get(P2 + 'discussion_only.tex', []) + [
  f"fosfomycin's naive resistance of {_fs.naive_prevalence_pct:.1f}\\%, from resistant or susceptible results in {_bin('FOSFOMYCIN'):.2f}\\% of eligible episodes"]
TEXT[P2 + 'abstract_body.tex'] = TEXT.get(P2 + 'abstract_body.tex', []) + [
  f"fosfomycin, with a resistant or susceptible result in {_bin('FOSFOMYCIN'):.2f}\\% of eligible episodes, had a naive prevalence of {_fs.naive_prevalence_pct:.1f}\\% but bounds of {_fs.prevalence_lower_bound_pct:.2f}--{_fs.prevalence_upper_bound_pct:.1f}\\%"]
# --- added 2026-10-28 (seal): why the largest masking errors arose (manuscript_edit_log/2026-10-28/masking_diag/;
# re-run scenario_positivity.py and trigger_positivity.py after a new export: they read export_archives/hpc_final)
_MD = pathlib.Path('manuscript_edit_log/2026-10-28/masking_diag')
_sp = pd.read_csv(_MD / 'scenario_positivity.csv').sort_values('no_tilt_error_pp').tail(2)
TEXT[P2 + 'results_only.tex'] += [
  f"In those two scenarios, {100 * _sp.trigger_in_kept.iloc[0]:.1f}\\% and {100 * _sp.trigger_in_kept.iloc[1]:.1f}\\% of the kept results came from episodes with an observed trigger, "
  f"against {100 * _sp.trigger_in_hidden.iloc[0]:.1f}\\% and {100 * _sp.trigger_in_hidden.iloc[1]:.1f}\\% of the hidden ones, so the resistance model extrapolated to most hidden episodes "
  f"from {int(_sp.kept_without_trigger.iloc[0])} and {int(_sp.kept_without_trigger.iloc[1])} kept episodes without one"]
assert sorted(round(x, 1) for x in _sp.no_tilt_error_pp) == sorted(round(x, 1) for x in _top2.err), 'scenario table and masking results disagree: re-run scenario_positivity.py'
_tp = pd.read_csv(_MD / 'trigger_positivity.csv')
_exposed = _tp[((1 - _tp.trigger_in_untested) > 0.10) & ((1 - _tp.trigger_in_tested) < 0.01)]
_gap = (sh.loc[_exposed.drug, 'naive_prevalence_pct'] - sh.loc[_exposed.drug, 'mnar_lambda0_prevalence_pct']).abs()
MASKING_MECHANISM_OK = sorted(_exposed.drug) == ['CEFTRIAXON', 'TIGECYCLIN'] and bool((_gap < 0.4).all())
print(f"masking mechanism: exposed antibiotics {sorted(_exposed.drug)}, |naive - no-tilt| {_gap.round(2).to_dict()} ('only for tigecycline and ceftriaxone ... within 0.4 points')")
# --- added 2026-11-02, updated 2026-11-07: the author's wording of both ESCMID abstracts (paper 1 saved 7 November,
# paper 2 saved 3 November). The abstracts' %-comment lines (earlier wording) are ignored, so only printed text counts.
_gn = pathlib.Path('paper1_detection_validation/generated_numbers.tex').read_text()
_num = lambda key: _re.search(r"^\\expandafter\\def\\csname amrnum:" + _re.escape(key) + r"\\endcsname\{(.*)\}\s*$", _gn, _re.M).group(1).replace("{,}", ",")
_tj = _tn['strong']
_p1r = pathlib.Path('paper1_detection_validation/results_only.tex').read_text()
_hv = _re.search(r"(\d+\.\d)\\% to (\d+\.\d)\\% were also validated", _p1r).groups()
_hn = _re.search(r"compared with (\d+\.\d)\\% to (\d+\.\d)\\% \(Supplementary", _p1r).groups()
_n15 = values['validated patterns at least 1.5-fold from ER 1']
# 2026-11-11: abstract 1 revised after the author's review comments (defined terms, plain exclusions, 'validated'
# instead of 'supported', fuller site names, longer conclusion); same numbers. Backup: manuscript_edit_log/2026-11-11/escmid_abstract1/.
TEXT[ESC + 'paper1_abstract.tex'] = [
  # 2026-11-15 final: the author's version with four fixes. Fosfomycin shares are within fosfomycin-eligible episodes
  # with a ceftriaxone R/S result (6,733 R / 65,531 S support rows; 593 / 97 with a fosfomycin result).
  f"Of {_num('opportunity_space.all_sites.eligible_rows')} eligible episode--antibiotic opportunities",
  f"Among {int(fo.resistant_support_n + fo.susceptible_support_n):,} fosfomycin-eligible episodes with ceftriaxone results, fosfomycin results were recorded in "
  f"{pct(fo.resistant_downstream_test_probability)} of ceftriaxone-resistant versus {p2(fo.susceptible_downstream_test_probability)} of ceftriaxone-susceptible episodes (ER {fo.escalation_ratio:.1f})",
  f"Although ceftriaxone-resistant episodes represented {_share_R:.1f}\\% of these, they contributed {_share_obs:.1f}\\% of recorded fosfomycin results",
  f"Directions persisted after adjustment ({_num('adjusted_direction.consistent')}/{_num('adjusted_direction.estimable')} estimable pairs) "
  f"and within institution--era strata ({int(ws.mh_site_era_same_direction.sum())}/{len(ws)})",
  f"{_hv[0]}--{_hv[1]}\\% of pairs validated at one institution replicated in the same direction at another, compared with {_hn[0]}--{_hn[1]}\\% of pairs not validated there",
  f"ERs varied significantly between years in {_tj['varies_beyond_sampling']}/{_tj['site_series_followed']} institution-level series of strong ($\\geq$1.5-fold) pairs",
  "(2006--2025)"]
TEXT[ESC + 'paper2_abstract.tex'] = [
  f"We studied {len(sh)} antibiotics with empirically validated associations between upstream resistance and increased downstream result availability",
  f"shifted these log-odds by $-\\lambda$ ($\\lambda$ from ${min(grid):g}$ to ${max(grid):g}$)",
  f"the median bound width was {(sh.prevalence_upper_bound_pct - sh.prevalence_lower_bound_pct).median():.1f} percentage points",
  f"Bounds were entirely below 20\\% for ertapenem and piperacillin\u2013tazobactam, entirely above 20\\% for ciprofloxacin, and crossed 20\\% for {t20['across']} antibiotics",
  f"Fosfomycin had {_bin('FOSFOMYCIN'):.2f}\\% binary-result coverage and {_fs.naive_prevalence_pct:.1f}\\% observed-result resistance, but bounds of {_fs.prevalence_lower_bound_pct:.2f}\u2013{_fs.prevalence_upper_bound_pct:.1f}\\%",
  f"Sensitivity estimates crossed 20\\% for {_WORDS.get(len(t20['cross']), len(t20['cross']))} antibiotics"]
# --- added 2026-11-08: Fig. 5 caption, drawn flows that stay >= 1.5-fold within site and era (top 30 by support per direction)
from amr_cascade_platform.reporting.builders import directional_views as _dv
def _sankey_survivors(direction):
    # the figure's own selection (display_selection + bipartite_links: support, then effect size, top 30);
    # 8 Nov: an earlier version sorted by support alone and broke a three-way tie in b differently (5 instead of 6)
    top = _dv.bipartite_links(_dv.display_selection(v, direction, 1.5), direction, top_n=30)
    m = top.merge(ws[['upstream_antibiotic', 'downstream_antibiotic', 'mh_site_era_same_direction', 'mh_site_era_fold']],
                  on=['upstream_antibiotic', 'downstream_antibiotic'], how='left')
    return int((m.mh_site_era_same_direction.fillna(False).astype(bool) & (m.mh_site_era_fold >= 1.5)).sum()), len(m)
_sa, _na = _sankey_survivors('escalation')
_sb, _nb = _sankey_survivors('suppression')
TEXT['paper1_detection_validation/results_only.tex'].append(
  f"{_sa} of the {_na} flows in \\textbf{{a}} and {_sb} of the {_nb} in \\textbf{{b}} remained at least 1.5-fold from ER $=1$")
# --- added 2026-11-17: Fig. 5 caption, how many patterns were eligible for display and how many are drawn
_ea, _eb = len(_dv.display_selection(v, 'escalation', 1.5)), len(_dv.display_selection(v, 'suppression', 1.5))
assert _na == _nb == 30, (_na, _nb)
TEXT['paper1_detection_validation/results_only.tex'] += [
  f"(${_ea}$ in the escalation and ${_eb}$ in the suppression direction), the 30 with the most episodes in each direction are shown".replace('$', ''),
  "Observation flows of the 30 validated patterns with the most episodes in each direction"]
# --- added 2026-11-20 (reviewer comments): AWaRe transitions validated relative to the retained candidates (Fisher's exact test)
from scipy.stats import fisher_exact as _fx
def _aw(u, d, cu, cd):
    o, p = _fx([[u, d], [cu - u, cd - d]])
    return f"{u} of {cu}, {100 * u / cu:.1f}\\%, versus {d} of {cd}, {100 * d / cd:.1f}\\%", f"{o:.2f}", f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
_e, _s = _aw(_eu, _ed, _ceu, _ced), _aw(_su, _sd, _csu, _csd)
TEXT['paper1_detection_validation/results_only.tex'] += [
  f"in the escalation direction ({_e[0]}; Fisher's exact test, odds ratio {_e[1]}, $p={_e[2]}$)",
  f"in the suppression direction ({_s[0]}; odds ratio {_s[1]}, $p={_s[2]}$)"]
assert float(_e[2]) < 0.05 <= float(_s[2]), (_e, _s)  # wording: 'somewhat more often' (escalation) vs 'not clearly' (suppression)
bad = 0
for f, strings in TEXT.items():
    t = pathlib.Path(f).read_text()
    if f.startswith(ESC):
        t = "\n".join(line for line in t.splitlines() if not line.lstrip().startswith('%'))
    for s in strings:
        if s not in t: print('MISSING in', f, ':', s); bad += 1
if not MASKING_WORDING_OK: print('CLAIM no longer holds: paper 2 masking wording ("mostly positive" / "usually too high")'); bad += 1
if not TEMPORAL_WORDING_OK: print('CLAIM no longer holds: year-by-year wording ("about half", "clustered", "mostly never present", "nearly all")'); bad += 1
if not SUPPRESSION_CLAIM_OK: print('CLAIM no longer holds: paper 1 Discussion, strong suppression-direction patterns lose most of their strength within site and era'); bad += 1
if not CEFAZOLIN_CLAIM_OK: print('CLAIM no longer holds: paper 2 masking, "the three no-tilt errors above 10 percentage points all concerned cefazolin"'); bad += 1
if not MASKING_MECHANISM_OK: print('CLAIM no longer holds: paper 2 Results, "such an imbalance arose only for tigecycline and ceftriaxone, within 0.4 percentage points"'); bad += 1
if not RIDGE_CLAIM_OK: print('CLAIM no longer holds: Supplement ridge note, "changes of direction occur mainly where the adjusted odds ratio is close to 1"'); bad += 1
print('text check:', 'OK' if not bad else f'{bad} missing')
sys.exit(1 if bad else 0)

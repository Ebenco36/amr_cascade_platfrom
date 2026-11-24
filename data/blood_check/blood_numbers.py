"""Numbers and Supplementary-table rows of the post hoc blood-culture analysis, computed from blood_run/ (the blood-only
pipeline run) and the final run's pooled validation. Used by apply_blood.py (to write the text) and by hand_numbers.py
(to check that the text still prints these values)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]; HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver
K = ["upstream_antibiotic", "downstream_antibiotic"]; VAL = ["robust", "supported"]


def compute(run_root="export_archives/hpc_final"):
    A = ROOT / run_root / "data/artifacts/cascade/combined/organisms/escherichia_coli"
    b = pd.read_parquet(HERE / "blood_run/validation_results.parquet")
    p = pd.read_parquet(A / "validation_results.parquet")
    ctb = pd.read_parquet(HERE / "blood_run/cotesting_pairs.parquet")
    escb = pd.read_parquet(HERE / "blood_run/escalation_results.parquet")
    m = b[K + ["observed_escalation_ratio", "validation_status"]].rename(columns={"observed_escalation_ratio": "er_b", "validation_status": "st_b"}).merge(
        p[K + ["observed_escalation_ratio", "validation_status"]].rename(columns={"observed_escalation_ratio": "er_p", "validation_status": "st_p"}), on=K, how="outer")
    m["bval"], m["pval"] = m.st_b.isin(VAL), m.st_p.isin(VAL)
    same = np.sign(np.log(m.er_b)) == np.sign(np.log(m.er_p))
    fold = lambda x: np.exp(np.abs(np.log(x)))
    bv, pv = m[m.bval], m[m.pval]
    inb = pv[pv.st_b.notna()]
    strong = pv[fold(pv.er_p) >= 1.5]; strong_b = strong[strong.st_b.notna()]
    notb = pv[pv.st_b.isna()].merge(ctb[K].assign(screened=True), on=K, how="left")
    opp = m[m.bval & m.pval & ~same]; only = m[m.bval & m.st_p.notna() & ~m.pval]
    fo = escb[(escb.upstream_antibiotic == "CEFTRIAXON") & (escb.downstream_antibiotic == "FOSFOMYCIN")].iloc[0]
    n = dict(screened=len(ctb), candidates=int(m.st_b.notna().sum()), validated=len(bv), robust=int(m.st_b.eq("robust").sum()),
             supported=int(m.st_b.eq("supported").sum()), same=int((bv.pval & same[bv.index]).sum()), opposite=len(opp),
             only_blood=len(only), not_pooled_candidate=int(bv.st_p.isna().sum()), pooled_validated=len(pv), pooled_in_blood=len(inb),
             pooled_not_in_blood=len(notb), not_in_blood_screened=int(notb.screened.eq(True).sum()),
             strong=len(strong), strong_in_blood=len(strong_b), strong_same=int((strong_b.bval & same[strong_b.index]).sum()),
             fos_blood_observed=int(fo.positive_tested_n + fo.negative_tested_n))
    n["not_in_blood_size"] = n["pooled_not_in_blood"] - n["not_in_blood_screened"]
    assert n["validated"] == n["same"] + n["opposite"] + n["only_blood"] + n["not_pooled_candidate"]
    assert n["robust"] + n["supported"] == n["validated"]
    cef_tet = m[(m.upstream_antibiotic == "CEFEPIM") & (m.downstream_antibiotic == "TETRACYCLIN")].iloc[0]
    n["cef_tet_pooled"], n["cef_tet_blood"] = f"{cef_tet.er_p:.2f}", f"{cef_tet.er_b:.2f}"
    n["opp_downstream"] = opp.downstream_antibiotic.value_counts().to_dict()
    res = AntibioticClassificationResolver(ROOT / "data/antibiotic_classification_complete.csv")
    name = lambda d: str(res.canonical_display_label(d)).capitalize().replace("/", "--")
    rows = []
    for title, frame in (("Opposite direction in the primary analysis", opp), ("Retained but not validated in the primary analysis", only)):
        rows.append(f"\\addlinespace\n\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{title} ({len(frame)})}}}} \\\\")
        f = frame.assign(fo=fold(frame.er_b)).sort_values(["fo", "upstream_antibiotic"], ascending=[False, True])
        for r in f.itertuples():
            rows.append(f"{name(r.upstream_antibiotic)} & {name(r.downstream_antibiotic)} & {r.er_b:.2f} & \\texttt{{{r.st_b}}} & {r.er_p:.2f} & \\texttt{{{r.st_p}}} \\\\")
    return n, "\n".join(rows)


if __name__ == "__main__":
    n, rows = compute(); print(n); print(rows[:600])


def texts(n, rows):
    """The new paper-1 text, built from the numbers (one place for apply_blood.py and hand_numbers.py)."""
    d = n["opp_downstream"]
    methods = ("A third post hoc analysis repeated candidate screening and validation in blood-culture episodes only, with the same "
               "settings: the co-testing screen, the sample-size thresholds, permutation testing with Benjamini--Hochberg correction "
               "across the blood-culture candidates, bootstrap sign stability and replication were all computed within these episodes, "
               "and the resulting labels were compared with those of the primary analysis (Supplementary Note~\\ref{sec:supp-blood}).")
    results = (f"In the post hoc blood-culture analysis (\\runnum{{cohort_characteristics.all_sites.specimen_blood.n}} episodes), "
               f"{n['candidates']} pairs were retained and {n['validated']} validated ({n['robust']} \\texttt{{robust}}, {n['supported']} "
               f"\\texttt{{supported}}; Supplementary Note~\\ref{{sec:supp-blood}}). Every pair validated in blood cultures had also been "
               f"retained in the primary analysis. Of the {n['validated']}, {n['same']} had been validated there in the same direction, "
               f"{n['only_blood']} had been retained but not validated and {n['opposite']} had been validated in the opposite direction, "
               f"{d['TETRACYCLIN'] + d['AMOXICILLIN/CLAVULANIC ACID']} of them with tetracycline or amoxicillin--clavulanic acid and "
               f"{d['CEFEPIM']} with cefepime as the downstream antibiotic; cefepime~$\\rightarrow$~tetracycline, for example, had an ER of "
               f"{n['cef_tet_pooled']} in all specimens and {n['cef_tet_blood']} in blood cultures. Of the \\runnum{{validation.validated}} "
               f"patterns validated in the primary analysis, {n['pooled_in_blood']} were retained in blood cultures and {n['same']} of these "
               f"were validated there in the same direction; of the other {n['pooled_not_in_blood']}, {n['not_in_blood_size']} did not meet "
               f"the sample-size criteria in blood cultures and {n['not_in_blood_screened']} were removed by the co-testing screen there. "
               f"Ceftriaxone~$\\rightarrow$~fosfomycin could not be tested, because fosfomycin was observed in only {n['fos_blood_observed']} "
               f"blood-culture episodes.")
    discussion = (f"Specimen type acted in a similar way. A blood-culture-only analysis with the same screen and validation found no "
                  f"pattern that the primary analysis had not retained, but {n['only_blood']} of its {n['validated']} validated patterns "
                  f"had not been validated in all specimens together and {n['opposite']} had the opposite direction, so the combined ERs "
                  f"describe the mostly urinary population and need not hold within a specimen type.")
    note = f"""% =============================================================================
\\section{{Blood-culture analysis}}
\\label{{sec:supp-blood}}
% =============================================================================

Urine made up \\runnum{{cohort_characteristics.all_sites.specimen_urine.share|pct1}} of the culture episodes, so the primary
analysis mainly reflects urinary cultures. To test whether the patterns differ in blood cultures, screening and validation
were repeated on the \\runnum{{cohort_characteristics.all_sites.specimen_blood.n}} blood-culture episodes alone, with the
settings of the primary analysis (Methods): the co-testing screen (PBI $\\ge0.95$), the sample-size thresholds, 1{{,}}000
within-site permutations with Benjamini--Hochberg correction across the blood-culture candidates, 1{{,}}000 bootstrap
replicates and the replication criteria. The co-testing screen removed {n['screened']} pairs and {n['candidates']} were
retained. Supplementary Table~\\ref{{tab:supp-blood}} compares the results with the primary analysis, and Supplementary
Table~\\ref{{tab:supp-blood-pairs}} lists the pairs whose result differed. Of the {n['strong']}
validated patterns at least 1.5-fold from ER $=1$ in the primary analysis, {n['strong_in_blood']} were retained in blood
cultures and {n['strong_same']} were validated there in the same direction.

\\begin{{table}}[H]
\\centering
\\small
\\caption{{Blood-culture analysis compared with the primary analysis (all specimens): numbers of antibiotic pairs.
  Validated, labelled \\texttt{{robust}} or \\texttt{{supported}} (Methods).}}
\\label{{tab:supp-blood}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{3.9in}}r@{{}}}}
\\toprule
 & \\textbf{{Pairs}} \\\\
\\midrule
\\textit{{Blood cultures}} & \\\\
\\quad Retained & {n['candidates']} \\\\
\\quad Validated & {n['validated']} \\\\
\\quad\\quad \\texttt{{robust}} / \\texttt{{supported}} & {n['robust']} / {n['supported']} \\\\
\\quad\\quad Same direction as in the primary analysis & {n['same']} \\\\
\\quad\\quad Opposite direction to the primary analysis & {n['opposite']} \\\\
\\quad\\quad Retained but not validated in the primary analysis & {n['only_blood']} \\\\
\\quad\\quad Not retained in the primary analysis & {n['not_pooled_candidate']} \\\\
\\addlinespace
\\textit{{Primary analysis}} & \\\\
\\quad Validated & {n['pooled_validated']} \\\\
\\quad\\quad Retained in blood cultures & {n['pooled_in_blood']} \\\\
\\quad\\quad\\quad Validated there in the same direction & {n['same']} \\\\
\\quad\\quad Not retained in blood cultures: sample-size criteria not met & {n['not_in_blood_size']} \\\\
\\quad\\quad Not retained in blood cultures: removed by the co-testing screen & {n['not_in_blood_screened']} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[H]
\\centering
\\small
\\caption{{The {n['opposite'] + n['only_blood']} pairs validated in blood cultures whose result differed in the primary analysis
  (all specimens). ER, continuity-corrected escalation ratio: above 1, the downstream antibiotic (B) was observed more often
  when the upstream antibiotic (A) was resistant; below 1, less often. Labels as in the primary analysis (Methods).}}
\\label{{tab:supp-blood-pairs}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{adjustbox}}{{max width=\\textwidth}}
\\begin{{tabular}}{{@{{}}llrlrl@{{}}}}
\\toprule
 & & \\multicolumn{{2}}{{c}}{{\\textbf{{Blood cultures}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Primary analysis}}}} \\\\
\\cmidrule(lr){{3-4}}\\cmidrule(l){{5-6}}
\\textbf{{Upstream antibiotic (A)}} & \\textbf{{Downstream antibiotic (B)}} & \\textbf{{ER}} & \\textbf{{Label}} & \\textbf{{ER}} & \\textbf{{Label}} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{adjustbox}}
\\end{{table}}

\\clearpage

"""
    return methods, results, discussion, note

"""Uses pair_specimen_counts.parquet (blood_check.py). Validates pooled counts against the edge report, then asks:
(1) the 18 pairs removed by the pooled co-testing screen: co-observation within blood; (2) retained candidates: blood-only ER
vs pooled ER (dilution)."""
import numpy as np, pandas as pd
A = "export_archives/hpc_final/data/artifacts/cascade/combined/organisms/escherichia_coli/"
c = pd.read_parquet("manuscript_edit_log/2026-11-20/blood_check/pair_specimen_counts.parquet")
c = c.rename(columns={"downstream_tested_sum": "obs", "downstream_tested_count": "n"})
w = c.pivot_table(index=["upstream_antibiotic", "downstream_antibiotic", "specimen"], columns="upstream_susceptibility",
                  values=["obs", "n"], aggfunc="sum", fill_value=0)
w.columns = [f"{a}_{b[0]}" for a, b in w.columns]; w = w.reset_index()
pooled = w.groupby(["upstream_antibiotic", "downstream_antibiotic"])[["obs_R", "n_R", "obs_S", "n_S"]].sum().reset_index()
e = pd.read_parquet(A + "edge_report.parquet")
m = pooled.merge(e[["upstream_antibiotic", "downstream_antibiotic", "resistant_support_n", "resistant_tested_n", "susceptible_support_n", "susceptible_tested_n", "escalation_ratio", "validation_status"]],
                 on=["upstream_antibiotic", "downstream_antibiotic"])
ok = (m.n_R == m.resistant_support_n) & (m.obs_R == m.resistant_tested_n) & (m.n_S == m.susceptible_support_n) & (m.obs_S == m.susceptible_tested_n)
print(f"validation: {ok.sum()} of {len(m)} edge-report pairs reproduced exactly")
def er(df):
    return ((df.obs_R + 0.5) / (df.n_R + 1)) / ((df.obs_S + 0.5) / (df.n_S + 1))
b = w[w.specimen.eq("blood")].copy(); b["er_blood"] = er(b)
u = w[w.specimen.eq("urine")].copy(); u["er_urine"] = er(u)
# thresholds as in the primary analysis: >=25 R/S rows, >=5 per branch, >=5 observed downstream
b["meets"] = (b.n_R + b.n_S >= 25) & (b.n_R >= 5) & (b.n_S >= 5) & (b.obs_R + b.obs_S >= 5)
# simple 95% CI for log ER (continuity-corrected, same formula as Methods)
se = np.sqrt((1/(b.obs_R+0.5) - 1/(b.n_R+1)) + (1/(b.obs_S+0.5) - 1/(b.n_S+1)))
b["lo"], b["hi"] = np.exp(np.log(b.er_blood) - 1.96*se), np.exp(np.log(b.er_blood) + 1.96*se)
fold = lambda x: np.exp(np.abs(np.log(x)))
# (1) pairs removed by the co-testing screen
ct = pd.read_parquet(A + "cotesting_pairs.parquet")
print("\n(1) pairs removed by the pooled co-testing screen:", len(ct))
x = ct.merge(b, on=["upstream_antibiotic", "downstream_antibiotic"], how="left")
x["p_obs_given_R_blood"] = x.obs_R / x.n_R; x["p_obs_given_S_blood"] = x.obs_S / x.n_S
print(x[["upstream_antibiotic", "downstream_antibiotic", "p_downstream_given_upstream", "n_R", "n_S", "p_obs_given_R_blood", "p_obs_given_S_blood", "er_blood", "lo", "hi"]].round(3).to_string())
# (2) retained candidates: blood-specific strong patterns diluted in the pooled ER
vr = pd.read_parquet(A + "validation_results.parquet")[["upstream_antibiotic", "downstream_antibiotic", "observed_escalation_ratio", "validation_status"]]
y = vr.merge(b, on=["upstream_antibiotic", "downstream_antibiotic"], how="left").merge(u[["upstream_antibiotic", "downstream_antibiotic", "er_urine"]], on=["upstream_antibiotic", "downstream_antibiotic"], how="left")
y["validated"] = y.validation_status.isin(["robust", "supported"])
est = y[y.meets.fillna(False)]
strong_blood = est[(fold(est.er_blood) >= 1.5) & ((est.lo > 1) | (est.hi < 1))]
print(f"\n(2) retained candidates: {len(vr)}; estimable in blood (same size thresholds): {len(est)}")
print(f"   blood ER >=1.5-fold with 95% CI excluding 1: {len(strong_blood)}")
same_dir = np.sign(np.log(strong_blood.er_blood)) == np.sign(np.log(strong_blood.observed_escalation_ratio))
print(f"     of which pooled ER in the same direction: {int(same_dir.sum())}; validated in the pooled analysis: {int(strong_blood.validated.sum())}")
missed = strong_blood[~strong_blood.validated | (fold(strong_blood.observed_escalation_ratio) < 1.5) | ~same_dir]
print(f"   blood-strong but NOT validated, or pooled <1.5-fold, or opposite direction (candidates for dilution): {len(missed)}")
print(missed.assign(pooled=missed.observed_escalation_ratio)[["upstream_antibiotic", "downstream_antibiotic", "n_R", "n_S", "er_blood", "lo", "hi", "pooled", "er_urine", "validation_status"]]
      .sort_values("er_blood", key=lambda s: -np.abs(np.log(s))).round(2).head(25).to_string())
y.to_csv("manuscript_edit_log/2026-11-20/blood_check/blood_vs_pooled.csv", index=False)

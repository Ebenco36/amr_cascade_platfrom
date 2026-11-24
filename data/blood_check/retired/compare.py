"""Compare the blood-only validation (blood_run/) with the pooled validation of the final run."""
import numpy as np, pandas as pd
A = "export_archives/hpc_final/data/artifacts/cascade/combined/organisms/escherichia_coli/"
K = ["upstream_antibiotic", "downstream_antibiotic"]
b = pd.read_parquet("manuscript_edit_log/2026-11-20/blood_check/blood_run/validation_results.parquet")
p = pd.read_parquet(A + "validation_results.parquet")
VAL = ["robust", "supported"]
b["bval"] = b.validation_status.isin(VAL); p["pval"] = p.validation_status.isin(VAL)
sgn = lambda x: np.sign(np.log(x))
m = b[K + ["observed_escalation_ratio", "validation_status", "bval"]].rename(columns={"observed_escalation_ratio": "er_b", "validation_status": "st_b"}).merge(
    p[K + ["observed_escalation_ratio", "validation_status", "pval"]].rename(columns={"observed_escalation_ratio": "er_p", "validation_status": "st_p"}), on=K, how="outer")
bv = m[m.bval.eq(True)]
print("blood candidates", int(m.st_b.notna().sum()), "| blood validated", len(bv), dict(b.validation_status.value_counts()))
same = sgn(bv.er_b) == sgn(bv.er_p)
print(" blood-validated: also pooled-validated, same direction:", int((bv.pval.eq(True) & same).sum()))
print("                  pooled-validated but opposite direction:", int((bv.pval.eq(True) & ~same).sum()))
print("                  pooled candidate, not pooled-validated:", int((bv.st_p.notna() & bv.pval.ne(True)).sum()))
print("                  not a pooled candidate:", int(bv.st_p.isna().sum()))
pv = m[m.pval.eq(True)]
inb = pv[pv.st_b.notna()]
print("pooled validated", len(pv), "| candidates in blood", len(inb))
s2 = sgn(inb.er_b) == sgn(inb.er_p)
print("  validated in blood, same direction:", int((inb.bval.eq(True) & s2).sum()), "| validated in blood, opposite:", int((inb.bval.eq(True) & ~s2).sum()),
      "| not validated in blood:", int((inb.bval.ne(True)).sum()), "| of which blood ER same side:", int((inb.bval.ne(True) & s2).sum()))
fold = lambda x: np.exp(np.abs(np.log(x)))
st = pv[fold(pv.er_p) >= 1.5]; stb = st[st.st_b.notna()]
print("pooled >=1.5-fold:", len(st), "| candidates in blood", len(stb), "| validated in blood same direction", int((stb.bval.eq(True) & (sgn(stb.er_b) == sgn(stb.er_p))).sum()))
for u, d in [("MOXIFLOXACIN", "CEFTOLOZANE/TAZOBACTAM"), ("CEFEPIM", "TETRACYCLIN"), ("CEFTRIAXON", "FOSFOMYCIN"), ("GENTAMICIN", "AMIKACIN"), ("AZTREONAM", "PIPERACILLIN/TAZOBACTAM")]:
    r = m[(m.upstream_antibiotic == u) & (m.downstream_antibiotic == d)]
    print(u, "->", d, r[["er_b", "st_b", "er_p", "st_p"]].round(2).to_dict("records"))
print("\nblood-validated and not pooled-validated (or opposite):")
x = bv[(bv.pval.ne(True)) | (sgn(bv.er_b) != sgn(bv.er_p))].copy(); x["fold_b"] = fold(x.er_b)
print(x.sort_values("fold_b", ascending=False)[K + ["er_b", "st_b", "er_p", "st_p"]].round(2).to_string())
m.to_csv("manuscript_edit_log/2026-11-20/blood_check/blood_vs_pooled_validation.csv", index=False)

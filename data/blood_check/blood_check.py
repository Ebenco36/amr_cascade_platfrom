"""Post hoc check (20 Nov 2026): are there blood-culture-only directional patterns that the pooled co-testing screen removed or the
pooled ER diluted? Reads the final run's gold tables (read-only). Step 1 reproduces pooled numbers to validate the definitions."""
import sys, numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow as pa, pyarrow.compute as pc
sys.path.insert(0, "src")
from amr_cascade_platform.cascade.statistics.cascade_covariate_builder import CascadeCovariateBuilder
G = "export_archives/hpc_final/data/gold/combined/organisms/escherichia_coli/"
A = "export_archives/hpc_final/data/artifacts/cascade/combined/organisms/escherichia_coli/"
ep = pd.read_parquet(G + "culture_episodes.parquet", columns=["source_site", "order_proc_id_coded", "culture_description"])
ep["specimen"] = ep.culture_description.map(CascadeCovariateBuilder._specimen_type_category)
print(ep.specimen.value_counts().to_dict())
d = ds.dataset(G + "drug_pair_episodes.parquet")
print(d.schema)
s = d.to_table(columns=["upstream_susceptibility"]).column(0)
print(pc.value_counts(s).to_pylist()[:10])

# ---- per pair and specimen: rows with an R/S upstream result and an eligible downstream antibiotic ----
spec = pa.Table.from_pandas(ep[["source_site", "order_proc_id_coded", "specimen"]], preserve_index=False)
ups = pc.unique(d.to_table(columns=["upstream_antibiotic"]).column(0)).to_pylist()
parts = []
for u in ups:
    t = d.to_table(columns=["source_site", "order_proc_id_coded", "upstream_antibiotic", "upstream_susceptibility", "upstream_eligible",
                            "downstream_antibiotic", "downstream_tested", "downstream_eligible"],
                   filter=(ds.field("upstream_antibiotic") == u) & (ds.field("upstream_eligible") == 1) & (ds.field("downstream_eligible") == 1)
                   & ds.field("upstream_susceptibility").isin(["RESISTANT", "SUSCEPTIBLE"]))
    t = t.join(spec, keys=["source_site", "order_proc_id_coded"], join_type="inner")
    g = t.group_by(["upstream_antibiotic", "downstream_antibiotic", "specimen", "upstream_susceptibility"]).aggregate(
        [("downstream_tested", "sum"), ("downstream_tested", "count")])
    parts.append(g.to_pandas())
c = pd.concat(parts)
c.to_parquet("manuscript_edit_log/2026-11-20/blood_check/pair_specimen_counts.parquet")
print("rows", len(c))

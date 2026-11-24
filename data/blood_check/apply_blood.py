"""Writes the blood-culture analysis into paper 1 (author-approved 24 Nov 2026). Each anchor must match exactly once."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import blood_numbers as bn
P = pathlib.Path("paper1_detection_validation")
n, rows = bn.compute(); methods, results, discussion, note = bn.texts(n, rows)
G1 = ("Because the co-testing screen and the ERs were computed across all specimen types together, a conditional testing pattern "
      "confined to a subset of specimens, such as blood cultures (\\runnum{cohort_characteristics.all_sites.specimen_blood.share|pct1} "
      "of episodes, against \\runnum{cohort_characteristics.all_sites.specimen_urine.share|pct1} urine), could have been removed by the "
      "screen when the two antibiotics were routinely tested together in urine, or diluted towards ER $=1$ by the much larger number of "
      "episodes from specimens without the pattern.")
E = [("M1", P / "methods_only.tex", "with intermediate years skipped (Supplementary Note~\\ref{sec:supp-temporal}).",
      "with intermediate years skipped (Supplementary Note~\\ref{sec:supp-temporal}). " + methods),
     ("R1", P / "results_only.tex", "(Supplementary Fig.~\\ref{fig:supp-temporal}).\n\n\n\\subsection*{Adjusted associations}",
      "(Supplementary Fig.~\\ref{fig:supp-temporal}).\n\n" + results + "\n\n\n\\subsection*{Adjusted associations}"),
     ("D1", P / "discussion_only.tex", "than with organism-level patterns that transfer between institutions.",
      "than with organism-level patterns that transfer between institutions. " + discussion),
     ("D2", P / "discussion_only.tex", " " + G1, ""),
     ("S1", P / "supplementary.tex", "% =============================================================================\n\\section{Adjusted models: non-estimability",
      note + "% =============================================================================\n\\section{Adjusted models: non-estimability")]
log = []
for i, f, old, new in E:
    t = f.read_text(); assert t.count(old) == 1, (i, t.count(old)); f.write_text(t.replace(old, new)); log.append({"id": i, "file": str(f), "old": old, "new": new})
json.dump(log, open(pathlib.Path(__file__).parent / "applied_blood.json", "w"), indent=1); print(len(log), "changes")

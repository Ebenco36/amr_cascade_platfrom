# Paper 1: post hoc blood-culture analysis written in (24 November 2026, author-approved: "yes go ahead and update paper 1")

Source: blood-only run of the pipeline's own screening and validation (blood_validation.py -> blood_run/, 20 Nov, env hpc settings);
numbers and text built by blood_numbers.py; written by apply_blood.py; backups in backup/. The wide table was then put in
\begin{adjustbox}{max width=\textwidth} (as other Supplement tables) to remove a 4.3 pt overfull line.
Checks: hand_numbers OK (new checks regenerate all blood text from the run outputs); post_run.py: 8 documents 0 errors, 0
undefined references, 0 overfull lines; paper 1 upload folder built in the real journal class.

## M1 (methods_only.tex)
OLD:
```latex
with intermediate years skipped (Supplementary Note~\ref{sec:supp-temporal}).
```
NEW:
```latex
with intermediate years skipped (Supplementary Note~\ref{sec:supp-temporal}). A third post hoc analysis repeated candidate screening and validation in blood-culture episodes only, with the same settings: the co-testing screen, the sample-size thresholds, permutation testing with Benjamini--Hochberg correction across the blood-culture candidates, bootstrap sign stability and replication were all computed within these episodes, and the resulting labels were compared with those of the primary analysis (Supplementary Note~\ref{sec:supp-blood}).
```

## R1 (results_only.tex)
OLD:
```latex
(Supplementary Fig.~\ref{fig:supp-temporal}).


\subsection*{Adjusted associations}
```
NEW:
```latex
(Supplementary Fig.~\ref{fig:supp-temporal}).

In the post hoc blood-culture analysis (\runnum{cohort_characteristics.all_sites.specimen_blood.n} episodes), 632 pairs were retained and 188 validated (138 \texttt{robust}, 50 \texttt{supported}; Supplementary Note~\ref{sec:supp-blood}). Every pair validated in blood cultures had also been retained in the primary analysis. Of the 188, 151 had been validated there in the same direction, 28 had been retained but not validated and 9 had been validated in the opposite direction, 5 of them with tetracycline or amoxicillin--clavulanic acid and 3 with cefepime as the downstream antibiotic; cefepime~$\rightarrow$~tetracycline, for example, had an ER of 0.63 in all specimens and 1.73 in blood cultures. Of the \runnum{validation.validated} patterns validated in the primary analysis, 321 were retained in blood cultures and 151 of these were validated there in the same direction; of the other 114, 102 did not meet the sample-size criteria in blood cultures and 12 were removed by the co-testing screen there. Ceftriaxone~$\rightarrow$~fosfomycin could not be tested, because fosfomycin was observed in only 3 blood-culture episodes.


\subsection*{Adjusted associations}
```

## D1 (discussion_only.tex)
OLD:
```latex
than with organism-level patterns that transfer between institutions.
```
NEW:
```latex
than with organism-level patterns that transfer between institutions. Specimen type acted in a similar way. A blood-culture-only analysis with the same screen and validation found no pattern that the primary analysis had not retained, but 28 of its 188 validated patterns had not been validated in all specimens together and 9 had the opposite direction, so the combined ERs describe the mostly urinary population and need not hold within a specimen type.
```

## D2 (discussion_only.tex)
OLD:
```latex
 Because the co-testing screen and the ERs were computed across all specimen types together, a conditional testing pattern confined to a subset of specimens, such as blood cultures (\runnum{cohort_characteristics.all_sites.specimen_blood.share|pct1} of episodes, against \runnum{cohort_characteristics.all_sites.specimen_urine.share|pct1} urine), could have been removed by the screen when the two antibiotics were routinely tested together in urine, or diluted towards ER $=1$ by the much larger number of episodes from specimens without the pattern.
```
NEW:
```latex

```

## S1 (supplementary.tex)
OLD:
```latex
% =============================================================================
\section{Adjusted models: non-estimability
```
NEW:
```latex
% =============================================================================
\section{Blood-culture analysis}
\label{sec:supp-blood}
% =============================================================================

Because urine made up \runnum{cohort_characteristics.all_sites.specimen_urine.share|pct1} of the culture episodes, the
co-testing screen and the ERs of the primary analysis mainly reflect urinary cultures. This post hoc analysis repeated
candidate screening and validation in the \runnum{cohort_characteristics.all_sites.specimen_blood.n} blood-culture
episodes alone, with the settings of the primary analysis (Methods): the co-testing screen (PBI $\ge0.95$), the
sample-size thresholds, 1{,}000 within-site permutations with Benjamini--Hochberg correction across the blood-culture
candidates, 1{,}000 bootstrap replicates and the replication criteria, all computed within these episodes. In blood
cultures the co-testing screen removed 34 pairs and 632 pairs were retained.
Supplementary Table~\ref{tab:supp-blood} compares the resulting labels with those of the primary analysis and lists every
pattern validated in blood cultures that was not validated, or was validated in the opposite direction, in all specimens
together. Of the 120 validated patterns at least 1.5-fold from ER $=1$ in the primary analysis, 63
were retained in blood cultures and 31 were validated there in the same direction.

\begin{table}[H]
\centering
\small
\caption{Patterns validated in blood cultures compared with the primary analysis. ER, continuity-corrected escalation
  ratio in blood-culture episodes and in all specimens; labels as in the primary analysis (Methods).}
\label{tab:supp-blood}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.9in}r@{}}
\toprule
Pairs retained in blood cultures & 632 \\
Validated in blood cultures (\texttt{robust} or \texttt{supported}) & 188 (138 and 50) \\
\quad also validated in the primary analysis, same direction & 151 \\
\quad validated in the primary analysis in the opposite direction & 9 \\
\quad retained but not validated in the primary analysis & 28 \\
\quad not retained in the primary analysis & 0 \\
\addlinespace
Patterns validated in the primary analysis & 435 \\
\quad retained in blood cultures & 321 \\
\quad\quad validated there in the same direction & 151 \\
\quad not retained in blood cultures: sample-size criteria not met & 102 \\
\quad not retained in blood cultures: removed by the co-testing screen & 12 \\
\bottomrule
\end{tabular}

\medskip
\begin{tabular}{@{}llrlrl@{}}
\toprule
\textbf{Upstream} & \textbf{Downstream} & \textbf{ER, blood} & \textbf{Label, blood} & \textbf{ER, all} & \textbf{Label, all} \\
\midrule
\addlinespace
\multicolumn{6}{@{}l}{\textit{Validated in the opposite direction in the primary analysis (9)}} \\
Cefepime & Tetracycline & 1.73 & \texttt{robust} & 0.63 & \texttt{robust} \\
Aztreonam & Tetracycline & 1.50 & \texttt{robust} & 0.67 & \texttt{robust} \\
Cefuroxime & Tetracycline & 1.42 & \texttt{robust} & 0.77 & \texttt{supported} \\
Ceftriaxone & Amoxicillin--clavulanic acid & 1.26 & \texttt{robust} & 0.89 & \texttt{robust} \\
Cefepime & Amoxicillin--clavulanic acid & 1.22 & \texttt{robust} & 0.71 & \texttt{supported} \\
Gentamicin & Cefepime & 0.85 & \texttt{supported} & 1.04 & \texttt{supported} \\
Levofloxacin & Cefepime & 0.87 & \texttt{supported} & 1.11 & \texttt{supported} \\
Ceftriaxone & Cefepime & 0.88 & \texttt{supported} & 1.16 & \texttt{supported} \\
Levofloxacin & Ampicillin--sulbactam & 0.90 & \texttt{robust} & 1.12 & \texttt{supported} \\
\addlinespace
\multicolumn{6}{@{}l}{\textit{Retained but not validated in the primary analysis (28)}} \\
Ceftazidime & Piperacillin & 7.16 & \texttt{robust} & 4.18 & \texttt{mixed} \\
Ertapenem & Ceftazidime--avibactam & 4.26 & \texttt{supported} & 1.66 & \texttt{mixed} \\
Ceftriaxone & Tetracycline & 1.70 & \texttt{robust} & 1.07 & \texttt{mixed} \\
Cefazolin & Meropenem--vaborbactam & 1.67 & \texttt{robust} & 1.09 & \texttt{mixed} \\
Cefotaxime & Ceftazidime--avibactam & 1.66 & \texttt{robust} & 1.01 & \texttt{insufficient} \\
Minocycline & Doripenem & 0.60 & \texttt{supported} & 0.99 & \texttt{insufficient} \\
Cefotaxime & Tetracycline & 1.63 & \texttt{robust} & 1.04 & \texttt{mixed} \\
Aztreonam & Moxifloxacin & 1.59 & \texttt{robust} & 1.31 & \texttt{mixed} \\
Cefepime & Minocycline & 1.58 & \texttt{robust} & 1.41 & \texttt{mixed} \\
Cefepime & Ceftazidime--avibactam & 1.49 & \texttt{robust} & 1.33 & \texttt{mixed} \\
Ceftriaxone & Moxifloxacin & 1.47 & \texttt{robust} & 1.12 & \texttt{mixed} \\
Cefotaxime & Cefoxitin & 1.46 & \texttt{robust} & 1.07 & \texttt{mixed} \\
Aztreonam & Tigecycline & 1.38 & \texttt{robust} & 1.22 & \texttt{mixed} \\
Ceftriaxone & Imipenem--cilastatin & 1.32 & \texttt{robust} & 1.30 & \texttt{mixed} \\
Cefuroxime & Tobramycin & 1.31 & \texttt{supported} & 1.05 & \texttt{mixed} \\
Ciprofloxacin & Moxifloxacin & 1.30 & \texttt{supported} & 0.90 & \texttt{mixed} \\
Ampicillin & Moxifloxacin & 1.26 & \texttt{robust} & 1.02 & \texttt{mixed} \\
Aztreonam & Ceftazidime--avibactam & 1.25 & \texttt{supported} & 1.22 & \texttt{mixed} \\
Ceftriaxone & Cefazolin & 0.80 & \texttt{robust} & 0.99 & \texttt{mixed} \\
Levofloxacin & Aztreonam & 0.83 & \texttt{robust} & 0.98 & \texttt{mixed} \\
Cefazolin & Aztreonam & 1.19 & \texttt{robust} & 1.02 & \texttt{mixed} \\
Sulfamethoxazole--trimethoprim & Cefepime & 0.86 & \texttt{robust} & 0.98 & \texttt{mixed} \\
Ciprofloxacin & Cefepime & 0.86 & \texttt{supported} & 1.00 & \texttt{insufficient} \\
Ciprofloxacin & Aztreonam & 0.88 & \texttt{supported} & 0.93 & \texttt{mixed} \\
Ceftriaxone & Ampicillin--sulbactam & 0.88 & \texttt{supported} & 0.88 & \texttt{mixed} \\
Levofloxacin & Ceftriaxone & 0.90 & \texttt{robust} & 0.98 & \texttt{mixed} \\
Ceftazidime & Ertapenem & 0.91 & \texttt{robust} & 0.96 & \texttt{mixed} \\
Ceftriaxone & Aztreonam & 0.93 & \texttt{supported} & 0.91 & \te
```

## 24 November, later: Supplementary table made clearer (author: "columns are confusing ... concise, precise and very clear")
Backups: backup/blood_numbers_before_table.py, backup/supplementary_before_table.tex. Template in blood_numbers.py; same numbers.
- One table split into two (the combined table no longer fitted one page):
  Supplementary Table 19 (tab:supp-blood): counts only, column "Pairs", two blocks "Blood cultures" and "Primary analysis",
  rows: Retained / Validated / robust / supported / Same direction as in the primary analysis / Opposite direction to the primary
  analysis / Retained but not validated in the primary analysis / Not retained in the primary analysis; Validated / Retained in
  blood cultures / Validated there in the same direction / Not retained in blood cultures: sample-size criteria not met /
  ... removed by the co-testing screen.
  Supplementary Table 20 (tab:supp-blood-pairs, new): the 37 differing pairs. Columns were "Upstream | Downstream | ER, blood |
  Label, blood | ER, all | Label, all"; now grouped headers "Blood cultures" (ER, Label) and "Primary analysis" (ER, Label) over
  "Upstream antibiotic (A) | Downstream antibiotic (B)"; group rows "Opposite direction in the primary analysis (9)" and
  "Retained but not validated in the primary analysis (28)". Caption defines ER > 1 / < 1 in words.
- Note text shortened: "Because urine made up ... mainly reflect urinary cultures. This post hoc analysis repeated ..." ->
  "Urine made up 93.5% of the culture episodes, so the primary analysis mainly reflects urinary cultures. To test whether the
  patterns differ in blood cultures, screening and validation were repeated on the 9,205 blood-culture episodes alone, ..."
- Later Supplementary Tables are renumbered by one more (all references are \ref, none typed).
Checks: hand_numbers OK; post_run: 8 documents, 0 errors, 0 overfull lines or pages; real journal class build done.

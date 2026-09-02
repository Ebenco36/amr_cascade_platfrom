"""Generate a native Keynote presentation for the AMR Cascade Platform talk.

This script uses Keynote via AppleScript to create a `.key` deck directly on macOS.
It is intentionally anchored to the current repository outputs so the results slides
only reference figures that actually exist.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "AMR_Cascade_Platform_Presentation.key"
THEME_NAME = "Basic Black"


@dataclass(frozen=True)
class Slide:
    title: str
    bullets: tuple[str, ...] = ()
    layout: str = "Title & Bullets"
    body_text: str | None = None
    image_path: Path | None = None
    caption_lines: tuple[str, ...] = ()
    footer: str | None = None


def _as_applescript_text(text: str) -> str:
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    parts = text.split("\n")
    if not parts:
        return '""'
    quoted = [f'"{part}"' for part in parts]
    return " & return & ".join(quoted)


def _body_from_slide(slide: Slide) -> str | None:
    if slide.body_text:
        return slide.body_text
    if slide.bullets:
        return "\n".join(slide.bullets)
    return None


def build_slide_plan() -> list[Slide]:
    figures = PROJECT_ROOT / "outputs" / "figures"

    return [
        Slide(
            title="The AMR Cascade Platform",
            layout="Title",
            body_text=(
                "Quantifying Selection Bias and Testing Behavior in Antimicrobial Susceptibility Data\n"
                "Awotoro Ebenezer and collaborators"
            ),
        ),
        Slide(
            title="AMR Is Already a Major Global Threat",
            bullets=(
                "2019 direct bacterial AMR deaths: 1.27 million",
                "2021 direct AMR deaths estimate: 1.14 million",
                "Projected 2025-2050 cumulative deaths: 39 million",
                "Surveillance quality matters because stewardship and policy depend on these numbers",
            ),
        ),
        Slide(
            title="AST Is Not a Neutral Denominator",
            bullets=(
                "Resistance rate looks simple: resistant / tested",
                "The hidden assumption is that tested isolates are representative",
                "In reality, downstream drugs are often tested only after upstream concern",
                "That makes the observation process itself part of the data-generating mechanism",
            ),
        ),
        Slide(
            title="The Cascade Effect",
            bullets=(
                "We treat testing behavior as the outcome of interest",
                "Upstream result on drug j can change whether downstream drug k is tested",
                "The platform measures this change directly",
                "This reframes AST from a passive label table into an observation process",
            ),
        ),
        Slide(
            title="Study Objectives",
            bullets=(
                "Quantify directed testing cascades using conditional probabilities and escalation ratios",
                "Test whether cascade associations persist after measured adjustment",
                "Predict downstream testing behavior across sites",
                "Estimate how selective testing distorts naive resistance prevalence",
            ),
        ),
        Slide(
            title="Three Sites, Two Environments, One Layered Pipeline",
            bullets=(
                "Sites: armd, armd_ecuh, armd_utsw",
                "Local verification: sampled development, iteration, figure generation",
                "Production run: full-scale final estimates",
                "Data layers: Raw/Sample -> Bronze -> Silver -> Harmonized -> Gold -> Artifacts -> Outputs",
            ),
        ),
        Slide(
            title="How Raw Tables Become an Opportunity Space",
            bullets=(
                "Bronze: canonical site-specific ingestion",
                "Silver: null normalization, key normalization, duplicate handling",
                "Harmonized: aligned shared-domain schema across sites",
                "Gold: culture_episodes, culture_drug_episodes, testing_matrix, eligible_pairs, drug_pair_episodes",
            ),
        ),
        Slide(
            title="Eligibility Defines the Denominator",
            bullets=(
                "Intrinsic resistance is handled explicitly",
                "Biologically meaningless organism-drug pairs are removed from the opportunity space",
                "This separates structural omission from behavioral non-testing",
                "E_ij = 1 - I_ij",
            ),
        ),
        Slide(
            title="Primary Cascade Metric: Escalation Ratio",
            bullets=(
                "ER(j->k) = P(test k | resistant j) / P(test k | susceptible j)",
                "ER > 1 indicates escalation",
                "Current support thresholds: minimum total support 25, branch support 5",
                "Downstream pair must be eligible",
            ),
        ),
        Slide(
            title="Adjusted Cascade Model",
            bullets=(
                "Outcome: downstream tested or not",
                "Main exposure: upstream resistant versus susceptible",
                "Adjustment set: ICU, prior antibiotics, calendar time, age, sex, prior same-organism history, comorbidity count, source site",
                "Interpretation boundary: robustness of association, not causality",
            ),
        ),
        Slide(
            title="Leakage Prevention Is Enforced in Code",
            bullets=(
                "Labs restricted to pre-culture rows only",
                "Ward and care-setting context must be at or before culture time",
                "Prior history uses non-negative pre-culture lags only",
                "Post-culture information is excluded from predictive features",
            ),
        ),
        Slide(
            title="Predictive Modeling Strategy",
            bullets=(
                "Prediction target: probability of downstream testing",
                "Models: Logistic Regression, Random Forest, XGBoost, Neural Network",
                "Primary cross-site split: train armd, validate armd_ecuh, test armd_utsw",
                "Threshold is selected on validation and frozen before test evaluation",
            ),
        ),
        Slide(
            title="Making Surveillance Bias Visible",
            bullets=(
                "Eligible organism-drug pairs define the denominator",
                "Model-free lower and upper prevalence bounds are computed explicitly",
                "Cascade-trigger fraction and observed enrichment describe how testing is selected",
                "MNAR sensitivity curves show how eligible-scale prevalence changes under explicit assumptions",
            ),
        ),
        Slide(
            title="Validated E. coli Cascades Survive the Defensibility Filter",
            layout="Title Only",
            image_path=figures / "combined" / "organisms" / "escherichia_coli" / "figure_validated_primary_cascade_forest.png",
            caption_lines=(
                "Current validated E. coli edges: 23 robust, 28 supported",
                "Examples: Cefepime -> Imipenem, Ceftazidime -> Imipenem, Ceftriaxone -> Amikacin",
                "The primary validated result is the subset that survives permutation, bootstrap, and replication checks",
                "Marker fill shows downstream AWaRe group; marker outline shows validation confidence",
            ),
        ),
        Slide(
            title="Validated Cascades Often Move Up the AWaRe Ladder",
            layout="Title Only",
            image_path=figures / "combined" / "organisms" / "escherichia_coli" / "figure_aware_transition_heatmap.png",
            caption_lines=(
                "Direction summary: upward 20 edges, lateral 18, downward 7",
                "The most common validated upward transition is Access -> Watch (17 edges)",
                "Support-weighted mean ER for Access -> Watch transitions: 2.36",
                "AWaRe is used as a secondary interpretation layer, not as the raw ER definition",
            ),
        ),
        Slide(
            title="The Cascade Has Network Structure",
            layout="Title Only",
            image_path=figures / "combined" / "organisms" / "escherichia_coli" / "figure_cascade_network_validated.png",
            caption_lines=(
                "The testing process forms a directed graph rather than isolated pair effects",
                "Downstream broad-spectrum agents accumulate incoming pressure from many upstream drugs",
                "This justifies treating selective testing as a structured observation process",
            ),
        ),
        Slide(
            title="Cross-Site Consistency in E. coli",
            layout="Title Only",
            image_path=figures / "combined" / "organisms" / "escherichia_coli" / "figure_site_vs_combined_summary.png",
            caption_lines=(
                "Retained edge counts: armd 163, armd_utsw 187, armd_ecuh 259, combined 332",
                "Exact topology differs by site, but the existence of cascade behavior is not local noise",
                "This supports cross-site consistency of the main phenomenon",
            ),
        ),
        Slide(
            title="Selective Testing Changes Naive Surveillance",
            layout="Title Only",
            image_path=figures / "combined" / "organisms" / "escherichia_coli" / "figure_prevalence_shift_forest.png",
            caption_lines=(
                "Prevalence output reports naive tested-only prevalence, eligible-denominator bounds, and MNAR sensitivity summaries",
                "Cascade-trigger fraction and observed enrichment show how much the tested denominator is shaped by validated upstream triggers",
                "MNAR lambda curves show how denominator-aware prevalence moves under explicit assumptions about unknown binary outcomes",
                "These are visibility-aware sensitivity summaries, not corrected true prevalence estimates",
            ),
        ),
        Slide(
            title="Prevalence Shift Is an Assumption-Visible Summary",
            bullets=(
                "No inverse-probability weighting is used to estimate resistance prevalence",
                "MNAR lambda is reported explicitly so the missing-outcome assumption remains visible",
                "Site-specific prevalence-shift statements require the full production run",
                "The claim is visible surveillance distortion, not recovered ground truth",
            ),
        ),
        Slide(
            title="Downstream Testing Behavior Is Learnable Across Sites",
            layout="Title Only",
            image_path=figures / "combined" / "organisms" / "escherichia_coli" / "figure_model_metrics_comparison.png",
            caption_lines=(
                "Cross-site test-set results, full feature set, combined E. coli",
                "Random Forest: ROC-AUC 0.736, PR-AUC 0.0354, Brier 0.181",
                "XGBoost: ROC-AUC 0.733, PR-AUC 0.0346, Brier 0.183",
                "Logistic Regression remains a strong baseline; neural network is not a headline model on the current sample",
            ),
        ),
        Slide(
            title="Threshold Selection Matters",
            layout="Title Only",
            image_path=figures / "combined" / "organisms" / "escherichia_coli" / "figure_model_threshold_analysis.png",
            caption_lines=(
                "Operating thresholds are selected on validation and frozen before test",
                "This keeps decision metrics honest and reproducible",
                "The platform reports both ranking metrics and threshold-dependent operating behavior",
            ),
        ),
        Slide(
            title="Discrimination and Calibration Answer Different Questions",
            layout="Title Only",
            image_path=figures / "combined" / "organisms" / "escherichia_coli" / "figure_model_calibration.png",
            caption_lines=(
                "ROC and PR summarize ranking performance",
                "Calibration and Brier score summarize probability quality",
                "This matters if predictions are used as probabilities rather than only for ranking",
            ),
        ),
        Slide(
            title="Klebsiella pneumoniae Reinforces the Main Story",
            layout="Title Only",
            image_path=figures / "combined" / "organisms" / "klebsiella_pneumoniae" / "figure_primary_cascade_forest.png",
            caption_lines=(
                "Current retained edges in combined K. pneumoniae: 299",
                "Tigecycline -> Ampicillin/Sulbactam: ER 43.67",
                "Gentamicin -> Meropenem/Vaborbactam: ER 14.52, adjusted OR 2.60",
                "A different organism shows the same fundamental observation-process phenomenon",
            ),
        ),
        Slide(
            title="Staph aureus MRSA Is Detectable but Sparse",
            layout="Title Only",
            image_path=figures / "combined" / "organisms" / "staph_aureus_mrsa" / "figure_primary_cascade_forest.png",
            caption_lines=(
                "Current retained edges in combined Staph aureus MRSA: 7",
                "TMP/SMX -> Tetracycline: ER 7.20",
                "Tetracycline -> Vancomycin: ER 5.17",
                "Useful as a secondary illustration, but not the flagship organism on the current sample outputs",
            ),
        ),
        Slide(
            title="Implied Susceptibility Must Stay Sensitivity-Only",
            bullets=(
                "Current pooled armd comparison: primary edge count 3, sensitivity edge count 162",
                "Shared edges: 2; sensitivity-only edges: 160",
                "Implied susceptibility materially changes the apparent cascade structure",
                "That is why the primary analysis stays restricted to directly observed AST",
            ),
        ),
        Slide(
            title="Interpretation Boundaries",
            bullets=(
                "We quantify association, not causation",
                "Adjusted odds ratios are robustness estimates, not causal effects",
                "Prevalence-shift curves show explicit-assumption surveillance consequences; they do not recover true prevalence",
                "Current local outputs are sampled verification outputs, not final production manuscript estimates",
            ),
        ),
        Slide(
            title="A Defensible, Auditable Pipeline",
            bullets=(
                "Staged data contracts from raw inputs to publication outputs",
                "Preflight checks, report manifests, and scientific audit artifacts",
                "Temporal leakage guards are implemented in code and covered by tests",
                "The same logic runs locally on samples and in production for final estimates",
            ),
        ),
        Slide(
            title="What We Have Shown",
            bullets=(
                "AST data are generated by a structured observation process",
                "Directed testing cascades are measurable and often strong",
                "Selective testing can materially distort naive resistance surveillance",
                "Downstream testing behavior is predictable across sites",
                "Observed-AST primary analyses and implied-susceptibility sensitivity analyses must remain separate",
            ),
        ),
        Slide(
            title="Thank You",
            layout="Title",
            body_text=(
                "Questions?\n"
                "Data partners: armd, armd_ecuh, armd_utsw"
            ),
        ),
    ]


def build_applescript(slides: list[Slide], output_path: Path) -> str:
    lines: list[str] = []
    lines.append(f'set outPath to POSIX file "{output_path}"')
    lines.append('tell application "Keynote"')
    lines.append("    activate")
    lines.append(f'    set d to make new document with properties {{document theme:theme "{THEME_NAME}"}}')

    first = slides[0]
    lines.append("    tell slide 1 of d")
    lines.append(f'        set base layout to master slide "{first.layout}" of d')
    lines.append(f"        set object text of default title item to {_as_applescript_text(first.title)}")
    first_body = _body_from_slide(first)
    if first_body:
        lines.append("        try")
        lines.append(f"            set object text of default body item to {_as_applescript_text(first_body)}")
        lines.append("        end try")
    if first.footer:
        lines.extend(_textbox_lines(first.footer, 80, 960, 1760, 40, indent="        "))
    lines.append("    end tell")

    for slide in slides[1:]:
        lines.append(
            f'    set newSlide to make new slide at end of slides of d with properties {{base layout:master slide "{slide.layout}" of d}}'
        )
        lines.append("    tell newSlide")
        lines.append(f"        set object text of default title item to {_as_applescript_text(slide.title)}")
        body = _body_from_slide(slide)
        if body and slide.layout != "Title Only":
            lines.append("        try")
            lines.append(f"            set object text of default body item to {_as_applescript_text(body)}")
            lines.append("        end try")
        if slide.image_path:
            lines.extend(_image_lines(slide.image_path, indent="        "))
        if slide.caption_lines:
            caption_text = "\n".join(slide.caption_lines)
            lines.extend(_textbox_lines(caption_text, 1210, 165, 620, 700, indent="        "))
        if slide.footer:
            lines.extend(_textbox_lines(slide.footer, 70, 1000, 1780, 40, indent="        "))
        lines.append("    end tell")

    lines.append("    save d in outPath")
    lines.append("    close d")
    lines.append("end tell")
    return "\n".join(lines)


def _image_lines(image_path: Path, indent: str = "") -> list[str]:
    path_text = str(image_path)
    return [
        f'{indent}set imgFile to POSIX file "{path_text}"',
        f"{indent}make new image with properties {{file:imgFile, position:{{55, 125}}, width:1120, height:630}}",
    ]


def _textbox_lines(
    text: str,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    indent: str = "",
) -> list[str]:
    return [
        f"{indent}make new text item with properties {{object text:{_as_applescript_text(text)}, position:{{{x}, {y}}}, width:{width}, height:{height}}}",
    ]


def create_keynote_deck(output_path: Path) -> None:
    slides = build_slide_plan()
    applescript = build_applescript(slides, output_path)
    if output_path.exists():
        output_path.unlink()
    subprocess.run(["osascript"], input=applescript, text=True, check=True)


def main() -> None:
    output_path = DEFAULT_OUTPUT
    create_keynote_deck(output_path)
    print(output_path)


if __name__ == "__main__":
    main()

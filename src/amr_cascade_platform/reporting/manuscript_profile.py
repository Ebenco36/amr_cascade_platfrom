"""Union of the outputs cited by the two current E. coli manuscripts.

This is a reporting/execution scope, not a change to an estimator or a gate.
The citation-contract tests detect drift when the local manuscripts change.
"""

from amr_cascade_platform.reporting.paper1_profile import (
    PAPER1_FIGURE_GROUPS,
    PAPER1_REPORT_FIGURES,
)

PUBLICATION_PROFILES = ("all", "paper1", "manuscripts")
MANUSCRIPT_FIGURE_GROUPS = PAPER1_FIGURE_GROUPS + (
    "prevalence_shift_forest", "cascade_consequence_summary",
    "prevalence_shift_curves", "mnar_prevalence_shift_distribution",
)
PAPER2_REPORT_FIGURES = (
    "figure_observation_coverage_by_site",
    "figure_prevalence_shift_forest",
    "figure_kappa_ranked",
    "figure_cascade_vs_independent_dumbbell",
    "figure_surveillance_sensitivity_map",
    "figure_mnar_prevalence_shift_distribution",
    "figure_prevalence_mnar_sensitivity_curves_stable",
    "figure_prevalence_mnar_sensitivity_curves_moderate",
    "figure_prevalence_mnar_sensitivity_curves_sensitive",
    "figure_antibiogram",
)
MANUSCRIPT_REPORT_FIGURES = PAPER1_REPORT_FIGURES + PAPER2_REPORT_FIGURES
STATIC_ONLY_FIGURES = frozenset(
    stem for stem in PAPER2_REPORT_FIGURES
    if stem.startswith("figure_prevalence_mnar_sensitivity_curves_")
)


def report_figures(publication: str) -> tuple[str, ...]:
    if publication == "paper1":
        return PAPER1_REPORT_FIGURES
    if publication == "manuscripts":
        return MANUSCRIPT_REPORT_FIGURES
    raise ValueError(f"No named manuscript contract for {publication!r}")


def required_exports(publication: str, formats: tuple[str, ...]) -> tuple[str, ...]:
    # The MNAR small multiples have a native Matplotlib exporter, not HTML.
    return tuple(
        f"{stem}.{fmt}" for stem in report_figures(publication) for fmt in formats
        if not (stem in STATIC_ONLY_FIGURES and fmt == "html")
    )

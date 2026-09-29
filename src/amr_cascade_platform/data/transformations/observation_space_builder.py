"""Build observed culture-drug episode tables."""

from __future__ import annotations

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.utils.antibiotic_names import (
    DEFAULT_AST_ENTITY_TYPE,
    classify_ast_entity,
    normalize_antibiotic_label,
)

LEDGER_COLUMNS = (
    "organism_rows",
    "no_result_rows",
    "other_non_interpretive_rows",
    "excluded_assay_rows",
    # Of excluded_assay_rows: composite labels (COMPOSITE_AST) that cannot be
    # attributed to one agent, reported separately in the exclusion flow.
    "composite_label_rows",
    "interpretable_rows",
    "identical_rows_collapsed",
)
UNRECOGNISED_LABEL_COLUMNS = ["source_site", "antibiotic", "canonical_label", "rows"]


class ObservationSpaceBuilder:
    """Build one row per observed culture-drug result."""

    def __init__(self, settings: Settings, reference_labels: frozenset[str] | None = None) -> None:
        self._settings = settings
        # Canonical labels of the antibiotic classification reference
        # (core.utils.antibiotic_reference); None skips the unrecognised-label audit.
        self._reference_labels = reference_labels
        # Rows removed as non-therapeutic entries by the last build, per site and
        # label (source_site, antibiotic, entity_type, rows) -- for the gold metadata.
        self.excluded_entities = pd.DataFrame(columns=["source_site", "antibiotic", "entity_type", "rows"])
        # Rows the last build kept under a label that is neither a listed special
        # entity nor in the classification reference, per site and label; None when
        # the audit was skipped.
        self.unrecognised_labels: pd.DataFrame | None = None

    def build(self, cohort: pd.DataFrame, organism_list: tuple[str, ...] | None = None) -> pd.DataFrame:
        return self.build_with_ledger(cohort, organism_list)[0]

    def build_with_ledger(
        self, cohort: pd.DataFrame, organism_list: tuple[str, ...] | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The observed results and, per source site, how many rows each rule removed.

        The ledger counts, in order: the organism's rows; rows with no result;
        rows whose result is not an interpretation (e.g. INCONCLUSIVE); entries
        that are not a therapeutic antibacterial susceptibility result
        (``excluded_assay_rows``: screening, phenotypic-confirmation and
        resistance-mechanism tests, antifungal and antimycobacterial agents,
        rows without an antibiotic label -- see gold.primary_ast_entity_types);
        the interpretable rows left; and those of them identical to another row
        on every observed-result column.
        """
        frame = cohort
        if organism_list:
            frame = frame[frame["organism"].isin(organism_list)]
        sites = frame["source_site"].astype("string").fillna("unknown")
        ledger = pd.DataFrame({"organism_rows": sites.value_counts()})

        result = frame["susceptibility"]
        interpretable = result.isin(self._settings.gold.observed_result_values)
        blank = result.isna() | result.astype("string").str.strip().fillna("").eq("")
        ledger["no_result_rows"] = sites[blank & ~interpretable].value_counts()
        ledger["other_non_interpretive_rows"] = sites[~blank & ~interpretable].value_counts()
        frame = frame[interpretable]

        # Only therapeutic antibacterial susceptibility results enter the observed
        # layer. Screening, confirmatory and mechanism tests are not AST results for
        # the drug they are named after (a cefoxitin screen detects methicillin
        # resistance; inducible clindamycin is a D-zone test), so they are removed
        # here, before any alias could merge them into the drug's own results.
        excluded = {normalize_antibiotic_label(label) for label in self._settings.gold.excluded_assay_labels}
        primary = set(self._settings.gold.primary_ast_entity_types)
        entity_of = {label: classify_ast_entity(label) for label in frame["antibiotic"].dropna().unique()}
        entity_of = {label: entity or "MISSING_LABEL" for label, entity in entity_of.items()}
        assay_labels = [
            label for label, entity in entity_of.items()
            if entity not in primary or normalize_antibiotic_label(label) in excluded
        ]
        assay = frame["antibiotic"].isin(assay_labels) | frame["antibiotic"].isna()
        ledger["excluded_assay_rows"] = sites[interpretable][assay.to_numpy()].value_counts()
        removed = frame.loc[assay, ["source_site", "antibiotic"]].copy()
        removed["entity_type"] = removed["antibiotic"].map(entity_of).fillna("MISSING_LABEL")
        ledger["composite_label_rows"] = (
            removed.loc[removed["entity_type"].eq("COMPOSITE_AST"), "source_site"]
            .astype("string").fillna("unknown").value_counts()
        )
        self.excluded_entities = (
            removed.astype("string").fillna("<missing>")
            .value_counts().rename("rows").reset_index()
            .loc[:, ["source_site", "antibiotic", "entity_type", "rows"]]
            .sort_values(["source_site", "rows"], ascending=[True, False], ignore_index=True)
        )
        frame = frame[~assay]
        self.unrecognised_labels = self._unrecognised_labels(frame, entity_of)
        ledger["interpretable_rows"] = frame["source_site"].astype("string").fillna("unknown").value_counts()

        observed = (
            frame.loc[:, list(self._settings.gold.culture_drug_columns)]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        kept = observed["source_site"].astype("string").fillna("unknown").value_counts()
        ledger["identical_rows_collapsed"] = ledger["interpretable_rows"].sub(kept, fill_value=0)
        observed["was_tested"] = 1
        observed["observation_layer"] = "observed_ast"
        ledger = ledger.reindex(columns=list(LEDGER_COLUMNS)).fillna(0).astype("int64")
        ledger.index.name = "source_site"
        return observed, ledger

    def _unrecognised_labels(self, kept: pd.DataFrame, entity_of: dict[str, str]) -> pd.DataFrame | None:
        """Kept rows whose label is an antibacterial only by default, per site and label.

        classify_ast_entity types every label it does not list as an ordinary
        antibacterial, so a new screening or confirmatory assay in a future extract
        would enter the observed layer unnoticed. Labels that are also absent from
        the antibiotic classification reference are recorded here (and logged by the
        gold build) rather than guessed at; they stay in the observed layer.
        """
        if self._reference_labels is None:
            return None
        unknown = {
            label
            for label, entity in entity_of.items()
            if entity == DEFAULT_AST_ENTITY_TYPE and normalize_antibiotic_label(label) not in self._reference_labels
        }
        rows = kept.loc[kept["antibiotic"].isin(unknown), ["source_site", "antibiotic"]] if unknown else kept.head(0)
        if rows.empty:
            return pd.DataFrame(columns=UNRECOGNISED_LABEL_COLUMNS)
        table = rows.astype("string").fillna("<missing>").value_counts().rename("rows").reset_index()
        table["canonical_label"] = table["antibiotic"].map(normalize_antibiotic_label)
        return table.loc[:, UNRECOGNISED_LABEL_COLUMNS].sort_values(
            ["source_site", "rows"], ascending=[True, False], ignore_index=True
        )

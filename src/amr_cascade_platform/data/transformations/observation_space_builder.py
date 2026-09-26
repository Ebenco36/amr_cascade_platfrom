"""Build observed culture-drug episode tables."""

from __future__ import annotations

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.utils.antibiotic_names import normalize_antibiotic_label

LEDGER_COLUMNS = (
    "organism_rows",
    "no_result_rows",
    "other_non_interpretive_rows",
    "excluded_assay_rows",
    "interpretable_rows",
    "identical_rows_collapsed",
)


class ObservationSpaceBuilder:
    """Build one row per observed culture-drug result."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(self, cohort: pd.DataFrame, organism_list: tuple[str, ...] | None = None) -> pd.DataFrame:
        return self.build_with_ledger(cohort, organism_list)[0]

    def build_with_ledger(
        self, cohort: pd.DataFrame, organism_list: tuple[str, ...] | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The observed results and, per source site, how many rows each rule removed.

        The ledger counts, in order: the organism's rows; rows with no result;
        rows whose result is not an interpretation (e.g. INCONCLUSIVE); ESBL
        phenotypic confirmation assays; the interpretable rows left; and those
        of them identical to another row on every observed-result column.
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

        excluded = {normalize_antibiotic_label(label) for label in self._settings.gold.excluded_assay_labels}
        assay_labels = [
            label for label in frame["antibiotic"].dropna().unique() if normalize_antibiotic_label(label) in excluded
        ]
        assay = frame["antibiotic"].isin(assay_labels)
        ledger["excluded_assay_rows"] = sites[interpretable][assay.to_numpy()].value_counts()
        frame = frame[~assay]
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

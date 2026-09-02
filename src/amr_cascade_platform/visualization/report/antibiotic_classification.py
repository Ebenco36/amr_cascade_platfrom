"""WHO AWaRe antibiotic classification helpers for reporting visuals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.utils.antibiotic_names import (
    ANTIBIOTIC_NAME_ALIASES,
    normalize_antibiotic_label,
)


@dataclass(frozen=True)
class AntibioticClassification:
    """Resolved metadata for one antibiotic label."""

    label: str
    aware_category: str
    broad_class: str
    antibiotic_class: str
    color: str
    display_label: str
    abbreviation: str = ""
    reference_labels: tuple[str, ...] = ()


class AntibioticClassificationResolver:
    """Resolve antibiotic labels to WHO AWaRe categories and colors."""

    CATEGORY_ORDER = ("Access", "Watch", "Reserve", "Not Set", "Unclassified")
    CATEGORY_COLORS = {
        "Access": "#2E8B57",
        "Watch": "#E69F00",
        "Reserve": "#C0392B",
        "Not Set": "#7F8C8D",
        "Unclassified": "#94A3B8",
    }
    CATEGORY_RANKS = {
        "Access": 1,
        "Watch": 2,
        "Reserve": 3,
        "Not Set": 0,
        "Unclassified": 0,
    }
    # Shared with amr_cascade_platform.core.utils.antibiotic_names so that every
    # analyzer grouping or joining on antibiotic name resolves the same aliases,
    # not just reporting code. Do not fork this table — edit the shared source.
    _ALIASES = ANTIBIOTIC_NAME_ALIASES

    def __init__(self, classification_path: Path) -> None:
        self._classification_path = classification_path
        self._lookup = self._build_lookup()

    def resolve(self, label: str | None) -> AntibioticClassification:
        normalized = self.normalize_label(label)
        metadata = self._lookup.get(normalized)
        if metadata is None:
            fallback = normalized or str(label or "Unknown")
            return AntibioticClassification(
                label=fallback,
                aware_category="Unclassified",
                broad_class="Unclassified",
                antibiotic_class="Unclassified",
                color=self.CATEGORY_COLORS["Unclassified"],
                display_label=fallback,
                abbreviation=self._make_abbreviation(fallback),
                reference_labels=(fallback.upper(),),
            )
        aware_category = metadata["aware_category"]
        return AntibioticClassification(
            label=str(label or "Unknown"),
            aware_category=aware_category,
            broad_class=metadata["broad_class"],
            antibiotic_class=metadata["antibiotic_class"],
            color=self.CATEGORY_COLORS.get(aware_category, self.CATEGORY_COLORS["Unclassified"]),
            display_label=metadata["display_label"],
            abbreviation=metadata.get("abbreviation", self._make_abbreviation(str(label or ""))),
            reference_labels=tuple(metadata.get("reference_labels", ())),
        )

    def get_abbreviation(self, label: str | None) -> str:
        """Return the standard abbreviation for an antibiotic label."""
        resolved = self.resolve(label)
        return resolved.abbreviation or self._make_abbreviation(str(label or ""))

    def add_metadata(self, dataframe: pd.DataFrame, antibiotic_column: str, prefix: str) -> pd.DataFrame:
        """Attach AWaRe metadata columns for one antibiotic column."""
        annotated = dataframe.copy()
        classifications = annotated[antibiotic_column].map(self.resolve)
        annotated[f"{prefix}_aware_category"] = classifications.map(lambda item: item.aware_category)
        annotated[f"{prefix}_broad_class"] = classifications.map(lambda item: item.broad_class)
        annotated[f"{prefix}_antibiotic_class"] = classifications.map(lambda item: item.antibiotic_class)
        annotated[f"{prefix}_aware_color"] = classifications.map(lambda item: item.color)
        annotated[f"{prefix}_display_label"] = classifications.map(lambda item: item.display_label)
        annotated[f"{prefix}_abbreviation"] = classifications.map(lambda item: item.abbreviation)
        return annotated

    def abbreviation_table(self, antibiotic_labels: list[str]) -> pd.DataFrame:
        """Return abbreviation metadata for supplementary figure decoding."""
        grouped: dict[str, dict[str, object]] = {}
        for label in antibiotic_labels:
            resolved = self.resolve(label)
            key = resolved.abbreviation
            if not key:
                continue
            row = grouped.setdefault(
                key,
                {
                    "abbreviation": resolved.abbreviation,
                    "full_name": resolved.display_label,
                    "aware_category": resolved.aware_category,
                    "broad_class": resolved.broad_class,
                    "antibiotic_class": resolved.antibiotic_class,
                    "source_labels": set(),
                    "canonical_labels": set(),
                    "reference_labels": set(),
                },
            )
            source = str(label or "").strip().upper()
            if source:
                row["source_labels"].add(source)
            row["canonical_labels"].add(resolved.display_label)
            row["reference_labels"].update(resolved.reference_labels)

        rows = []
        for row in grouped.values():
            source_labels = sorted(row.pop("source_labels"))
            canonical_labels = sorted(row.pop("canonical_labels"))
            reference_labels = sorted(row.pop("reference_labels"))
            rows.append(
                {
                    **row,
                    "source_labels": "; ".join(source_labels),
                    "canonical_labels": "; ".join(canonical_labels),
                    "reference_labels": "; ".join(reference_labels),
                    "source_label_count": len(source_labels),
                    "abbreviation_collision": len(canonical_labels) > 1,
                }
            )
        df = pd.DataFrame(rows).sort_values("abbreviation").reset_index(drop=True)
        return df

    def abbreviation_axis_label_map(self, antibiotic_labels: list[str]) -> dict[str, str]:
        """Return context-safe axis labels without merging distinct drugs."""
        resolved_by_label = {str(label): self.resolve(label) for label in antibiotic_labels}
        displays_by_abbrev: dict[str, set[str]] = {}
        for resolved in resolved_by_label.values():
            displays_by_abbrev.setdefault(resolved.abbreviation, set()).add(resolved.display_label)

        axis_labels: dict[str, str] = {}
        for label, resolved in resolved_by_label.items():
            if len(displays_by_abbrev.get(resolved.abbreviation, set())) <= 1:
                axis_labels[label] = resolved.abbreviation
            else:
                axis_labels[label] = f"{resolved.abbreviation} ({self._compact_label(resolved.display_label)})"
        return axis_labels

    def canonical_display_label(self, value: str | None) -> str:
        normalized = self.normalize_label(value)
        metadata = self._lookup.get(normalized)
        if metadata is None:
            return str(value or "Unknown")
        return metadata["display_label"]

    @classmethod
    def normalize_label(cls, value: str | None) -> str:
        return normalize_antibiotic_label(value)

    def category_sort_key(self, value: str | None) -> int:
        try:
            return self.CATEGORY_ORDER.index(str(value))
        except ValueError:
            return len(self.CATEGORY_ORDER)

    def category_rank(self, value: str | None) -> int:
        return int(self.CATEGORY_RANKS.get(str(value or "Unclassified"), 0))

    def transition_step(self, upstream_category: str | None, downstream_category: str | None) -> int | None:
        upstream_rank = self.category_rank(upstream_category)
        downstream_rank = self.category_rank(downstream_category)
        if upstream_rank == 0 or downstream_rank == 0:
            return None
        return downstream_rank - upstream_rank

    def transition_label(self, upstream_category: str | None, downstream_category: str | None) -> str:
        upstream = str(upstream_category or "Unclassified")
        downstream = str(downstream_category or "Unclassified")
        return f"{upstream} -> {downstream}"

    def transition_direction(self, upstream_category: str | None, downstream_category: str | None) -> str:
        step = self.transition_step(upstream_category, downstream_category)
        if step is None:
            return "unclassified"
        if step > 0:
            return "upward"
        if step < 0:
            return "downward"
        return "lateral"

    @staticmethod
    def with_alpha(hex_color: str, alpha: float) -> str:
        alpha_int = max(0, min(255, round(alpha * 255)))
        return f"{hex_color}{alpha_int:02x}"

    @staticmethod
    def with_alpha_rgba(hex_color: str, alpha: float) -> str:
        stripped = hex_color.lstrip("#")
        if len(stripped) != 6:
            return hex_color
        red = int(stripped[0:2], 16)
        green = int(stripped[2:4], 16)
        blue = int(stripped[4:6], 16)
        clamped_alpha = max(0.0, min(1.0, alpha))
        return f"rgba({red}, {green}, {blue}, {clamped_alpha:.3f})"

    def _build_lookup(self) -> dict[str, dict[str, object]]:
        reference = pd.read_csv(self._classification_path)
        lookup: dict[str, dict[str, object]] = {}
        for _, row in reference.iterrows():
            category = self._normalize_category(row.get("Category"))
            # Skip 'NO MATCH' WHO_Match sentinel and low-confidence fuzzy matches —
            # fall through to the row's own Antibiotic Name. A low-confidence match
            # (e.g. FUZZY(0.84) mapping "Ticarcillin/clavulanic acid" onto
            # "Amoxicillin/clavulanic Acid") is a different real drug, not a spelling
            # variant, and using it as the display label silently mislabels every
            # row for that drug in every downstream report.
            who_match_raw = str(row.get("WHO_Match") or "").strip()
            match_type = str(row.get("Match_Type") or "").strip()
            who_match_usable = (
                who_match_raw
                if who_match_raw
                and who_match_raw.upper() != "NO MATCH"
                and self._is_high_confidence_match(match_type)
                else None
            )
            raw_display = str(
                who_match_usable
                or row.get("Antibiotic Name")
                or row.get("Full Name")
                or row.get("Abbreviation")
                or "Unknown"
            ).strip().replace("_", " ").replace("-oral", "").replace("-ORAL", "")
            display_label = " ".join(raw_display.split()).upper()
            # Strip trailing " ORAL" suffix introduced when WHO_Match contains "_oral"
            if display_label.endswith(" ORAL"):
                display_label = display_label[:-5].strip()
            abbrev_raw = str(row.get("Abbreviation") or "").strip().upper()
            abbreviation = abbrev_raw if abbrev_raw and abbrev_raw != "NAN" else self._make_abbreviation(display_label)
            # Guard against NaN broad_class (e.g. CPT/Ceftaroline has no Broad Class)
            broad_class_raw = row.get("Broad Class")
            broad_class = (
                "Unclassified"
                if broad_class_raw is None or (isinstance(broad_class_raw, float) and pd.isna(broad_class_raw))
                else str(broad_class_raw).strip() or "Unclassified"
            )
            reference_labels = self._reference_labels(row, who_match_usable, display_label)
            payload = {
                "aware_category": category,
                "broad_class": broad_class,
                "antibiotic_class": str(row.get("Class", "Unclassified") or "Unclassified").strip(),
                "display_label": display_label,
                "abbreviation": abbreviation,
                "reference_labels": reference_labels,
            }
            labels = [
                row.get("Abbreviation"),
                row.get("Antibiotic Name"),
                row.get("Antibiotic Name German"),
                row.get("Full Name"),
            ]
            for label in labels:
                if label is None or pd.isna(label):
                    continue
                normalized = self.normalize_label(str(label))
                if normalized:
                    lookup[normalized] = payload
                if " - " in str(label):
                    trailing = str(label).split(" - ", maxsplit=1)[-1]
                    normalized_trailing = self.normalize_label(trailing)
                    if normalized_trailing:
                        lookup[normalized_trailing] = payload
            # Register display_label itself so re-resolution of already-canonicalized
            # names (e.g. after _canonicalize_pair_table) still resolves correctly.
            display_normalized = self.normalize_label(display_label)
            if display_normalized and display_normalized not in lookup:
                lookup[display_normalized] = payload
        for alias, canonical in self._ALIASES.items():
            canonical_payload = lookup.get(canonical)
            if canonical_payload is not None:
                labels = set(canonical_payload.get("reference_labels", ()))
                labels.update({alias, canonical})
                canonical_payload["reference_labels"] = tuple(sorted(labels))
                lookup[alias] = canonical_payload
        return lookup

    def _reference_labels(
        self,
        row: pd.Series,
        who_match_usable: str | None,
        display_label: str,
    ) -> tuple[str, ...]:
        labels: set[str] = {display_label}
        for column in ("Abbreviation", "Antibiotic Name", "Antibiotic Name German", "Full Name"):
            value = row.get(column)
            if value is None or pd.isna(value):
                continue
            raw = str(value).strip()
            if not raw:
                continue
            labels.add(raw.upper())
            if " - " in raw:
                labels.add(raw.split(" - ", maxsplit=1)[-1].strip().upper())
        if who_match_usable:
            labels.add(str(who_match_usable).strip().replace("_", " ").replace("-oral", "").replace("-ORAL", "").upper())
        return tuple(sorted(label for label in labels if label))

    @staticmethod
    def _make_abbreviation(label: str, max_len: int = 4) -> str:
        """Generate a short fallback abbreviation from a full antibiotic name."""
        clean = (
            label.replace("/", " ").replace("-", " ").replace("_", " ")
                 .replace("(", "").replace(")", "").strip().upper()
        )
        words = [w for w in clean.split() if w and w not in {"AND", "OF", "THE"}]
        if not words:
            return "UNK"
        if len(words) == 1:
            return words[0][:max_len]
        # Compound: first letters of each word, up to max_len
        initials = "".join(w[0] for w in words)
        if len(initials) <= max_len:
            return initials
        # Too many words: take first 3 letters of first word
        return words[0][:3]

    @staticmethod
    def _compact_label(label: str, max_len: int = 20) -> str:
        clean = " ".join(str(label or "").replace("/", "-").split())
        return clean if len(clean) <= max_len else f"{clean[: max_len - 3]}..."

    _FUZZY_MATCH_CONFIDENCE_FLOOR = 0.90

    @classmethod
    def _is_high_confidence_match(cls, match_type: str) -> bool:
        """PERFECT/MANUAL matches are trusted outright; FUZZY only above the floor.

        Below the floor, spot-checking the reference data found genuinely
        different drugs fuzzy-matched together (Ticarcillin/clavulanic acid to
        Amoxicillin/clavulanic Acid at 0.84; Ceftarolin to Cefazolin at 0.84;
        Apalcillin to Ampicillin at 0.80; among others), while matches at or
        above 0.90 were consistently the same drug under a spelling or
        transliteration variant. MANUAL denotes a deliberately curated entry
        (e.g. a newer FDA-approved combination added by hand), not an
        algorithmic guess, so it is trusted like PERFECT rather than scored.
        """
        if match_type in ("PERFECT", "MANUAL"):
            return True
        match = re.fullmatch(r"FUZZY\((\d+(?:\.\d+)?)\)", match_type)
        if not match:
            return False
        return float(match.group(1)) >= cls._FUZZY_MATCH_CONFIDENCE_FLOOR

    @staticmethod
    def _normalize_category(value: str | None) -> str:
        normalized = str(value or "Unclassified").strip()
        if normalized.lower() == "watch":
            return "Watch"
        if normalized.lower() == "access":
            return "Access"
        if normalized.lower() == "reserve":
            return "Reserve"
        if normalized.lower() == "not set":
            return "Not Set"
        return "Unclassified"

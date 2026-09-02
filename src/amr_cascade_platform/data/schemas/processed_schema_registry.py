"""Schema expectations for processed datasets."""

from __future__ import annotations

from amr_cascade_platform.core.config.config_models import Settings


class ProcessedSchemaRegistry:
    """Resolve required columns and key expectations for processed tables."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def required_columns_for(self, canonical_table: str) -> tuple[str, ...]:
        default = ()
        if canonical_table not in self._settings.schema.default_exempt_tables:
            default = self._settings.schema.required_columns.get("default", ())
        specific = self._settings.schema.required_columns.get(canonical_table, ())
        return tuple(dict.fromkeys([*default, *specific]))

    def required_non_null_columns_for(self, canonical_table: str) -> tuple[str, ...]:
        default = ()
        if canonical_table not in self._settings.schema.default_exempt_tables:
            default = self._settings.schema.required_non_null_columns.get("default", ())
        specific = self._settings.schema.required_non_null_columns.get(canonical_table, ())
        return tuple(dict.fromkeys([*default, *specific]))

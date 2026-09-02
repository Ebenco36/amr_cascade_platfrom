from types import SimpleNamespace

import pandas as pd

from amr_cascade_platform.features.preprocessors.comorbidity_aggregator import (
    ComorbidityAggregationRequest,
    ComorbidityAggregator,
)


def test_normalize_column_name() -> None:
    assert ComorbidityAggregator.normalize_column_name("Chronic Kidney Disease / ESRD") == (
        "comorb_chronic_kidney_disease_esrd"
    )


def test_request_validation_rejects_multiple_selection_modes() -> None:
    request = ComorbidityAggregationRequest(site="armd", top_k=20, min_count=5)
    try:
        request.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when multiple selection modes are provided.")


def test_active_unique_rows_uses_culture_minus_event_day_offsets() -> None:
    aggregator = object.__new__(ComorbidityAggregator)
    aggregator._settings = SimpleNamespace(
        platform=SimpleNamespace(
            id_columns=("anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded")
        )
    )
    keys = {
        "anon_id": "p1",
        "pat_enc_csn_id_coded": "e1",
        "order_proc_id_coded": "o1",
    }
    dataframe = pd.DataFrame(
        [
            {
                **keys,
                "comorbidity_component": "active_ongoing",
                "comorbidity_component_start_days_culture": 365,
                "comorbidity_component_end_days_culture": None,
            },
            {
                **keys,
                "comorbidity_component": "active_crosses_culture",
                "comorbidity_component_start_days_culture": 20,
                "comorbidity_component_end_days_culture": -2,
            },
            {
                **keys,
                "comorbidity_component": "future_onset",
                "comorbidity_component_start_days_culture": -5,
                "comorbidity_component_end_days_culture": None,
            },
            {
                **keys,
                "comorbidity_component": "resolved_before_culture",
                "comorbidity_component_start_days_culture": 100,
                "comorbidity_component_end_days_culture": 5,
            },
        ]
    )

    result = aggregator._active_unique_rows(dataframe)

    assert set(result["comorbidity_component"]) == {
        "active_ongoing",
        "active_crosses_culture",
    }

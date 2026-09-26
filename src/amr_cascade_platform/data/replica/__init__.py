"""Synthetic replica of the raw ARMD-family extracts for end-to-end pipeline tests.

``raw_profiler`` reads ``data/raw`` once and keeps only aggregates (file layout,
spellings, vocabularies, quantiles, fitted AST models). ``replica_generator``
writes synthetic CSVs with the same layout from that profile, so the full
pipeline can run locally on data that behaves like the real extracts without
containing any record from them.
"""

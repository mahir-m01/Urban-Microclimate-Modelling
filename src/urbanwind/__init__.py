"""
urbanwind – shared code for the urban pedestrian-wind surrogate models.

Sub-modules
-----------
data     : load_case(), case_targets(), exclusion and flag lists
splits   : load_splits(), group assignment, leakage tests
metrics  : per-case and pooled evaluation metrics
evaluate : CLI entry-point; write a row to results/runs.csv
features : case-level and pixel-level feature extraction
dl       : deep-learning data pipeline and models (Phase 2)
"""

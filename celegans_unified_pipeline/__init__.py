"""Unified C. elegans pipeline (unified_v1).

This package merges the key functional features that existed across the earlier
pipelines (v7/v10 demos + newer rigor work):

- Discovery that can be broad (optionally include home), with skip lists and optional archive extraction
- Loading BrainScanner and exported_data recordings
- Role assignment (sensory / motor / interneuron-like) without throwing away interneurons
- Pair metrics: lag scan (simple + nested), correlation, MI (continuous + discrete), discrete TE, Gaussian predictive information, MI-vs-bins, surrogate nulls + BH-FDR
- Model: trajectory prediction with contributor toggles (sensory/motor/interneurons/other) and target-history toggle, plus multi-dataset training/eval
- Plots: v10-style diagnostics and prediction overlays

The notebook `UNIFIED_celegans_pipeline.ipynb` is the intended entry point.
"""

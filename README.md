# Unified C. elegans pipeline (unified_v8)

## What this is
A single, coherent codebase that merges the key functional features into one workflow:
- file discovery
- loaders (BrainScanner + exported_data)
- preprocessing (NaN interpolation + photobleach correction + robust z-score)
- pairwise metrics (lagged correlation + MI + TE + Gaussian predictive information)
- surrogate nulls + BH-FDR for screening
- trajectory prediction (ElasticNet + MLP) with neuron-role toggles and target-history toggle
- multi-dataset training across 0..112, evaluate on a hold-out dataset

Open:
- `Final_celegans_pipeline.ipynb`

## Feature-by-feature intent and accuracy expectations

### Photobleach correction
- Removes slow drift that otherwise inflates MI/TE.
- Accuracy: improves estimator stability; may remove true low-frequency biological components if window too short.

### Lag selection
- `simple`: fast, can overestimate if used for inference.
- `nested`: slower, but produces a credible held-out `r_test` and more defensible MI/TE summaries.

### MI (continuous + discrete) and MI-vs-bins
- Discrete MI depends on binning; MI-vs-bins is a diagnostic to avoid over/under-binning.
- Continuous MI uses kNN proxy; less sensitive to bin choices, but still finite-sample biased.

### Transfer entropy (discrete plug-in)
- Directional dependence summary, but biased in small samples and sensitive to nonstationarity.
- Always interpret alongside surrogate controls and stationarity checks.

### Gaussian predictive information
- Linear/Gaussian analogue of TE computed from held-out residual variances.
- Often more stable than discrete TE on small samples; does not capture nonlinear effects.

### Screening with surrogates + BH-FDR
- Produces p-values under a circular-shift null and q-values for multiple comparisons.
- This is the primary guardrail against spurious sensory→motor hits.

### Prediction + contribution weights
- ElasticNet gives robust baseline, MLP gives nonlinear capacity.
- Contributor percentages are grouped permutation importance on held-out data: a predictive, not causal, decomposition.
- Accuracy: use held-out R²; improvements when toggling interneurons indicate genuine network predictive value.

## Overleaf / LaTeX
`overleaf_summary.tex` contains:
- all equations
- methodology and interpretation notes
- limitations


Update (unified_v2):
- Added neural contributor bar plot and separate ground-vs-model overlays.
- Fixed multi-dataset training when the chosen target neuron is absent from the reference dataset.

- Added atlas-only contributor bar plot (grouped permutation importance on atlas feature subset).

Update (unified_v4): Added marginal entropy bar plot H(source), H(target) and MI upper bound min(H) in pair report.

Update (unified_v5):
- Added pair report for top contributor neuron vs target (includes entropy bound plot).
- Added artifact-removal diagnostics plots (raw+baseline, corrected vs prepared, metric deltas raw vs filtered).

Update (unified_v6):
- Added pair_selection='best'|'worst' to choose which screened pair is deep-dived.
- Added run_pair_of_choice() for explicit user-chosen neuron pairs.
- Added TE vs lag diagnostic plot.

Update (unified_v7):
- Added TE surrogate null (circular-shift-source) with empirical p-value and histogram.

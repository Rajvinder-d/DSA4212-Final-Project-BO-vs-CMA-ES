# V4-aligned V5 results

Imported from the V5 collection. This is a **partial collection**:
279/280 trajectories completed, with nineteen complete initial-design groups.
Scalar BO failed on design 21 after 166 counted evaluations. The supplied error
is documented in [failure_note.md](failure_note.md). Do not describe this export
as a complete twenty-design confirmation.

The original exporter excludes the whole incomplete design at both checkpoints.
Thus the paired results use 19 runs per BO/random arm and 95 per CMA handler;
13 other completed trajectories from design 21 remain in the checkpoint and
incumbent exports, labelled `complete_group=False`. The collection records
83,866 search calls, 1,116 assessments and 17,724.67 s across invocations.

## Findings used in the report

- Projection/resampling MSE ratio at 100 calls: **0.744 [0.625, 0.894]**;
  at 300: **0.890 [0.819, 0.958]**. These exponentiate the mean design-level
  log10 contrast and its pointwise 95% bootstrap interval. Five CMA log losses
  are averaged within each design before inference.
- All six CMA/BO pointwise intervals favour BO at each budget. This is a
  complete-design comparison, conditional on excluding design 21; it is not a
  failure-inclusive ranking or a simultaneous confidence statement.
- At 300 calls, 6 mV attainment is 18/19 for scalar BO, 19/19 for ARD and pool
  BO, 43/95 for resampling CMA and 53/95 for projection. Random search has 0/19.
- A separately labelled post hoc handler-only sensitivity uses all twenty
  designs, since every CMA run completed. Its ratios are 0.753 [0.637, 0.895]
  and 0.892 [0.823, 0.957]. This does not recover the missing BO trajectory or
  remove possible selection bias from BO–CMA comparisons.

The rerun uses V4's prescribed scientific executor, random-stream construction,
initial-design labels and replication, with NumPy as the configured primary
backend. The intended interventions are ball projection and extending BO/random
to 300 calls. V4's Mac execution and this Windows execution remain distinct.
Shared scalar-pool and random-search 100-call losses match to numerical precision
on all nineteen included designs. Continuous BO and CMA endpoints do not all
match, so common settings are not a claim of identical cross-machine paths.

## Verification and reproduction

`validation.json` is the execution-side export audit. All twelve listed output
hashes now match. Git had normalised line endings in ten text exports; restoring
CRLF reproduced their recorded SHA-256 hashes exactly, without changing values.
`.gitattributes` preserves these bytes on later checkouts. The imported PNGs
already matched. The reference point hashes in `initial_designs.csv` were checked
against V4 construction by the execution-side exporter. Historical V4 raw point
arrays are not supplied for an independent cross-machine array comparison.

Run `python -B scripts/analyse_aligned_v5.py` from the repository root to verify
export hashes, coverage, 279 saved loss sequences, checkpoint minima, all fourteen
bootstrap contrasts and target summaries. It regenerates `report_checks.json`
and `v4_aligned_v5_endpoint_pairs.csv` under ignored `artifacts/report-checks/`
by default; `--output-dir PATH` selects another destination. These are report-stage
derivatives, not original experiment exports. Submitted evidence is preserved. No battery model or optimiser is run.

The runner, configuration and verification guard accompany this export.
See [the aligned protocol](../../docs/V5_ALIGNED.md) and
[the source comparison](../../docs/V5_ALIGNMENT_REVIEW.md). The latter records
why matching parameter values alone was insufficient. Original V5 evidence in
`results/v5/` is preserved and is not pooled into this report's new estimates.

The raw collection manifest, environment/data provenance and per-candidate
parameter arrays remain external. The audit records their digests but they
cannot be reverified from this export alone. Per-evaluation losses and elapsed
times are present; loss-curve checks do not establish candidate feasibility or
reproduce the underlying simulator outputs. Preserve the original collection
on the execution PC, including the failed attempt.

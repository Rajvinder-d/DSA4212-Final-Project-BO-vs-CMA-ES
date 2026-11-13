# Report evidence map

This map follows the final report’s scientific questions. The experimental
sequence and superseded findings belong in [Research progression](RESEARCH_PROGRESSION.md).
Original V5 is retained as historical evidence; the final boundary/quality results
use **aligned V5 only**. All paths in the table are repository-relative.

## Claims, comparisons and sources

| Report result | Experimental control and analysis | Saved evidence | Limit on interpretation |
| --- | --- | --- | --- |
| Section 2 / Figure 1: two-RC reference fit improves voltage prediction | Notebook 02; `analysis.study.reference_fit`, 300 counted simulator calls per fit in the physical box | `results/tables/reference_fits.csv`; `results/figures/model_reference_voltage.png` | Residual-based least squares uses more information and a larger domain than the black-box ball comparison; the fit is not a competitor or certified optimum |
| Section 3 / Tables 1–2: fixed problem, initial information and counted budgets | `configs/protocols/rotation_battery_v1.json`; V4 and aligned-V5 configs; `objectives.models`, `optimisers.adapters` and V4 executor | V4 `execution_record.json`, `run_status.csv`, `coverage.csv`; aligned V5 `initial_designs.csv`, `coverage.csv` | BO uses all initial observations; CMA starts at their best point. Controlled settings are not performance-tuned settings |
| H1 / Figure 2: coordinate-specific models have signed rotation effects | V4 battery ball, Q404/Q505/Q606; 20 designs; five CMA search replicates averaged within each design | `results/followup_v4/rotation_pairs.csv`, `rotation_intervals.csv` | Each interval compares a method with itself under rotation; a null interval does not establish equivalence |
| H1 / Section 4: ARD versus scalar and diagonal versus full CMA | Post hoc paired differences of rotation effects; `scripts/analyse_rotation_contrasts.py`; 5,000 design-bootstrap resamples, seed 4129 | `results/report_review/rotation_method_pairs.csv`, `rotation_method_intervals.csv` | All six ARD-minus-scalar intervals and three CMA-300 intervals are positive. CMA-100 is mixed. Pointwise intervals concern signed effects, not a universal quality ranking |
| Section 4 / Table 3: absolute quality alongside rotation effects | V4 `followup_v4_analysis.export_analysis` | `results/followup_v4/quality_and_timing.csv` | A smaller rotation effect can result from poor original-orientation performance; pooled medians are descriptive |
| Figure 3 / Section 4.3: exploratory conditioning trend | Scalar/ARD controls at κ = 1, 100, 10,000; Q202/Q303; five designs per condition | `results/followup_v3/rotation_effects_by_orientation.csv`; `scripts/prepare_report_evidence.py` | The observed ARD trend is exploratory; lines are not a fitted trend or a general conditioning law. V4 tests only κ = 10,000 on ten designs |
| H2 / Section 5: frozen mathematical agreement does not guarantee numerical agreement | Frozen-GP pilot control and fixed-history fitting/acquisition replays | `results/tables/frozen_gp_rotation_control.json`; `results/followup_v3/development_replays.csv`; V4 `development_replays.csv` | Replays test components on fixed histories; they do not isolate a unique solver fault |
| H2 / Figure 4A: corresponding pools restore scalar path agreement | Physically matched candidate pools under rotation; V4 `propose` and saved mapped candidate distances | `results/followup_v4/path_distances.csv`, `final_path_summary.csv` | Scalar-pool differences are below 10⁻¹⁵. ARD-pool divergence persists; path agreement alone says nothing about quality |
| H2 / Figure 4B: pool search can worsen quality | Paired pool/continuous contrasts on the κ = 10,000 quadratic; ten designs | `results/followup_v4/pool_quality_pairs.csv`, `pool_quality_intervals.csv` | ARD identity loss is about 88× higher geometrically. Pool coverage and numerical acquisition procedure change together |
| Boundary rotation result / Table 4, Sections 4.3 and 7 | Same physical box; resampling/projection interaction and paired 100-to-300-call changes | `results/followup_v4/boundary_intervals.csv`, `boundary_budget_intervals.csv` | All intervals include zero. The earlier V3 Q202 finding does not establish a general interaction |
| H3 / Figure 5: projection improves identity-ball CMA quality | Aligned V5 at matched 100/300 calls; five paired CMA replicates per design; 10,000 design-bootstrap resamples, seed 5129 | `results/v5-aligned/checkpoints.csv`, `contrasts.csv` | Ratios are 0.744 [0.625, 0.894] and 0.890 [0.819, 0.958]. Projection changes accepted candidates and later adaptation; it is not a clean mediation test |
| Figure 5A: matched-call BO–CMA quality difference | Six secondary CMA/BO contrasts at each budget, conditional on nineteen complete designs | `results/v5-aligned/contrasts.csv`; `scripts/analyse_aligned_v5.py` | All twelve pointwise intervals favour BO; no simultaneous-coverage or universal-ranking claim |
| Table 5 / Section 6: boundary occupancy, accuracy targets and measured cost | Complete-design medians and target hit counts; recommendation and search time from the same collection | Aligned V5 `checkpoints.csv`, `incumbents.csv`, `targets.csv`, `target_summary.csv`, `report_checks.json` | Time is implementation/hardware-specific; successful-run times accompany hit rates. No equal-time or objective-cost sweep was performed |
| Section 7: incomplete coverage and handler sensitivity | Declared complete-group rule, followed by a post hoc handler-only analysis retaining all twenty CMA designs | Aligned V5 `run_status.csv`, `coverage.csv`, `failure_note.md`, `report_checks.json` | One scalar-BO run stopped after 166 calls. Both all-design handler intervals still favour projection; the missing BO outcome is not recovered |
| Section 7: assessed prediction and external validity | Frozen recommendations on previously inspected FUDS80/FUDS50 windows | Aligned V5 `checkpoints.csv`; medians in `report_checks.json` | Same cell, fixed OCV/SOC assumptions, branch symmetry and an artificial ball limit physical interpretation and parameter identification |

## Files and reproduction

The executable settings are:

- V1: `configs/experiments/pilot.json` and `configs/protocols/rotation_battery_v1.json`.
- V2: `configs/diagnostics/development_v2.json`.
- V3: `configs/experiments/followup_v3.json`.
- V4: `configs/experiments/followup_v4.json`.
- Aligned V5: `configs/followup_v5/boundary_comparison_v4_aligned.json`, with frozen
  V4 input/version checks in `configs/followup_v5/v4_reference.json`.

The deferred V1 `main.json` is not a reported experiment. Original V5 uses
`configs/followup_v5/boundary_comparison.json` and separate exports under
`results/v5/`; its numerical conclusions are not substituted for aligned V5.

`python scripts/reproduce.py --verify-results` checks the submitted exports and
recomputes the aligned report quantities using standard-library Python.
`scripts/analyse_rotation_contrasts.py` reconstructs the direct rotation contrasts
with the numerical environment installed. `scripts/plot_report_figures.R`
regenerates Figures 2, 4 and 5 from their numerical tables.
`scripts/prepare_report_evidence.py` checks the twelve box interactions, six budget
changes and twelve conditioning intervals, checks Table 4 against the exports,
and renders Figure 3. `scripts/build_report.py --soffice soffice` rebuilds the PDF. Figure 1 is a supplied crop
of the archived scientific fit plot; notebook 02 regenerates the underlying fits.
The [reproduction guide](REPRODUCING.md) gives the complete run order and outputs.

## What the evidence cannot establish

The report develops and tests behavioural explanations, rather than claiming a
novel dataset or a universally superior optimiser. Representation, fitting and
search numerics are partly entangled; projection also changes adaptation. The
independent resampling unit is the initial design, not each CMA trajectory.
Intervals are pointwise and conditional on the selected rotations and battery
window. The V5 extension reuses V4 designs, so it is not independent confirmation.

Historical raw run directories are absent from the checkout. Curated exports,
executed notebooks and their hashes support saved-evidence checks, while new
raw runs can be generated using the supplied code. Neither export verification
nor a fresh run recreates an audit of omitted historical candidate arrays,
manifests or the exact acquisition failure. These limits remain distinct from
whether the numerical summaries agree with the final report.

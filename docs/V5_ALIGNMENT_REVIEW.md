# V5 alignment with V4: source and execution review

Reviewed 12 November 2026 before implementation. The aligned rerun has now been
implemented and executed; the current [result export](../results/v5-aligned/README.md) contains 279/280 completed trajectories. The
[aligned protocol and commands](V5_ALIGNED.md) record the implementation and
validation boundary. The comparison below records the analysis that motivated
these changes. The report now uses the nineteen complete aligned design groups, with the failed trajectory disclosed.

## Finding

V4 and V5 already share the main model and optimiser settings. The substantive
alignment work concerns random-design construction, search streams, replication
and the numerical backend. Copying hyperparameter values or changing a namespace
number in V5's JSON would not make it a controlled continuation of V4.

V5 calls the V3 executor in `followup.py`. V4 uses `followup_v4.py`, which
introduced separate, evaluation-indexed random streams. That difference was
carried into V5 because it was built from the earlier implementation.

The original V5 handler comparison remains useful: its two CMA handlers were
compared within the same design, budget, backend and settings. The differences
between studies prevent treating their trajectories as a paired continuation;
they do not demonstrate that V5's within-study treatment effect is invalid.
Different random draws also do not, by themselves, imply a different objective
or tuning policy. Their contribution to between-study score differences has
not been isolated.

## Evidence inspected

- V4 source and its executed provenance in
  `results/followup_v4/execution_record.json`.
- V4 recorded jobs in `results/followup_v4/run_status.csv`.
- V5 source, especially `v5.py`, `v5_jax.py`, `v5_analysis.py`
  and its imported `followup.py` executor.
- V5 execution/environment/data records in
  `results/v5/execution_validation.json` and the committed result exports.

The common protocol, optimiser adapters, objective/preprocessing modules,
evaluation wrapper, V3 executor, replay helpers and dependency lock are
byte-identical between the two implementations. Their hashes also match V4's recorded
execution provenance. V5's exported record does not include the complete raw
collection manifest/source-hash inventory, so the available implementation is the
reference rather than a fresh audit of every executed file.

All four processed-data hashes match between the execution records. Recorded
versions of NumPy, SciPy, pandas, Torch, BoTorch, GPyTorch, pycma and openpyxl
also match. Python differs: V4 records 3.13.6 and V5 3.13.17. V4 ran on macOS
ARM64; V5 ran on a Windows x64 Ryzen 5 5500 CPU.

No optimiser, battery simulator, experiment test suite or measured-data run was
executed during the initial comparison review. A small independent NumPy calculation reconstructed
random point sets and seed identifiers only. The V4 seed mapping matched all
100 recorded identity/full-CMA jobs. All ten overlapping integer design labels
30–39 generated different V4 and V5 initial point sets.

## What matches and what differs

| Component | Executed V4 | Original V5 | Rerun treatment |
| --- | --- | --- | --- |
| Data and objective | CALCE DST80, corrected cross-cell OCV, five-parameter two-RC MSE | Same processed bytes, model and parameter mapping | Preserve |
| Primary domain | Unit ball in normalised log parameters | Same ball | Preserve |
| GP and acquisition | Scalar/ARD Matérn-5/2; LogEI; length scales [0.03, 5]; initial 0.7; output scale [0.01, 100], initial 1; fixed variance 10⁻⁶; 60 fit iterations; 256 acquisition starts, best 8, up to 80 iterations | Same baseline settings | Preserve; V4's upper-bound-20 control is not the baseline |
| CMA | Population 8, initial σ = 0.3, isotropic start; active updates, mirrors and extra decoding disabled | Same adapter and settings | Preserve |
| Initial information | 20 charged evaluations; BO fits all, CMA mean starts at best | Same policy | Preserve |
| Initial-design labels | 20–39 | 30–39 | Use all 20 V4 designs |
| Initial-point RNG | `SeedSequence([4107, design, 0, 0, 0])` | `default_rng(design)` | Use V4 construction, not just its labels |
| CMA/Torch seed | Integer drawn from V4 stream kind 1 | Truncated SHA-256 of `[5107, design, replicate]` | Use V4's exact seed function |
| BO starts, pool and random-search RNG | Fresh indexed stream per evaluation; kinds 2, 3 and 4 | One advancing generator per run | Use V4's indexed streams |
| CMA repetitions | Five per design | Three per handler/design | Five per handler/design |
| Budget | BO/random 100; CMA 300 | Every arm 300, checkpoints 100/300 | Keep V5's matched 300-call budget |
| Arms/orientations | Four BO variants, random, full/diagonal CMA; identity and three rotations; supporting blocks | Three BO variants, random, full CMA with two handlers; identity only | Keep the focused V5 question; extra V4 factors are not required for alignment |
| Objective backend | NumPy reference | CPU float64 JAX | Recommend NumPy for the aligned primary collection |
| Bootstrap | 5,000 NumPy resamples, seed 4129 | 10,000 Python resamples, seed 5129 | Retain V5's analysis specification; this is not an optimiser setting |
| Execution controls | Four-hour cumulative cap; POSIX locking | Optional per-session limit, trajectory completion before pause; portable file locking | Retain V5's practical controls |

Both pool implementations use the same scalar GP specification and candidate
sampling rule. V4 has an existing parity test against the earlier scalar-pool
implementation; it was inspected here, not rerun. Using V4's proposal function
in the aligned runner also removes this separate code path.

For a concrete example, design 30, replicate 0 gives search seed 507603206 in
V4 and 81147440 in V5. Setting V5's namespace to 4107 would still use the wrong
hash-based construction and would leave its initial-point generator unchanged.

## Recommended rerun — historical specification

Use V4's baseline numerical settings and stream construction for designs 20–39.
Run the existing V5 arms at the identity orientation:

| Arm | Runs per design | Total runs |
| --- | ---: | ---: |
| Scalar continuous BO | 1 | 20 |
| ARD continuous BO | 1 | 20 |
| Scalar-pool BO | 1 | 20 |
| Uniform random search | 1 | 20 |
| Full CMA, resampling | 5 | 100 |
| Full CMA, radial projection | 5 | 100 |
| **Total** | **14** | **280** |

Every trajectory has 20 initial calls within a total budget of 300. Record
checkpoints at 100 and 300, then evaluate both FUDS profiles at both checkpoints.
This is **84,000 search calls and 1,120 assessment calls**, excluding declared
validation/setup work and failed attempts. There are 20 independent design
groups, not 100 independent CMA designs.

The V4-matched baseline is full CMA with resampling and the shared BO/random
arms. Projection is the handler intervention; extending BO/random to 300 is
the deliberate budget extension. The recommendation does not copy V4's entire
3,160-run grid. Diagonal CMA, additional rotations, box controls and alternate
length-scale bounds would answer additional questions.

Adding ARD-pool BO is an optional completeness check, not needed to repair the
original V5 comparison. It would add 20 runs, bringing totals to 300 runs,
90,000 search calls and 1,200 assessments. Choose its inclusion before outcomes
are inspected; it need not be added merely because V4 contained it.

Keep the primary endpoint as the design-level log10 projection/resampling MSE
ratio at 100 calls. Preserve the 300-call secondary contrast, BO–CMA comparisons,
6/10 mV targets, boundary proximity and time accounting. Matching V4 provides a
cleaner extension; it does not turn reused designs into independent confirmation
or isolate surface access from projection's effect on CMA adaptation.

## Backend and timing

NumPy removes a known implementation difference from the primary rerun. JAX
implements the same equations, and the old execution records a maximum loss
gap of 2.60 × 10⁻¹⁸ across eleven fixed check points. That is useful local parity
evidence, not proof that adaptive trajectories remain identical for 300 calls.
Near-ties and numerical fitting can amplify tiny differences.

Retain JAX support for plotting and optional separately labelled compatibility
checks. If JAX is retained for the primary collection instead, describe the
study as algorithm/design-aligned with V4 and explicitly retain the backend
difference. Do not present that as full numerical matching.

Run all rerun arms on the same machine/backend, with the existing one-thread
controls. The laptop or Windows desktop is suitable; copying V4's POSIX CLI
would introduce an unnecessary Windows incompatibility. Same package versions
do not guarantee identical paths across macOS ARM and Windows x64. Compare
runtime within the new collection, not by combining old V4 times with new V5
times. Retain setup/compilation outside search time and separately accounted.

## Implementation specification — historical

Create a versioned aligned configuration/runner and a new artifact collection;
preserve the completed V4 and original V5 implementations and exports.

1. Reuse `followup_v4.stream`, `search_seed`, `propose` and the V4 job executor
   for scientific execution. The executor already accepts supplied jobs and
   an optional objective. Give aligned jobs `lengthscale_upper=5.0` and a
   configuration containing V4's namespace and pool settings. Do not modify
   V4's frozen `load_config()` or expand its original grid.
2. Keep V5's portable file lock, per-session soft limit, graceful stop and
   completed-run verification around that executor. Include the new wrapper,
   configuration and imported scientific files in provenance. A new collection
   prevents reuse of original V5 jobs under changed random streams.
3. Replace the V5 three-replicate assumptions in configuration validation and
   `contrasts()` with the declared replicate list. Coverage is already derived
   from jobs, but analysis counts, tests, plot labels and documentation must
   reflect five replicates. Preserve grouping by initial design before inference.
4. Replace the old “fresh seed” validation for this version with an explicit
   matched-design relationship to V4. Reusing its initial designs is intentional;
   the original unseen-seed rule cannot describe this rerun honestly.
5. Keep scientific stream keys independent of handler, method and job order.
   V5's within-design execution shuffle may remain for timing balance; use a
   separate scheduling RNG so it cannot alter candidate streams.

Switching only the imported executor is insufficient: the job seed generation,
configuration fields and hard-coded analysis replicate count must change too.
The earlier V5 implementation predates V4's module; start from a source tree that
contains V4 and bring over the V5 wrapper deliberately, without replacing the
current report material.

## Checks specified before the run

- Verify initial physical points, search seeds, acquisition-start samples and
  candidate-pool hashes against V4's construction, including the first post-initial
  evaluation (stream step 20). These should match exactly on the same environment.
- Test the unchanged baseline through both execution paths on one environment.
  Candidate/loss prefixes should agree under identical inputs and backend.
  Comparing with the historical machine requires recorded tolerances and
  diagnosis of numerical divergence, not tuning until a preferred loss returns.
- Check the planned 280 jobs, full CMA generations, 100/300 checkpoints,
  1,120 assessments, distinct handler labels and five-replicate analysis. Reject
  incomplete groups and retain unsuccessful attempts in resource accounting.
- Export full incumbent curves, matched initial-design identifiers and their
  hashes, as well as existing checkpoint tables. The current checkout lacks
  raw V4/V5 candidate histories; exact historical-prefix auditing requires
  those saved traces. Existing V4 checkpoint exports support endpoint checks
  but cannot substitute for a trajectory audit.

These are the acceptance checks specified by the review. See the aligned
protocol for which checks ran locally and which require the numerical environment. The report should adopt the aligned results only
after they exist and have been verified. Preserve the original V5 collection as
a separate study rather than pooling it with the rerun.

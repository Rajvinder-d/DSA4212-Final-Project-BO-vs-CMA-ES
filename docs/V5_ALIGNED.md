# V5 rerun aligned with V4

This version tests the original V5 boundary-handler question using V4's baseline
settings and random-design construction. The [comparison review](V5_ALIGNMENT_REVIEW.md)
explains why the alignment concerns streams, replication and backend rather than
new optimiser tuning. The rerun was executed on Windows; the current export contains 279/280 completed trajectories and nineteen complete designs. See [the results and coverage](../results/v5-aligned/README.md).

## Fixed design

Use the existing corrected CALCE DST80 objective, normalised log parameters and
identity-oriented unit ball. All arms receive the same 20 initial observations
per design. Those calls count towards the 300-call budget; checkpoints at 100
and 300 are assessed on FUDS80 and FUDS50 after search. Assessment never feeds back
into optimisation.

| Arm | Runs per design | Total |
| --- | ---: | ---: |
| Scalar continuous BO | 1 | 20 |
| ARD continuous BO | 1 | 20 |
| Scalar-pool BO | 1 | 20 |
| Uniform random | 1 | 20 |
| Full CMA, resampling | 5 | 100 |
| Full CMA, projection | 5 | 100 |
| **Total** | **14** | **280** |

The design requires **84,000 search calls and 1,120 assessment calls**, excluding
failed attempts and any explicitly recorded backend checks. Designs 20–39 and
CMA replicate labels 0–4 match V4. The 20 initial designs are the resampling units
for inference; the 100 CMA trajectories per handler are not 100 independent
design groups.

The shared scientific executor is an unchanged copy of `followup_v4.py` from
the V4 reference export, verified against V4's executed source hash. The original
`followup_v4.json` is retained as its frozen specification. Its full experiment
grid and operating-system-specific launcher are not used.

The aligned job planner reproduces V4's kind-1 search-seed calculation without
importing Torch. During execution, V4 itself constructs the initial points and
per-evaluation streams: kind 0 for initialisation, kind 2 for continuous starts,
kind 3 for candidate pools and kind 4 for random search. Stream keys exclude
method, handler and scheduling order. The first post-initial proposal uses step
20. A separate scheduling seed shuffles the 14 jobs within each design.

The primary contrast remains the mean design-level log10 MSE ratio of CMA
projection to resampling at 100 calls. Replicate log losses are averaged within
design first. The 300-call handler contrast and six CMA/BO contrasts at each
budget remain secondary. Use the existing 10,000-resample bootstrap and seed
5129; intervals are pointwise. The 6 mV and 10 mV targets are unchanged.

Reusing V4's designs makes this a matched exploratory extension, not independent
confirmation. Projection changes accepted proposals and subsequent covariance
adaptation. A benefit supports a contribution from boundary handling; it does
not isolate surface access as the sole cause. This remains a controlled ball
experiment, not evidence that the ball is a physical feasibility boundary.

## Run on the existing Windows environment

For a new reproduction, the commands below use the implementation in this
repository. The existing V5 environment and verified processed data can be reused.
Resuming a historical collection requires its recorded source revision and
matching provenance; a pull is not a substitute for those checks.

```powershell
git switch main
git pull --ff-only
.\.venv\Scripts\python.exe -m pytest -q tests/test_v5.py tests/test_v5_aligned.py
.\.venv\Scripts\python.exe scripts/run_v5_aligned.py
.\.venv\Scripts\python.exe scripts/run_v5_aligned.py --run
.\.venv\Scripts\python.exe scripts/analyse_v5.py --collection artifacts/boundary-comparison-v5-v4-aligned/collection.json --plots
```

Start the measured run only after the tests pass, including the four execution
tests in `test_v5_aligned.py`. On Arch, replace `.\.venv\Scripts\python.exe` with
`.venv/bin/python`. Environment creation remains in the [reproduction guide](REPRODUCING.md#environment-and-data).
No additional dependencies or GPU setup are needed for an existing V5 environment.

The default objective backend is **NumPy**, matching V4. The runner verifies
scientific-source, protocol, dependency-lock and processed-data hashes against
`v4_reference.json`. It also checks the data manifest's OCV correction and other
model-relevant settings, and V4's eight recorded numerical package versions.
Python build, operating system and hardware are recorded, but need not match
V4's Mac. Historical paths may therefore differ even with matching settings.

There is no wall-time limit by default. Optional `--max-seconds N` pauses after
the active trajectory; it limits this invocation, not the lifetime collection.
The first Ctrl+C also requests that pause. Repeat the run command to verify and
reuse completed trajectories. Failed attempts and elapsed time remain recorded.
An unclosed invocation after a hard kill still requires inspection. Do not edit
source, packages or inputs during a collection.

The new collection is `artifacts/boundary-comparison-v5-v4-aligned/`. Original V5
collections and committed `results/v5/` results remain separate. The original
runner still selects the original design; this rerun uses `run_v5_aligned.py`.
Use the original revision if resuming an old collection whose source provenance
predates this change; do not weaken provenance checks to reuse it.

Optional `--backend jax` creates a separate `-jax` collection and retains its
recorded NumPy parity checks. That variant retains a backend difference from V4;
it is not the aligned primary run. `--plot-backend jax` may be used independently
when analysing saved NumPy results. Compare timings within the new collection,
not by pooling them with the old Mac or Windows runs.

## Exports and validation

The existing analysis command verifies the declared jobs, manifests, trace hashes,
checkpoint minima and complete-design coverage. It now handles both original
three-replicate and aligned five-replicate designs. Duplicate replicate labels
are rejected even when the total count is correct.

Alongside checkpoints, contrasts, targets and attempt accounting, it exports:

- `incumbents.csv`: every evaluation's loss, incumbent MSE/RMSE, elapsed search
  time and per-call objective time; incomplete design groups are labelled.
- `initial_designs.csv`: design labels, point counts and SHA-256 hashes of the
  initial normalised physical-coordinate arrays, using sorted-key JSON encoding.
  Aligned arrays must also equal V4's reconstructed initial design.

The [validation record](IMPLEMENTATION_VALIDATION.md) retains the pre-execution
checks, reduced-effort fixture failure, corrected production-settings test and
later production failure. The two failures are distinct: the corrected fixture
passed, while scalar BO later failed on design 21 after 166 objective calls.

The current report adopts the verified exports using nineteen complete designs,
with the failed trajectory and possible selection bias disclosed. This is a
partial collection, not a completed twenty-design study. Original V5 remains
a separate study; its evidence is not pooled with the aligned results. The
[validation record](IMPLEMENTATION_VALIDATION.md) separates local export checks
from raw-artifact verification and historical environment checks.

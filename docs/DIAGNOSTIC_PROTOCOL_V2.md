# Development diagnostic protocol v2

Status, 10 November 2026: the bounded v2 battery diagnostics completed on the
supported local Python 3.13.6 environment after all 27 pre-follow-up tests passed,
including both GP integration tests. Earlier dependency/data blockers were
specific to a different environment. Artifacts:
`artifacts/diagnostics_v2/20261110T141950872864Z/`. V1 remains unchanged.
Original-history replays and confirmation are separately versioned in
[protocol v3](FOLLOWUP_PROTOCOL_V3.md).
## Fixed scope and allowance

Settings are in [development_v2.json](../configs/diagnostics/development_v2.json).
Development uses initial-design seeds 0 and 1, rotation 101, and history lengths
20, 24 and 40. Seeds 10–19 and rotations 202/303 are reserved for later confirmation;
configuration validation rejects overlap. No confirmation experiment is launched.

The runner reconstructs two short identity-orientation isotropic-BO histories
using the unchanged v1 objective and adapter, costing at most 80 simulator calls.
These are new trajectories on the execution machine, not the original partner’s
macOS runs. Recorded differences between platforms are not silently interpreted
as reproduction of the original divergence at evaluation 25.

Six fixed-history replays make no further simulator calls. The unit-ball reference
has a separate allowance of 1,200 residual evaluations across three starts. The
combined maximum is 1,280 calls. A 1,200-second wall allowance is checked between
solver calls; an in-flight solver can exceed that deadline. No FUDS assessment,
CMA control experiment or large main grid is run by this diagnostic command.

## Scalar-GP and acquisition replay

At each checkpoint, transform one identical physical observation history into
identity and rotated coordinates. Match v1 target scaling, Matérn-5/2 scalar
kernel, fixed variance, hyperparameter bounds/initialisation and fitting settings.
Compare frozen predictions first, then separately refit in both orientations.
Record fit status, warnings, likelihood, fitted parameters, and posterior/LogEI
differences on 100 corresponding physical query points (seed 930).

For acquisition replay, load the identity fit’s hyperparameters into new models
in both orientations. Generate 256 feasible physical candidates with seed 941,
rank them once under the common identity acquisition and map the top eight starts
into both representations. These are declared diagnostic starts, not recovered
v1 RNG states. Hold the model, history, starts, constraints and tolerance fixed;
compare acquisition iteration caps of 80 and 240 with `ftol=1e-9`.

Record candidate distance in physical log coordinates, feasibility, solver
warnings and elapsed time. Cross-score both selected candidates under the same
identity acquisition. A large distance with a negligible acquisition-value gap
does not establish a worse numerical solution. This comparison is exploratory:
it tests one solver-effort change at fixed histories and does not yet measure
its effect on complete sequential trajectories.

## Unit-ball reference

Use SLSQP on mean squared residuals with `1 - uᵀu >= 0`, box bounds [-1,1], and
three-point numerical derivatives. Start at zero and two random interior points
(seed 710). Each start receives a cumulative share of the 1,200-call allowance;
unused allowance can carry forward. Cap iterations at 200, `ftol=1e-10`, and
relative finite-difference step at `1e-5`.

Every residual callback is counted, including derivative perturbations, failed
calls and infeasible trial points. SLSQP may evaluate points outside the ball
while solving the constrained problem; these cannot become the returned reference.
Reserve one call per start to evaluate its final point after radial projection
if needed. Select the best actually evaluated point with norm at most one.
If a block exhausts its allocation, retain its best feasible point and record the
termination. Numerical failure is not equivalent to successful convergence.

This remains a computed feasible reference with different solver information
from the black-box methods. It is not a certified optimum. Ball-reference
curvature, held-out assessment and further restarts remain follow-up work.

## Provenance and execution

[run_diagnostics.py](../scripts/run_diagnostics.py) checks readiness by default.
It requires explicit `--run` to execute; `--stage replay` or `--stage reference`
can select a component. Preflight checks supported Python, required packages,
processed-data hashes against v1 and the unchanged v1 source fingerprint.
Package versions are recorded; cross-platform identity is not assumed.

Each invocation creates a new directory under `artifacts/diagnostics_v2/` with
resolved configs, source/data/config hashes, environment, Git revision/dirty
state, history records, replay outputs and reference callbacks. Partial histories
and completed checkpoints survive later failure; manifests record failure and
tracebacks. V1 artifacts, notebooks and report tables are not overwritten.

## Historical validation on the restricted environment

On 10 November, ten numerical/control tests passed using the available Python
3.14.7, NumPy 2.5.3 and SciPy 1.18.1. These isolated tests check interior and
boundary optima, actual callback counts, feasible recommendation selection,
failure/budget/deadline behaviour, deterministic ball sampling, confirmation
separation and read-only preflight. They do not validate the full project on
Python 3.14, which is outside its declared Python range.

At that earlier execution, two GP integration tests were skipped: scalar prediction invariance
and identity-candidate agreement with the unchanged v1 BO adapter. Torch,
BoTorch, GPyTorch, pandas and CMA were unavailable in that environment; processed
CALCE inputs were also absent. Shell DNS resolution failed for both PyPI and CALCE, preventing
dependency installation and source-data retrieval. The original 15-test suite
was not rerun in that environment. No generated data were substituted for battery results.

## Completed supported-environment execution

The local full pre-follow-up suite passed 27 tests with no skips. The v2 run
completed six fixed-history replays, 80 history evaluations and 1,199 reference
callbacks. The iteration caps 80/240 selected identical candidates at each
checkpoint. One reconstructed-history replay (seed 1, length 40) had mapped
candidate distance about 0.81 and common-model LogEI gap about 0.0084; closely
matching fitted predictions do not prove a unique acquisition-solver cause.

The best feasible reference has loss about 3.4843e-5 V² (5.90 mV RMSE) and norm
0.9999976. One start converged and two hit their callback allocations. Some pilot
BO recommendations are better: the reference is not an optimum or lower bound.
These development findings must not be described as sequential confirmation.
The v2 scientific settings and source remain frozen; v3 adds distinct artifacts.

# V5: boundary handling and the BO–CMA comparison

This is the original V5 protocol. The separately versioned
[V4-aligned rerun](V5_ALIGNED.md) retains its question and contrasts while
matching V4's design construction, replication and NumPy backend.

Prepared 11 November 2026 and executed later that day on a Windows CPU. The design below was fixed before the run and was not changed
afterwards. This remained a prospective test of a hypothesis developed from v3,
conditional on the same battery objective. Execution and accounting are recorded
separately; this protocol does not interpret the results.

## Why this experiment

V3 provides a substantive explanation of numerical rotation sensitivity within
BO. It does less to explain BO's short-budget advantage over CMA on the battery
problem. The clearest unresolved comparison concerns the primary ball domain:
BO can deliberately select its surface, whereas CMA resamples proposals outside
it. The v3 boundary ablation uses a box, where the relevant good solutions and
sampling geometry differ. It therefore cannot settle the ball-domain question.

The exported v3 checkpoints show the following descriptive medians at 100 calls.
BO has 30 runs per method; CMA has 90, including three search replicates. These
pooled medians motivate the experiment; they are not independent-sample tests.

| Method on the ball | Calibration RMSE, mV | Evaluations on boundary | Rejected proposals |
| --- | ---: | ---: | ---: |
| Scalar GP-BO | 6.83 | 70.5% | 0% |
| ARD GP-BO | 6.09 | 78.0% | 0% |
| Finite-pool scalar GP-BO | 6.16 | 66.0% | 0% |
| Full CMA with resampling | 9.81 | 0% | 37.0% |

Source: [v3 checkpoints](../results/followup_v3/checkpoints.csv), battery-ball
rows at 100 calls. Boundary means norm at least `1 - 1e-8`; rejection fractions
use proposals as their denominator. Initial evaluations are included.

These observations suggest a mechanism, not its conclusion. A continuous
interior sampler almost surely misses the exact surface but can approach it
arbitrarily closely. Zero exact-boundary evaluations alone does not establish
poor optimisation. Distances to the boundary and attained losses matter.

## What still needs improvement in the existing study

| Gap | Consequence | Treatment in v5 |
| --- | --- | --- |
| CMA boundary ablation is on the box, main BO comparison on the ball | Boundary findings do not directly explain the primary method gap | Change the CMA handler on the same ball with the same battery objective |
| Only CMA continues to 300 calls | Its improvement cannot establish a BO–CMA crossover at equal calls | Every method runs to 300; compare prefixes at 100 and 300 |
| Covariance changes while CMA improves | Extra evaluations, step-size adaptation and covariance learning remain entangled | Do not label this a causal test of covariance adaptation |
| Full/diagonal CMA settings were modified for the invariance experiment | Results concern these implementations, not every standard CMA configuration | Keep the existing full-CMA implementation fixed; qualify algorithm-level claims |
| BO uses all 20 initial observations; CMA takes only the best as its mean | Equal available information does not imply equal use of that information | Preserve and disclose the interface; do not claim it is isolated here |
| Pool BO changes candidate coverage as well as numerical search | Stable paths do not prove a unique solver failure or better fitting | Retain it as a labelled secondary comparator, not a replacement chosen after results |
| Battery fits often lie on the artificial ball constraint | Findings may concern the restricted domain more than battery calibration generally | Limit conclusions to this domain; do not call it a physical feasibility limit |
| Many contrasts and only a few independent designs | Favourable secondary results are easy to overinterpret | Declare one primary contrast and retain null/adverse findings |

The 300-call CMA extension is useful evidence, but is not an adaptation ablation.
A later covariance-learning intervention would need to preserve mean/step-size
updates and validate the resulting algorithm. That is a separate question,
rather than another axis added to this experiment.

## Hypothesis and possible outcomes

**H1:** On the selected ball-constrained battery objective, resampling outside
the ball contributes to full CMA's loss relative to scalar GP-BO at 100 calls.
Replacing resampling with radial projection should improve CMA's attained loss
and reduce that method gap.

Keep mean initialisation, population, covariance/step-size updates, initial
observations and evaluation budget unchanged. Projection maps a proposed base
point `u` to `u / max(1, ||u||)`. The existing runner tells CMA the evaluated,
projected points. This intervention changes the accepted distribution and its
subsequent adaptation; it does not isolate surface access from all feedback.

| Observation | Supported interpretation |
| --- | --- |
| Projection improves loss and reduces the gap | Boundary handling contributes to the gap for these implementations and cases |
| Projection increases surface access without improving loss | Exact surface access alone is insufficient; H1 is not supported |
| Projection worsens loss | The proposed remedy is harmful here; retain this result |
| Difference is small or uncertain | The evidence does not resolve a handler contribution; do not infer equivalence |
| CMA improves by 300 calls with both handlers | More search helps; covariance adaptation remains an unisolated explanation |

The primary endpoint tests loss, not boundary occupancy. Occupancy and proximity
are mechanism diagnostics. This design cannot establish that the ball optimum
lies on its surface, or quantify a universal percentage attributable to bounds.

## Planned comparison

The [configuration](../configs/followup_v5/boundary_comparison.json) contains ten
new proposed initial-design seeds, 30–39, and a separate search namespace. All
runs use the identity orientation, the same CALCE inputs, log parameter ranges,
OCV correction, deterministic objective and ball as v3. Rotation confirmation
already exists; adding another rotation grid is not necessary for this question.

| Arm | Search replicates per initial design | Calls per trajectory |
| --- | ---: | ---: |
| Scalar GP-BO, ARD GP-BO, finite-pool scalar GP-BO, random | 1 each | 300 each |
| Full CMA, resampling | 3 | 300 each |
| Full CMA, projection | 3 | 300 each |

This gives **100 trajectories, 30,000 search calls and 400 assessment calls**.
Every trajectory pays for the same 20 initial observations. Checkpoints 100 and
300 follow complete CMA generations (10 and 35 respectively). Both FUDS
assessments are performed for each checkpoint only after all search finishes.
They are secondary transfer diagnostics, not untouched validation: the team has
already inspected these profiles. No assessment feeds back into optimisation.

Population eight, sigma .3 and all existing adapter settings remain unchanged.
Proposal rejections cost time but no simulator evaluation. Repairs and failed
simulator attempts retain the runner's existing accounting. No method-specific
tuning, target changes, stopping on a favourable result or manual restarts are
part of the design. Search seeds are shared across CMA handlers, but differing
proposal consumption prevents claiming identical candidate streams.

## Estimands and analysis

The primary contrast is the mean across initial designs of
`mean_replicates(log10(L_project / L_resample))` at **100 calls**, where L is the
lowest observed calibration MSE. Negative values favour projection. Report its
estimate and a 95% percentile bootstrap interval, resampling ten whole initial
designs, not individual trajectories or evaluations. The small number of
designs limits precision. Do not convert the interval into a general claim over
other cells, domains or CMA settings.

Report `log10(L_CMA / L_BO)` for each handler against scalar GP-BO on the same
design, averaging CMA replicates first at the log-ratio level. Positive values
favour BO. Gap reduction is algebraically the handler contrast with opposite
sign, not independent corroboration or a causal mediation percentage.

The 300-call handler contrast, ARD/pool comparisons and target summaries are
secondary. Keep both checkpoints and all declared arms in the output. Report
target attainment at 10 and 6 mV with non-attainment and calls/time conditional
on success. Actual search time includes optimisation and logging; it excludes
setup and the later assessments. Different hit rates prevent interpreting
conditional successful-run times alone as an equal-time ranking. No extrapolated
runtime crossover or artificial delay experiment is included.

Save recommendation margin, exact-boundary occupancy, fractions within .01 and
.05 of the boundary, rejected/repaired proposals, GP diagnostics and CMA states.
Inspect whether a loss improvement accompanies changed boundary proximity;
correlation is not proof of mediation. Display every attempt's status. Formal
contrasts use only designs with all ten planned trajectories completed. Report
how many designs were excluded and that exclusions can bias the comparison.
Do not silently turn failures into ordinary non-attainment or remove them from
the study record.

## Preventing selection after results

This hypothesis was developed after inspecting v3, so v3 is exploratory evidence
for it. New seeds assess repeatability on the same objective, not generalisation
to an independent battery dataset. Fix the design before inspecting v5 outcomes,
record any subsequent change and its reason, and distinguish prospective tests
from later exploratory analyses. Keep the originally proposed configuration in
Git history. A new seed alone is not a new battery problem.

If results do not support H1, report that result without searching over sigmas,
domains, seeds or BO variants until a favourable story appears. A technically
sound rejection is useful for the brief's requirement to develop and test an
explanation. [Project brief](https://alexxthiery.github.io/teaching/bo-vs-es/bo-vs-es.html).

## Prepared files and execution status

`scripts/run_v5.py` prints the planned design by default using standard-library
imports only. `--run` starts work after environment and data checks, without a
time limit. Optional `--max-seconds N` sets a soft limit for that invocation:
finish the active trajectory, then pause before starting another. A later
invocation may use any positive limit or none; earlier time is still recorded.
The first Ctrl+C requests the same graceful pause; a second interrupts the active
trajectory and retains its partial attempt. Completed matching runs are reused.
An unclosed invocation after a hard process kill requires investigation before
restart. Existing study outputs are not overwritten. The runner does not need
ignored v2/v3 experiment artifacts.

## Optional JAX implementation and timing

`--backend jax` selects CPU float64 evaluation of the same two-RC recurrence.
Native sample times, left-held currents, zero initial polarisation, OCV correction
and log parameter transform remain unchanged. `jax.lax.scan` replaces the Python
recurrence loop, with `jax.jit` compilation. No gradients are supplied to the
black-box optimisers. BO still uses BoTorch/PyTorch and CMA still uses pycma;
neither optimiser nor its settings are ported or retuned. The default NumPy
backend remains available.

Before search, compare JAX and NumPy losses at the centre and the ten signed
coordinate-axis surface points. Abort if agreement fails at relative tolerance
1e-10 and absolute tolerance 1e-12. This compiles the calibration objective and
records **11 JAX plus 11 NumPy setup calls per invocation**, outside the 30,000
search calls. Failed setup calls remain counted. Setup duration is logged in the
invocation; search times exclude it. Returning a scalar synchronises each JAX
evaluation, so timing includes actual execution. The original NumPy FUDS
assessments remain unchanged and outside search timing.

The backend, JAX/jaxlib versions, CPU device and float64 setting enter provenance.
JAX uses its own collection by default. Small floating-point differences can
alter later decisions; parity at fixed points does not guarantee identical
trajectories. Never pool NumPy/JAX or different-machine timings as one experiment.
JAX/XLA CPU scheduling is separate from the one-thread Torch/BLAS settings;
record the environment without claiming the whole process uses only one thread.
No speed-up has been measured. Compilation may outweigh savings on small runs.

The implementation follows the official [scan documentation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
and [benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).

`scripts/analyse_v5.py` reads only the selected v5 collection, verifies hashes
and exports tables to a new directory within the selected collection. The verified
exports are committed byte-for-byte in [`results/v5/`](../results/v5/README.md) so
their recorded hashes still hold; the raw collection remains ignored. `--plots`
also writes convergence/time and handler-contrast figures. Optional
`--plot-backend jax` computes saved-trajectory cumulative minima with `vmap` and
`lax.associative_scan`; Matplotlib renders the figures. This performs no new
battery evaluations. Tests cover the design, analysis, session behaviour and
JAX parity. **The v5 suite and the 100-trajectory design have now been executed;
see [the execution audit](../results/v5/execution_validation.json). This protocol
still does not interpret the results.**

The original model, optimiser adapters and v3 executor remain unchanged. See
[reproduction guide](REPRODUCING.md#native-windows-and-original-jax-v5) for platform-specific environment paths. Review
and test the new files in that environment before treating them as validated.

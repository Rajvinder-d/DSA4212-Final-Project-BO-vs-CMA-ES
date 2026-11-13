# Follow-up v3: numerical search, CMA variability and boundaries

Accepted implementation scope: 10 November 2026. Configuration:
[`followup_v3.json`](../configs/experiments/followup_v3.json). This supersedes the
unresolved follow-up settings in the research plan, not the historical v1 pilot.
The old 2,360-run main recipe remains deferred. V3 completed all 885 planned
trajectories; [the saved audit](../results/followup_v3/validation.json) records
coverage and accounting. The specification below retains its pre-execution
settings, followed by dated completion and analysis records.

## Questions and intervention

The primary comparison is deterministic calibration loss after 100 counted calls,
including 20 common initial observations. Test whether numerical acquisition
search contributes to scalar-BO path divergence; compare CMA orientation effects
with ordinary sampling variation. Boundary-handler attribution concerns **CMA
only**, on the same physical box. BO's active constraints are context, not an
isolated boundary intervention. No percentage of rotation sensitivity attributable
to boundaries is claimed.

V1 objective, data, log parameter ranges, model assumptions and original adapters
are unchanged. V2 replays reconstructed histories at 20/24/40 observations for
development seeds 0/1 and rotation 101, comparing acquisition iteration caps
80/240 and independently fitted versus common scalar hyperparameters. V3 repeats
those checks using the **original saved pilot histories**, with source run IDs
and trace hashes. These are different evidence sources; diagnostic start seed 941
does not recover the original acquisition RNG state.

The predeclared `bo_iso_pool` intervention changes acquisition search only. At
each step, use 2,048 uniform interior and 2,048 uniform surface physical points,
mapped into each orientation. Score analytic LogEI in chunks of 256 with the same
scalar model, fitting rules and target scaling as v1. Select the first maximum
index, including deterministic exact-tie handling. The pool is renewed each step
using a separate search stream; its indexed physical points correspond across
orientations even after histories diverge. Record pool hash/index/top-score gap.
No hyperparameter bounds, priors, pool sizes or targets are retuned on confirmation.

A fixed-history check evaluates the pool under independent fits and under one
common fit. Compare candidate distance **and** common-model LogEI gap. The full
sequential intervention may fail to restore matching paths or worsen loss; either
outcome is retained. It does not test every possible acquisition optimiser.

## Frozen confirmation design

Initial designs: seeds 10–19. Orientations: identity, 202 and 303. Quadratic
designs use the subset 10–14. Initial designs use the same base-domain sampler as
v1. Search seeds derive from `SeedSequence([3107, initial_design_seed, replicate])`,
mapped to a positive 31-bit integer. Neither method nor orientation nor handler
enters that seed. Equal seeds do not imply pathwise Gaussian coupling for CMA.

| Block | Methods | Search replicates | Calls | Trajectories |
| --- | --- | --- | --- | ---: |
| Battery ball | scalar/ARD BO, finite-pool scalar BO, random | 1 | 100 | 120 |
| Battery ball | full/diagonal CMA, resampling | 3 | 300; checkpoint 100 | 180 |
| Battery box | full/diagonal CMA × resampling/projection | 3 | 300; checkpoint 100 | 360 |
| Quadratic ball, κ=1/100/10000 | original five methods | 1 | 100 | 225 |
| **Total** | | | **196,500 search calls** | **885** |

CMA's primary result uses its 100-call prefix (ten generations). Its 300-call
extension has 35 generations and is a separate adaptation/handler comparison,
not an equal-budget BO ranking. Snapshot recommendations and search timings are
saved after full generation updates. All held-out battery assessments happen
after the entire search, including the extension: two per checkpoint, **2,400
assessment calls** for the complete grid. No held-out result informs search.

The unchanged v2 allowance is at most 1,280 simulator calls; its actual execution
used 80 history calls and 1,199 reference callbacks. V3 original-history/pool
replays add no simulator calls. Reference curvature adds ten calls, and its two
FUDS assessments add two. These are separate from confirmation budgets.

## Reference and interpretation

Keep the v2 three-start SLSQP reference, terminations and callback traces. Its
best feasible loss is approximately 3.4843e-5 V² (5.90 mV RMSE), with norm
0.9999976. One start terminated successfully and two exhausted their allocations.
Some pilot BO points are better: this is a **computed feasible reference**, not
a certified optimum, lower bound or demonstration of the entire domain gap.

Compute ambient local Gauss–Newton curvature at this feasible point using step
1e-4 and ten residual calls, clipping derivative points only to the physical box.
Some perturbations may leave the ball; log that explicitly. This is not a
constrained tangent Hessian or a global condition number. Keep the full-box
reference and its different feasible domain separately identified.

## Estimands and uncertainty

- Report signed `log10(rotated loss / identity loss)` by orientation before
  pooling. Average associated search replicates/orientations within initial
  design, then bootstrap complete designs (2,000 resamples, RNG129).
- For CMA, also report all independent-search identity-versus-identity pairs.
  Those pairs overlap; they are descriptive variability controls, not independent
  replicates. Do not subtract their absolute mean from rotation effects and call
  the remainder causal sensitivity.
- On the box, report `(rotation effect under projection) − (rotation effect under
  resampling)`, with proposal rejection/repair and evaluated boundary occupancy.
- Use condition-specific quadratic comparisons. Changing normalised curvature
  spectra prevents interpreting raw cross-condition losses as a universal
  difficulty ranking. Five design groups provide limited precision.
- Plot loss against calls and measured search time. Battery targets are 10 mV
  and 6 mV calibration RMSE, chosen from pilot evidence before confirmation.
  Report hit rates and non-attainment alongside conditional calls/time-to-target.
  Target times use completion of the attaining objective evaluation; checkpoint
  search timings additionally include generation updates/logging.
- Inspect fit statuses, warnings and length scales within 1% of declared bounds.
  Preserve unsuccessful runs, partial groups and non-attainment. Contrasts use
  only complete planned initial-design groups, explicitly reporting coverage and
  potential selection bias if failures exclude groups.

Curves and raw checkpoint tables retain all completed runs, labelled by coverage.
No inference over all rotations/cells, fresh-noise performance, equal-time winner
or physical parameter uniqueness is implied. FUDS was already inspected during
development and is not untouched validation of subsequent modelling choices.

## Resource policy and provenance

One CPU worker; one Torch/BLAS/OpenMP thread. Execute battery-ball groups first,
then boundary groups, then quadratics. Finish all planned members of an initial
design group before starting another. Four hours covers original v2 runtime and
all v3 development/confirmation invocation time, including failed retries; tests,
analysis and authoring are separately reported. Deadlines are checked between
calls/solves, so an in-flight solve can overrun. Partial attempts survive.

The new runner freezes source/data/configuration/environment provenance and
rejects mixing a changed implementation into an existing collection. Completed
matching runs are reused only after hash/count/loss validation; incomplete runs
receive a retry suffix. No mid-run optimiser state is resumed. A process lock
prevents concurrent writers. Abnormally terminated invocation records require
investigation before a restart so compute accounting is not silently reset.

V1 source, notebooks and audit remain unchanged. Development records identify
the original v2 artifact directory. Member contributions are recorded separately
in [Project contributions](STUDENT_ENGAGEMENT.md).


## Confirmation completion — 11 November 2026

All 885 planned v3 trajectories completed successfully on 10 November at
23:57 Singapore time. Confirmation elapsed time was 4,590.42 seconds (76.51
minutes), within the four-hour allowance. The complete grid contains 196,500
search calls and 2,400 separately counted assessment calls. Final analysis is
regenerated by notebook 05; its hash/accounting audit is
`results/followup_v3/validation.json`. Earlier running/pending statements describe
the dated handoff, not the current execution state. Current interpretation is
maintained in [Report interpretations](REPORT_INTERPRETATIONS.md).

## Post-confirmation analysis review — 11 November 2026

Notebook 05 now displays sequential path matching, intervention loss intervals
and descriptive timings, CMA signed rotation contrasts beside independent-search
identity variability, paired 100/300-call loss changes, conditioning and numerical
health, and the adaptation plot. Modified diagonal learning rates remain explicit.
Covariance anisotropy alone is not evidence of useful directional alignment or
of the causal benefit of covariance adaptation.

The additional paired boundary-budget contrast estimates B300 minus B100 within
each design/search replicate/orientation before averaging search replicates and
bootstrapping designs. It is post-confirmation exploratory analysis, not a new
predeclared hypothesis. Diagonal CMA gives Q202 -0.182 [-0.255, -0.116] and Q303
-0.087 [-0.202, 0.016]. Thus attenuation has stronger evidence under Q202;
including zero at 300 calls does not establish disappearance. Attribution remains
CMA-specific on the same physical box. Raw experiments and budgets are unchanged.
New outputs and analysis source hashes are recorded in the regenerated validation
audit with revision `v3-review-2026-11-11`. The optional
[V4 scope review](V4_SCOPE_REVIEW.md) is a proposal, not an executed protocol.

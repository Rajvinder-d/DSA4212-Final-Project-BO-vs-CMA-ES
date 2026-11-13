# Research direction and follow-up experiments

Historical research plan, 10–11 November 2026. The proposed focused studies
subsequently completed as V3 and V4; original V5 also completed, and aligned V5
completed 279/280 trajectories. [Research progression](RESEARCH_PROGRESSION.md)
connects these outcomes to the decisions below. [Report interpretations](REPORT_INTERPRETATIONS.md)
contains the current synthesis; future-tense statements below retain their dated context.

Historical planning update, 10 November 2026. The project retains measured battery calibration as
the application and BO versus CMA-ES as the comparison. The next stage develops
and tests an explanation for the pilot observations before expanding the study.
The [v3 protocol](FOLLOWUP_PROTOCOL_V3.md) now freezes the implementation of this
research direction; this document retains the rationale and alternatives. V1
remains the specification of the completed pilot. On the supported local Python
3.13.6 environment, all 27 pre-follow-up tests passed, including GP integration,
and the bounded v2 battery execution completed. The earlier missing-package/data
and DNS blockers concerned another environment. V3 adds original-history replays,
a finite-pool acquisition intervention, CMA variability controls and a 100/300-call
analysis. Current execution/coverage comes from its manifests, not this plan.

## Question and contribution

How do search representation and numerical implementation affect the response of
BO and CMA-ES to equivalent coordinate rotations, and what does their evaluation
efficiency cost in elapsed time?

The [assignment brief](https://alexxthiery.github.io/teaching/bo-vs-es/bo-vs-es.html)
asks for an experimentally tested explanation of a behavioural difference. Its
geometry direction explicitly includes theoretical invariances in practical
implementations. The contribution will be the controlled comparison and its
explanation. Reusing battery data is appropriate; a new application or algorithm
is not required by the supplied page. No particular grade is implied.

The five-parameter two-RC model, measured CALCE traces, scalar/ARD GP kernels,
full/diagonal CMA and random search provide the starting comparison. Quadratics
remain supporting controls for known conditioning. Additional datasets, algorithm
families and hybrids are deferred unless they resolve a specific unanswered
question. The existing 2,360-run `main.json` is an unrun v1 recipe, not the next
experiment to execute unchanged.

## What the pilot supports

The following observations come from the committed v1 outputs, reviewed on
10 November. They are descriptive, conditional on the selected cell, domains,
settings and short budget.

| Observation | Evidence | Consequence for the next experiment |
| --- | --- | --- |
| BO has lower error at 100 calls but much greater overhead | In the battery-ball block, median calibration RMSE is 5.40–6.11 mV for BO and 10.40–12.03 mV for CMA; median search times are 22.15–31.59 s and about 0.16 s respectively | Report evaluation efficiency and elapsed time separately; an equal-time winner has not been established |
| The proposed ordering of rotation sensitivity is not supported by the pilot | Both absolute-effect contrasts are negative; only three seeds were used | Investigate alternative explanations rather than selecting a favourable orientation or assuming the original hypothesis must hold |
| Scalar-GP mathematics and sequential BO paths behave differently | Frozen posterior/LogEI differences are near floating-point precision, while seed-0 candidate distance reaches about 1.08 at evaluation 25 | Separate hyperparameter fitting from acquisition optimisation using identical histories |
| The ball restriction is active | All 12 primary BO recommendations lie on the boundary; the corrected two-RC box reference has coordinate norm about 1.131 | Obtain a reference within the ball and examine boundary behaviour explicitly |
| CMA absolute paired differences mix orientation and search randomness | Equal integer seeds do not map Gaussian proposals to the same physical samples | Add a same-orientation stochastic control and examine outcome distributions |

Sources: [run summaries](../results/tables/pilot_runs.csv),
[sensitivity contrasts](../results/tables/pilot_sensitivity_contrasts.csv),
[frozen GP control](../results/tables/frozen_gp_rotation_control.json),
[candidate distances](../results/tables/pilot_path_deviations.csv),
[recommendation parameters](../results/tables/pilot_recommendation_parameters.csv)
and [reference fits](../results/tables/reference_fits.csv). Medians above pool
three seeds and two orientations per method; those six outcomes are not six
independent initial designs.

The review verified exported counts, rotation-ratio arithmetic, source/notebook
hashes and saved notebook error status. The original delivery records 15 passing
tests and successful runs. A subsequent local review independently passed all 27 then-current tests and
validated all saved pilot artifacts without regenerating the pilot. The expanded
implementation suite now has 36 tests; its log is recorded separately.

## Priority 1: locate and test the numerical mechanism

The working hypothesis is that numerical fitting or acquisition maximisation can
produce different candidates from equivalent scalar-kernel BO problems, and
sequential feedback can amplify those differences. The pilot motivates this
hypothesis but does not identify the responsible component.

Use a small, declared set of histories from the pilot, including histories before
the observed divergence. At each checkpoint, take one physical observation
history and transform it into both coordinate systems. Do not compare two
already-diverged histories and attribute their differences to the current solver.

| Stage | Held fixed | Intervention and recorded evidence |
| --- | --- | --- |
| Kernel control, already available | Observations, scalar hyperparameters and physical query points | Reproduce posterior and LogEI agreement under rotation |
| Fitting replay | Identical history, target scaling, initial hyperparameters and fitting allowance | Refit in each orientation; compare fitted parameters, likelihood, statuses and predictions on corresponding query points |
| Acquisition replay | Common fitted hyperparameters, history and corresponding acquisition starts | Solve in each orientation; compare mapped candidates, acquisition values and feasibility, including whether different candidates have nearly equal acquisition values |
| Targeted intervention | Objective, domain and outer evaluation budget | Change one implicated numerical component; record whether the discrepancy and downstream loss difference change, together with added computation |

A larger acquisition-solver allowance is one possible intervention. Selecting
from a common mapped finite candidate pool is another diagnostic control; it
changes the acquisition search and must be labelled accordingly. Choose the
intervention after the replay evidence, record that development choice, then
freeze it before confirmation. Do not change several solver settings together
and attribute the result to one of them.

Confirm on optimiser seeds and orientations not used to choose the intervention.
The precise split, checkpoint set and compute allowance must be recorded in the
follow-up configuration before execution. Stable fitted predictions followed by
different acquisition candidates would localise the discrepancy; identical
acquisition values at different candidates would also require considering
non-unique maxima. A failed intervention weakens its proposed explanation.
No direction of the final performance change is assumed.

This diagnostic supports the BO–CMA comparison: interpret scalar/ARD differences
only after distinguishing representation from numerical search, and interpret
full/diagonal CMA differences against the stochastic control below.

## Priority 2: strengthen the comparison controls

**CMA variability.** Separate initial-design seeds from search seeds. For each
fixed initial design, run multiple search seeds in the identity and each selected
orientation. Include identity-versus-identity comparisons with independent search
randomness. Compare per-orientation log-loss distributions, signed mean effects
and within-orientation spread. Retain absolute differences as descriptive values,
not pure estimates of rotation dependence. Do not subtract two absolute-effect
averages and automatically interpret the remainder as a causal effect.

Account for the shared initial design and reused identity baseline when computing
uncertainty; resample independent design groups with their associated runs rather
than individual trajectory points. Report orientation-specific results before
pooling, since positive and negative effects can cancel. The selected rotations
do not justify inference over all rotations. Exact physical sampling coupling is
an optional separate control, not something established by using equal seeds.

**Domain reference.** Fit the same residual model subject to the unit-ball
constraint. Count all simulator calls, including derivative evaluations, and
report constraint feasibility, termination and the best feasible result. This is
a computed reference, not a certified optimum or an identical-information
competitor. Keep the full-box reference for its own domain. Inspect curvature
and residuals at relevant feasible points; the existing curvature at the
out-of-ball reference cannot alone explain behaviour inside the ball. Changing
the ball to improve the result would define a separate experimental condition.

## Secondary analysis: evaluation efficiency and time

Show best calibration loss against counted evaluations and measured elapsed time.
Use common hardware/thread settings and preserve all included timing components.
At a fixed budget, CMA must have enough generations to make adaptation behaviour
observable; examine its trajectory rather than assuming 100 calls are sufficient.

Predeclare common loss targets from development evidence, then report target-hit
rates, calls and time on confirmation runs. Runs that do not reach a target stay
visible as failures to reach it within budget; do not report successful-run times
as though every run succeeded. A true equal-time comparison needs a declared
stopping and population-accounting rule, not extrapolation from final timings.

An optional cost calculation can explain when BO's saved evaluations compensate
for its overhead. For a reached target, let N_m be calls and H_m be non-objective
time up to that target. Under an explicitly hypothetical constant evaluation cost
c, T_m(c) = N_m c + H_m. If N_CMA > N_BO and H_BO > H_CMA, the crossover is

`c* = (H_BO - H_CMA) / (N_CMA - N_BO)`.

This calculation assumes the search trajectory is unchanged when only evaluation
cost changes. Report its variation across runs and cases with no positive
crossover. Do not compute it from the current equal-call final totals, manufacture
a value when a method misses the target, or describe it as a measured runtime on
a different battery model. Extra artificial delays are unnecessary.

## Evidence required before expanding the study

Proceed from replay diagnostics to a small versioned follow-up, then to repeated
confirmation of the question the diagnostics support. Freeze exact numerical
settings, independent random streams, targets, confirmation split, failure rules
and total compute allowance in a new protocol/configuration. These details are now executable in the v3 configuration and documented in
[FOLLOWUP_PROTOCOL_V3.md](FOLLOWUP_PROTOCOL_V3.md). That protocol selects the finite
candidate-pool intervention before confirmation and retains iteration-cap
comparisons as development diagnostics. Boundary attribution is CMA-specific.

Retain v1 artifacts and their original interpretation. Record post-pilot choices
as exploratory development and keep confirmation outcomes separate. Revised
analysis of old results must identify the v1 run IDs and the new analysis version.
Full/diagonal CMA's modified learning rates remain disclosed; conclusions concern
these variants, not every default CMA implementation.

The report should establish a reproducible comparison, test a specific mechanism,
and explain its practical cost. An unsupported hypothesis or stable null result
is acceptable when the experiment distinguishes plausible alternatives. Scope
remains one dynamic cell, transferred OCV, fixed capacity/SOC assumptions and
selected traces; the held-out profiles already inspected in the pilot are not
untouched confirmation data for later modelling choices.


## Confirmation completion — 11 November 2026

All 885 planned v3 trajectories completed successfully on 10 November at
23:57 Singapore time. Confirmation elapsed time was 4,590.42 seconds (76.51
minutes), within the four-hour allowance. The complete grid contains 196,500
search calls and 2,400 separately counted assessment calls. Final analysis is
regenerated by notebook 05; its hash/accounting audit is
`results/followup_v3/validation.json`. Earlier running/pending statements describe
the dated handoff, not the current execution state. Current interpretation is
maintained in [Report interpretations](REPORT_INTERPRETATIONS.md).

# Research progression: EDA through aligned V5

This account connects experimental choices to the evidence available at each
stage. The [decision log](DECISIONS.md) retains dated decisions, the
[protocol index](EXPERIMENT_PROTOCOL.md) specifies designs, and
[report interpretations](REPORT_INTERPRETATIONS.md) gives the current synthesis.
The stages reuse one battery application; they are not independent battery studies.
Member contributions are recorded separately in [the contribution record](STUDENT_ENGAGEMENT.md).

## Initial EDA: establish the objective and comparison

**Question.** Could measured battery calibration provide a controlled black-box
problem for investigating coordinate geometry?

**Design and rationale.** Inspection covered CALCE workbook channels, profile
stages, current sign, native sampling and OCV construction. The selected problem
fits a five-parameter two-RC model to SP20-2 DST voltage, with FUDS assessments
and a declared SP20-3 OCV transfer. Normalised log parameters permit rotations
without changing physical parameter ranges when the domain is transformed too.
Shifted quadratics provide known conditioning separately from battery curvature.

**Findings.** The data preparation and model checks established an executable
deterministic calibration objective. Repeated search runs measure optimiser
variation, not freshly sampled measurement noise.

**Limitations.** Cross-cell OCV, fixed SOC/capacity and the constant-parameter
model are approximations; low voltage error does not identify unique battery physics.

**Next change.** V1 froze the data/model settings and compared representations
under preserved physical domains and common initial observations.

Evidence: [EDA notebook](../notebooks/01_data_inspection_and_preparation.ipynb),
[model/geometry notebook](../notebooks/02_battery_model_and_geometry.ipynb),
[problem-selection review](PROBLEM_SELECTION_REVIEW.md), [V1 protocol](EXPERIMENTAL_PROTOCOL_V1.md).

## V1: test the initial geometry hypothesis

**Question.** Would scalar GP-BO and full CMA be less rotation-sensitive than
ARD BO and diagonal CMA, and would conditioning or boundary handling matter?

**Design and rationale.** The 84-run pilot compared five methods, charged twenty
common initial observations within a 100-call budget, preserved the ball or
transformed physical box, and included quadratic controls and CMA handlers.
Full/diagonal CMA learning-rate settings were controlled rather than assumed
identical from library defaults.

**Findings.** Scalar-BO paths diverged despite near-floating-point agreement in
the frozen-GP control. The pilot did not support the proposed absolute-sensitivity
ordering. BO offered lower error within the evaluation budget at greater overhead.

**Limitations.** Three initial designs and only ten subsequent CMA generations
provided limited evidence. Equal CMA integer seeds did not couple physical draws;
absolute paired differences could reflect ordinary search variability.

**Next change.** The large V1 main grid was deferred. V2 investigated fitting and
acquisition decisions at common histories before expanding confirmation.

Evidence: [V1 protocol](EXPERIMENTAL_PROTOCOL_V1.md), [pilot notebook](../notebooks/03_pilot_experiments.ipynb),
[pilot interpretation](../notebooks/04_results_and_interpretation.ipynb),
[research plan](RESEARCH_PLAN.md), [pilot audit](../results/tables/delivery_validation.json).

## V2: locate the numerical discrepancy

**Question.** Did scalar-BO divergence arise before or during acquisition search?

**Design and rationale.** Six fixed-history replays compared corresponding
predictions, independent/common hyperparameters and mapped acquisition starts.
The acquisition iteration cap increased from 80 to 240. A separate constrained
reference search assessed the ball domain.

**Findings.** Fitted predictions agreed closely, but one replay selected physical
points about 0.81 apart with a common-model LogEI gap about 0.0084. The higher
iteration cap did not change selected points at any checkpoint. The feasible
reference reached about 5.90 mV RMSE, worse than some BO recommendations; one
start converged and two exhausted their callback allowances.

**Limitations.** These were reconstructed histories, not the original pilot
trajectories. Diagnostic starts did not recover the original acquisition RNG
state. Point distance alone did not establish a materially worse acquisition
solution, and the reference was not a certified optimum.

**Next change.** V3 returned to saved original histories and tested a corresponding
finite candidate pool as an intervention on the acquisition decision procedure.

Evidence: [V2 protocol and execution record](DIAGNOSTIC_PROTOCOL_V2.md),
[diagnostic notebook](../notebooks/05_followup_diagnostics_and_confirmation.ipynb),
[V3 original-history replays](../results/followup_v3/development_replays.csv).

## V3: intervene on search and measure ordinary variability

**Question.** Would corresponding pools restore scalar path agreement, and how
large were CMA orientation differences relative to ordinary stochastic variation?

**Design and rationale.** Original-history replays preceded 885 confirmation
trajectories. Scalar-pool BO scored 2,048 interior and 2,048 surface physical
candidates per step. CMA had three search replicates per design, a 300-call
extension and resampling/projection comparisons on the same box. BO retained
the 100-call budget. Settings were frozen before confirmation.

**Findings.** Pool controls selected corresponding scalar candidates on the
original histories and restored sequential path agreement. Short-budget CMA
differences needed qualification against ordinary variability. A Q202 box
boundary finding was orientation-specific rather than a universal mechanism.

**Limitations.** The pool changed candidate coverage as well as numerical search.
There was no ARD-pool arm to separate kernel dependence from continuous search.
The CMA extension did not establish an equal-budget 300-call BO comparison.

**Next change.** V4 crossed kernel and acquisition-search choices, increased CMA
replication, used new designs/rotations and tested length-scale bounds.

Evidence: [V3 protocol](FOLLOWUP_PROTOCOL_V3.md), [notebook 05](../notebooks/05_followup_diagnostics_and_confirmation.ipynb),
[V3 audit](../results/followup_v3/validation.json), [V4 scope review](V4_SCOPE_REVIEW.md).

## V4: separate mechanisms and test robustness

**Question.** Which effects persisted after controlling acquisition procedure,
sampling variability and hyperparameter bounds?

**Design and rationale.** The 3,160-run confirmation crossed scalar/ARD kernels
with continuous/pool search, increased CMA to five replicates, used Q404/Q505/Q606,
and added pool-only upper-length-scale-bound sensitivity. The difficult quadratic
remained a supporting stress control, not an estimate of global battery conditioning.

**Findings.** Scalar-pool paths matched below 10⁻¹⁵, with battery quality intervals
including zero relative to continuous search and lower search time. ARD rotation
effects persisted under pool search. The battery kernel-by-search interaction
was resolved only at Q606. Diagonal CMA showed modest 300-call orientation
penalties of about 17–20%; full-CMA intervals included zero. Same-box boundary
interactions were unresolved, and the V3 Q202 finding did not replicate on the
new cases. ARD pool search was about 88 times worse at identity on the difficult
quadratic. Raising the length-scale upper bound reduced occupancy without
resolving changes in rotation effects.

**Limitations.** Representation, fitting and bounds remained partly entangled.
Smaller rotation sensitivity did not imply better optimisation. The modest CMA
effect did not isolate covariance learning causally, and the boundary controls
concerned box rotation effects rather than the primary ball-domain BO–CMA gap.

**Next change.** V5 tested handlers on the same ball and extended every compared
arm to 300 calls. It investigated a separate quality question at identity;
it did not supersede V4's rotation experiments.

Evidence: [V4 protocol](FOLLOWUP_PROTOCOL_V4.md), [V4 results](V4_RESULTS.md),
[notebook 06](../notebooks/06_v4_mechanism_and_confirmation.ipynb),
[V4 audit](../results/followup_v4/validation.json).

## Original V5: test ball handling at matched budgets

**Question.** Did projecting CMA proposals onto the ball improve quality, and
did longer equal evaluation budgets close the BO–CMA gap?

**Design and rationale.** The original Windows/JAX collection completed 100/100
trajectories across ten designs, with three CMA replicates per handler and
100/300-call checkpoints for every arm. The primary contrast was the design-level
projection/resampling log-MSE ratio at 100 calls.

**Findings.** Its geometric-mean MSE ratio was 0.665 [0.505, 0.868] at 100 calls;
the secondary 300-call ratio was 0.946 [0.846, 1.062], leaving the latter
unresolved. All six CMA/BO intervals favoured BO at 300 calls, with much greater
BO search overhead.

**Limitations.** V5 inherited the V3 executor and differed from V4 in initial-design
construction, streams, replication and backend. Matching model hyperparameters
did not make it a paired continuation of V4. These differences did not invalidate
the within-V5 handler comparison. Projection changed accepted proposals and adaptation.

**Next change.** The alignment review specified a separate V4-matched extension;
the original collection was retained rather than pooled or overwritten.

Evidence: [original V5 protocol](FOLLOWUP_PROTOCOL_V5.md), [original results](../results/v5/README.md),
[original contrasts](../results/v5/contrasts.csv), [alignment review](V5_ALIGNMENT_REVIEW.md).

## Aligned V5: align construction and retain the production failure

**Question.** Did the handler result persist with V4's executor, indexed streams,
twenty design labels, five CMA replicates and NumPy backend?

**Design and rationale.** The 280 planned trajectories retained the same ball
objective and initial information. Projection and the BO/random budget extension
were the intended interventions. Reusing V4 designs made this an exploratory
matched extension, not independent confirmation.

**Findings.** 279 trajectories completed. Scalar BO failed on design 21 after
166 evaluations; complete-design inference excludes that design at both budgets,
leaving nineteen BO/random runs per arm and 95 CMA runs per handler. The primary
projection/resampling ratio is **0.744 [0.625, 0.894]**; the secondary 300-call ratio
is **0.890 [0.819, 0.958]**. All six CMA/BO intervals favour BO at both budgets.
A post hoc handler-only analysis of all twenty designs also favours projection.
Scalar-pool and random 100-call losses match V4 numerically on all nineteen
included designs; continuous BO and CMA endpoints do not consistently match.

**Limitations.** Exclusion may bias BO–CMA comparisons. The supplied failure log
reports infeasible acquisition candidates, not an infeasible ball or a proven
numerical cause. Different execution machines preclude a claim of identical
paths. Raw manifests and candidate arrays remain external; exported loss checks
do not independently verify feasibility or simulator outputs. Pointwise intervals
are not simultaneous guarantees, and no budget-interaction test establishes
attenuation of the handler effect.

**Next change.** The current report uses this qualified nineteen-design analysis.
Further causal or cross-machine investigation remains an open research question,
not a completed experiment or a reason to discard the original V5 evidence.

Evidence: [aligned design](V5_ALIGNED.md), [aligned results](../results/v5-aligned/README.md),
[failure record](../results/v5-aligned/failure_note.md), [report-stage audit](../results/v5-aligned/report_checks.json).

## Decision rationale carried across iterations

| Choice | Rationale | Evidence that would weaken the explanation |
| --- | --- | --- |
| Preserve the physical domain under rotation | Isolates coordinate representation | Unequal mapped physical points or domains |
| Replay one common history | Removes feedback from already-diverged observations | Prediction discrepancies preceding acquisition search |
| Test mapped finite pools, including surface points | Controls candidate correspondence and boundary access | Persistent scalar divergence or degraded absolute quality; quality degradation was observed on the quadratic |
| Replicate CMA within each design | Separates ordinary variability from signed orientation effects | Effects comparable to ordinary variability or intervals spanning zero |
| Retain the declared ball | Avoids changing the problem after observing results | Strong domain dependence limits generalisation rather than justifying silent replacement |
| Separate short-budget and extended comparisons | Tests behaviour after additional adaptation | Longer budgets still fail to resolve an effect; unequal budgets do not support method rankings |

The account retains both successful controls and explanations weakened by the
results. Technical checks and execution histories are maintained in
[implementation validation](IMPLEMENTATION_VALIDATION.md), rather than treated as contribution claims.

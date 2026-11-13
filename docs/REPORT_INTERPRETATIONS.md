# Report interpretations: geometry, invariance and optimisation quality

Updated 13 November 2026. This is the maintained scientific synthesis for the
report. [Research progression](RESEARCH_PROGRESSION.md) explains the chronology;
[validation](IMPLEMENTATION_VALIDATION.md) records technical checks and provenance.
V1–V4 and original V5 are complete; aligned V5 is a partial collection analysed
using nineteen complete design groups.

## Current V5 evidence: aligned rerun

The V5 numerical claims in Sections 5–7 use the [aligned results](../results/v5-aligned/README.md)
and [report-stage audit](../results/v5-aligned/report_checks.json). Section 5.3
retains the original Windows/JAX comparison as historical evidence. The two
collections are not pooled.

The rerun addresses the group's concern that different uncontrolled settings
weakened the progression from V4 to V5. The model and baseline hyperparameters
already matched. The corrected extension reuses V4's executor, indexed random
streams, twenty initial-design labels, five CMA repetitions and NumPy backend.
Projection on the same ball and the BO/random budget extension remain the
intended changes. Reused designs make this an exploratory continuation rather
than independent confirmation; different machines still prevent a claim of
identical numerical trajectories.

279/280 runs completed. Scalar BO failed on design 21 after 166 evaluations,
when all eight acquisition restarts returned infeasible candidates. The declared
complete-design analysis therefore uses nineteen designs, with 95 CMA runs per
handler and nineteen per other arm. This production failure must remain visible;
it was not the earlier reduced-effort test fixture. The missing design may bias
the BO–CMA estimates and cannot be treated as an ordinary target non-hit.

Projection/resampling MSE is 0.744 [0.625, 0.894] at 100 calls and 0.890
[0.819, 0.958] at 300. The earlier approximately 34% benefit is replaced by
approximately 26%; the later interval now favours projection. Both effects
support a contribution from the handler while leaving accepted-distribution
and adaptation effects entangled. A post hoc all-twenty-design CMA sensitivity
also favours projection at both budgets. Neither analysis establishes a causal
percentage of the BO–CMA gap or a tested difference between the two budgets.

All six CMA/BO pointwise intervals favour BO at both budgets. At 300 calls,
6 mV attainment is 18/19 for scalar BO, 19/19 for ARD/pool BO, 43/95 for
resampling and 53/95 for projection; random reaches 0/19. Continuous BO takes
median search times of 401–428 s, versus 0.87–0.90 s for CMA. These are costs
within the new collection, not a NumPy/JAX speed comparison or equal-time test.

The shared scalar-pool and random baselines reproduce V4's 100-call loss on
all nineteen included designs to numerical precision. Continuous BO and CMA
endpoints do not consistently match across machines. This supports a careful
progression: the prescribed construction is aligned, the intervention is tested
within the aligned collection, and cross-machine numerical reproducibility is
a remaining limitation. It does not justify calling the entire study erratic.

## Post hoc direct rotation contrasts — 13 November 2026

The report-stage analysis in `scripts/analyse_rotation_contrasts.py` adds twelve
battery contrasts from the unchanged V4 exports. For each orientation, define
R = log10(rotated/identity loss) and compare ARD minus scalar (continuous and
pool, 100 calls), or diagonal minus full CMA (ball resampling, 100 and 300).
Pair within initial design; average five CMA replicate effects before resampling
20 designs. Intervals use 5,000 bootstrap draws, seed 4129, and are pointwise.
This is post hoc analysis, not an addition to the frozen V4 protocol.

All six ARD-minus-scalar intervals are positive, as are all three diagonal-minus-full
intervals at 300 calls. At 100, Q404 and Q505 span zero, whereas Q606 is negative:
−0.084 [−0.172, −0.005]. This early exception prevents claiming uniformly greater
diagonal sensitivity. Signed effects are not absolute sensitivity, and the
comparison across budgets is not a directly tested interaction. See
[all estimates and provenance](../results/report_review/README.md).

V4's same-box handler-by-rotation contrasts and their paired budget changes all
span zero. This does not establish the earlier V3 Q202-specific attenuation, nor
prove absence of a boundary effect. V5 tests identity-ball solution quality, a
different estimand; do not translate its improvement into a percentage of rotation
sensitivity explained by boundary handling.

## 1. The argument to develop

Coordinate representation changes how these implementations use information
about the objective. ARD BO and diagonal CMA adapt separately along coordinate
axes; their performance can therefore depend on how useful directions align with
those axes. Scalar GP models and full CMA have different invariance properties,
but the scalar GP's symmetry does not guarantee matching numerical BO paths.
The experiments distinguish this mathematical structure from the implemented
search, then examine whether a change in sensitivity actually improves attained
quality. Rotation sensitivity and the consequences of restoring path agreement
should form the main argument.

The supporting BO–CMA comparison establishes why those distinctions matter. BO
generally obtains lower calibration error within the tested evaluation budgets,
while spending substantially more time selecting evaluations. Its frequent
boundary evaluations motivate a separate intervention: projecting CMA proposals
onto the same ball improves short-budget performance. Together, these findings
show that representation, numerical acquisition search and constraint handling
affect different aspects of the comparison. None alone explains the complete
BO–CMA performance gap.

This organisation addresses the [project brief][brief]: explain when the methods
succeed, identify relevant trade-offs, and develop and test an explanation for a
behavioural difference. The report should lead with the rotation question and
the numerical intervention. Boundary handling and computational cost support
that argument. The smaller projection contrast at 300 calls is a qualification
within the boundary discussion, not an additional report theme. This priority
reflects the report emphasis recorded on 12 November.

## 2. What the experiments compare

The application fits a five-parameter, constant-parameter two-RC battery model
to measured CALCE SP20-2 DST voltage. Parameters are the ohmic resistance, two RC
resistances and their time constants. Positive physical parameters are obtained
by exponentiating affine transforms of normalised coordinates. The objective is
sample-averaged voltage MSE on a fixed trace. Repeated evaluations are
deterministic; measurement error in the recorded data does not make repeated
objective calls independent noisy observations. The [v1 protocol](EXPERIMENTAL_PROTOCOL_V1.md)
defines the model, data preparation and approximation used for cross-cell OCV.

Rotations act in normalised log-parameter space. Writing physical base
coordinates as $u=Qz$, with orthogonal $Q$, leaves the objective unchanged
when corresponding points are compared. The primary feasible set is the unit
ball, so $\|Qz\|=\|z\|$. The box controls instead enforce
$-1\leq Qz\leq1$; retaining an untransformed coordinate box would change the
physical problem. Rotation preserves the quadratic controls' Hessian spectrum.
A rotation penalty therefore concerns the method's response to alignment, not
an increase in the objective's intrinsic condition number.

Every method pays for twenty common initial evaluations. BO fits all twenty;
CMA uses the best point as its initial mean. CMA's population is eight, giving
ten subsequent generations at 100 calls and thirty-five at 300. Full and
diagonal CMA are controlled pycma variants: diagonal learning rates are matched
to the full variant, and active covariance updates, mirrors and additional
diagonal decoding are disabled. The comparison concerns these settings, not all
default CMA implementations. Equal evaluation budgets establish a common cost
constraint; they do not imply equal use of initial information or equal numbers
of adaptive decisions.

| Stage | Role in the argument | Evidence and scope |
| --- | --- | --- |
| V1 | Establish the comparison and expose unexpected path behaviour | 84 pilot trajectories across battery and control blocks; three initial designs in the primary battery comparison |
| V2 | Investigate scalar-BO fitting and acquisition decisions | Six fixed-history replays on reconstructed histories; separate feasible-reference search |
| V3 | Test corresponding-pool acquisition search and quantify ordinary CMA variability | 885 confirmation trajectories; ten battery designs; BO at 100 calls, CMA also at 300 |
| V4 | Separate kernel and search effects with stronger replication | 3,160 confirmation trajectories; twenty primary designs, ten supporting designs; three non-identity rotations |
| Original V5 | Test ball handling and matched longer budgets | 100/100 trajectories; ten complete designs; three CMA replicates; identity orientation |
| Aligned V5 | Repeat the handler intervention with V4 construction | 279/280 trajectories; nineteen complete designs used for inference; five CMA replicates; identity orientation; all arms assessed at 100/300 calls |

These stages form a sequence of development and prospective follow-up on the
same application. They are not five independent battery datasets. V4's primary
intervals resample twenty complete initial designs; original V5 resamples ten
and aligned V5 resamples nineteen. CMA search
replicates are averaged within designs rather than treated as independent
initial designs. Intervals are conditional on the chosen objective and
orientations, and secondary comparisons are not adjusted for multiplicity.

For reported log contrasts, $r=\operatorname{mean}\log_{10}(L_A/L_B)$, the
corresponding geometric-mean MSE ratio is $10^r$; the RMSE ratio is
$10^{r/2}$. Transform both interval endpoints in the same way. A ratio of
pooled median errors is a different quantity. The numerical tables below label
which summary is used.

## 3. Main interpretation: sensitivity to coordinate alignment

### 3.1 Why the representations predict different behaviour

An isotropic kernel depends on Euclidean distance, which orthogonal rotation
preserves. ARD instead uses a distance of the form
$d^2(u,u')=(u-u')^T D(u-u')$, where
$D=\operatorname{diag}(\ell_1^{-2},\ldots,\ell_5^{-2})$. In rotated
coordinates the same metric is $Q^TDQ$, generally a full matrix. A model
restricted to diagonal metrics cannot retain every rotated metric. This
derivation explains why learning separate length scales permits sensitivity to
alignment; it does not predict that every rotation must worsen performance.
The kernel background is given by Rasmussen and Williams [GPML, Chapter 4][gpml].

Full CMA can represent general covariance ellipsoids, whereas diagonal CMA
restricts their principal axes to the chosen coordinates. This supplies a
parallel explanation for coordinate dependence in the sampling distribution.
Hansen describes full CMA's rotational invariance under corresponding
initialisation [CMA tutorial, discussion of invariance][cma]. In this project,
ordinary full-CMA trajectories need not match under equal integer seeds:
the Gaussian draws are not explicitly coupled in physical coordinates.
Invariance in distribution must be assessed separately from equality of two
particular sampled paths.

### 3.2 What the rotation results establish

V4 gives the clearest repeated evidence. The table reports mean design-level
log10 rotated/identity MSE ratios and pointwise 95% bootstrap intervals, directly
from [rotation_intervals.csv](../results/followup_v4/rotation_intervals.csv).
Positive values indicate higher loss after rotation.

| Method and budget | Q404 | Q505 | Q606 |
| --- | ---: | ---: | ---: |
| ARD BO, continuous, 100 calls | 0.134 [0.057, 0.220] | 0.127 [0.044, 0.236] | 0.169 [0.075, 0.287] |
| ARD BO, pool, 100 calls | 0.133 [0.055, 0.220] | 0.109 [0.042, 0.181] | 0.085 [0.020, 0.156] |
| Scalar BO, continuous, 100 calls | −0.018 [−0.059, 0.021] | 0.010 [−0.025, 0.046] | −0.027 [−0.111, 0.063] |
| Diagonal CMA, resampling, 300 calls | 0.067 [0.034, 0.104] | 0.079 [0.044, 0.115] | 0.081 [0.042, 0.124] |
| Full CMA, resampling, 300 calls | 0.003 [−0.023, 0.029] | −0.002 [−0.021, 0.018] | −0.001 [−0.023, 0.022] |

Continuous ARD BO incurs approximately 34–48% higher geometric-mean battery MSE
under the selected rotations. Its positive effects persist with pool search,
which removes continuous acquisition optimisation from the decision step.
Continuous acquisition optimisation is therefore not necessary for the observed
ARD sensitivity. The results support the coordinate-dependence explanation,
while leaving representation, fitted hyperparameters and their numerical
interactions partly entangled. Increasing the pool models' length-scale upper
bound from 5 to 20 does not clearly resolve this attribution: all corresponding
rotation-effect-change intervals include zero.

Diagonal CMA incurs approximately 17–20% higher geometric-mean MSE at 300 calls;
full CMA's estimates remain close to zero. At 100 calls, both variants' signed
rotation intervals include zero, and absolute differences resemble ordinary
same-orientation search variability. The longer budget reveals a diagonal
effect that was unresolved in the short-budget comparison. This is compatible
with coordinate restrictions becoming consequential during adaptation; it does
not isolate the causal effect of covariance learning. Full CMA does not need to
learn rotational invariance. Also, a positive interval for one variant and an
interval containing zero for another do not themselves establish a significant
between-variant contrast.

### 3.3 Sensitivity can coexist with useful adaptation

ARD BO's sensitivity does not make it uniformly inferior. V4's descriptive median
battery RMSE is 5.05 mV for ARD versus 6.46 mV for scalar BO in the original
orientation. After rotation, ARD medians are 5.39, 5.64 and 6.21 mV, compared
with scalar medians of 6.66, 6.49 and 6.44 mV. The original-orientation advantage
narrows, but ARD remains lower in all four descriptive comparisons. These
medians are not a formal paired ARD/scalar superiority test. See
[quality_and_timing.csv](../results/followup_v4/quality_and_timing.csv).

The useful interpretation is a trade-off: coordinate-specific flexibility can
exploit favourable alignment while making results depend on that alignment.
The battery's original resistance/time-constant coordinates are meaningful,
but the experiment does not prove that they are optimally aligned. On the
quadratic controls, alignment is deliberately known. Their findings clarify
the mechanism without assigning a known global condition number to the battery.

This pattern is consistent with the representations' geometry rather than
unexpected in itself. The contribution is the controlled application-level test,
the treatment of ordinary stochastic variability, and the distinction between
relative sensitivity and absolute quality. The report should retain both
quantities instead of ranking methods by invariance alone.

## 4. Main interpretation: invariance of the model does not ensure invariance of the search

### 4.1 From the pilot anomaly to the intervention

V1's scalar-BO paths diverged despite near-floating-point agreement in the frozen
GP control. V2 then compared corresponding fixed histories, independently fitted
models and acquisition searches with shared model parameters and mapped starts.
At one reconstructed history, selected candidates were about 0.81 apart in
normalised physical coordinates, with a common-model LogEI gap of about 0.0084.
Increasing the acquisition iteration cap from 80 to 240 left the selected
candidates unchanged at all six checkpoints. This weakens a simple explanation
based only on insufficient iterations, while leaving the source of the numerical
selection difference unresolved. [V2 record](DIAGNOSTIC_PROTOCOL_V2.md)

V3 repeated the diagnostic on original saved pilot histories. These are distinct
from v2's reconstructed histories, and the diagnostic starts do not recover the
original acquisition random state. Nevertheless, the saved replays again show
closely agreeing fitted predictions alongside the large candidate difference
at the selected checkpoint. Candidate distance alone is insufficient to diagnose
a materially worse acquisition solution; the common-model acquisition values
must be considered with it.
[Original-history replays](../results/followup_v3/development_replays.csv)

Numerical acquisition optimisation is itself an optimisation problem. The
implemented BO uses fitted GP predictions and multiple constrained local
acquisition solves. BoTorch documents multistart optimisation and the use of
SLSQP for constrained acquisition search [BoTorch optimisation][botorch].
Small numerical differences can therefore alter the selected point and the
history used by subsequent fits. This is a plausible explanation for path
divergence; the diagnostics do not establish that the GP's mathematical kernel
symmetry has failed or identify one uniquely responsible solver operation.

### 4.2 What corresponding candidate pools demonstrate

V3 and v4 replace continuous acquisition optimisation with scoring a corresponding
physical candidate pool. Each step supplies 2,048 interior and 2,048 surface
points, maps them into each orientation and selects the first maximum of the
acquisition score. The scalar kernel and fitting rules are retained. Scalar-pool
sequential physical paths agree to approximately floating-point precision; v4's
final differences remain below $10^{-15}$. This is an intervention on the
decision procedure, supported by the earlier fixed-history checks, rather than
an observation of two accidentally similar endpoints.

The result shows that a practical acquisition procedure can preserve path
agreement on these cases. The intervention also changes the available candidate
set and its coverage: half the candidates lie on the surface. It therefore
does not quantify a pure solver effect on loss. Nor does agreement of scalar
paths imply that ARD predictions must agree. ARD-pool paths diverge and its
rotation effects remain positive, which separates an effect persisting under
pool search from the continuous-search contribution to scalar path divergence.
See [path summaries](../results/followup_v4/final_path_summary.csv) and
[rotation effects](../results/followup_v4/rotation_intervals.csv).

Ordinary scalar BO's substantial path differences also coexist with v4 signed
loss intervals that include zero. The evidence therefore supports a distinction
between numerical path sensitivity and a systematic orientation penalty in
final quality. It does not establish equivalent performance, and it does not
make path diagnostics irrelevant; it identifies which outcome those diagnostics
measure.

### 4.3 Restoring agreement can sacrifice quality

The difficult quadratic provides the strongest check against equating smaller
rotation effects with a better optimiser. ARD continuous BO has log10 rotation
effects around 1.96–2.09; ARD-pool effects are around 0.36–0.42. Yet pool search
worsens absolute loss for both kernels in every orientation: all eight paired
pool/continuous quality intervals are positive. ARD's original-orientation
pool/continuous log10 loss ratio is 1.945 [1.674, 2.216], approximately **88 times
the geometric-mean loss**. The reduced rotation contrast partly reflects the
loss of strong original-orientation performance.
[Pool quality contrasts](../results/followup_v4/pool_quality_intervals.csv)

A finite pool restricts the locations available to acquisition search. Missing
useful points in a narrow quadratic valley is one plausible explanation for
this quality loss, but pool size and coverage were not varied, so that mechanism
is not isolated. The battery results also differ: scalar-pool quality intervals
all include zero, whereas ARD-pool is worse at identity and Q505. A fixed
candidate procedure is therefore not uniformly beneficial across objectives
or kernels.

The report should show path agreement, orientation effects and absolute loss
together. This is the study's most useful qualification of the original
invariance hypothesis: an implementation can preserve symmetry while becoming
less effective at finding good solutions. Conversely, distinct paths need not
produce a systematic final-loss penalty. The explanation is supported by both
an intervention that restores agreement and an adverse quality result that
prevents treating that intervention as an unconditional improvement.

## 5. Supporting interpretation: BO–CMA quality and boundary handling

### 5.1 The aligned matched-budget comparison

Aligned V5 extends every arm to 300 calls. The table contains descriptive medians
from [aligned checkpoints](../results/v5-aligned/checkpoints.csv), restricted to
nineteen complete designs. There are nineteen runs per BO/random arm and 95 per
CMA handler, with five search replicates within each design. The scalar-BO
failure on design 21 excludes the whole design at both checkpoints; these are
complete-design results, not a failure-inclusive method ranking.

| Method | Calibration RMSE at 100 calls, mV | Calibration RMSE at 300 calls, mV | Search time at 300 calls, s | Runs reaching 6 mV by 300 calls |
| --- | ---: | ---: | ---: | ---: |
| Scalar BO | 6.45 | 4.76 | 401.451 | 18/19 |
| ARD BO | 5.13 | 4.73 | 428.262 | 19/19 |
| Scalar pool BO | 6.41 | 5.15 | 52.032 | 19/19 |
| Full CMA, resampling | 10.22 | 6.05 | 0.896 | 43/95 |
| Full CMA, projection | 8.36 | 5.95 | 0.871 | 53/95 |
| Random search | 14.55 | 12.53 | 0.840 | 0/19 |

All six paired CMA/BO pointwise intervals favour BO at both budgets. Additional
CMA evaluations improve its quality, but these comparisons show no crossover.
The intervals are conditional on the complete designs and are not simultaneous
confidence statements. [Aligned contrasts](../results/v5-aligned/contrasts.csv)

BO's evaluation efficiency is consistent with surrogate-guided search. Its
advantage cannot be assigned wholly to that mechanism: initial-information use,
sequential decisions and boundary handling also differ. The low-dimensional
problem suits part of BO's intended setting, while its cheap simulator makes
selection overhead consequential [Frazier's tutorial][frazier].

### 5.2 The ball-handler intervention

The primary contrast is the mean design-level log10 projection/resampling MSE
ratio at 100 calls. CMA replicate log losses are averaged within each design
before the bootstrap. Exponentiating gives **0.744 [0.625, 0.894]**, about a 26%
reduction in geometric-mean MSE. Sixteen of nineteen design-level contrasts
favour projection. This supports a handler contribution to attained quality on
the selected ball, not an attributable causal percentage of the BO–CMA gap.

The secondary 300-call ratio is **0.890 [0.819, 0.958]**, also favouring projection.
Its smaller point estimate of improvement does not establish a difference
between budgets: no direct budget-interaction interval was tested. A separately
labelled post hoc handler-only sensitivity retains all twenty designs because
all CMA runs completed. Its ratios, 0.753 [0.637, 0.895] and 0.892 [0.823, 0.957],
also favour projection. It does not recover the missing BO trajectory or remove
selection concerns from BO–CMA comparisons.

Resampling discards exterior proposals; projection maps them onto the surface
and passes evaluated points into CMA's updates. This changes both accepted
proposals and subsequent adaptation. Surface occupancy alone does not establish
quality or distinguish exploitation from uncertainty-driven BO choices. The
handler effect therefore does not isolate surface access as the sole mechanism
[practical guidance][cma-practical], [CMA constraint discussion][cma].

Radial projection obeys $\Pi(Qz)=Q\Pi(z)$, and ball membership is unchanged by
rotation. A handler can improve quality without structurally breaking rotational
symmetry. V4's unresolved same-box handler-by-rotation interactions concern a
different estimand from V5's ball-handler effect at identity. Neither establishes
a universal boundary mechanism or makes the artificial ball a physical
feasibility boundary.

### 5.3 Historical comparison: original V5

The original Windows/JAX collection completed 100/100 trajectories across ten
designs, with three CMA replicates per handler. Its primary projection/resampling
ratio was **0.665 [0.505, 0.868]** at 100 calls; its secondary 300-call ratio was
**0.946 [0.846, 1.062]**, leaving the later effect unresolved. Those conclusions
remain valid descriptions of [the original contrasts](../results/v5/contrasts.csv),
but do not describe the current aligned collection. The original design's
executor, streams, replication and backend differed from V4; the
[alignment review](V5_ALIGNMENT_REVIEW.md) explains the separate rerun. The new
result does not invalidate the original within-study comparison, and the two
collections are not pooled.

## 6. Supporting interpretation: computational cost and required accuracy

Aligned V5 uses NumPy on the Windows execution machine. Median search times at
300 calls are 401–428 seconds for continuous BO and 0.87–0.90 seconds for CMA.
Search timing includes selection and logging, while setup and later assessments
are separately accounted. These are measured implementation costs within this
collection, not a controlled NumPy/JAX speed comparison.

At 300 calls, the 6 mV target is reached by 18/19 scalar BO runs, 19/19 ARD and
scalar-pool BO runs, 43/95 resampling CMA runs and 53/95 projected CMA runs;
random search reaches 0/19. At 10 mV, resampling reaches 94/95 runs and projection 95/95.
[Target summaries](../results/v5-aligned/target_summary.csv) report hit counts
alongside conditional attainment times. A conditional time describes successful
runs only; the failed scalar-BO design was excluded, not counted as an ordinary
non-hit. Search time and call efficiency answer different questions. No
controlled equal-time extension or runtime crossover was measured.

## 7. Model interpretation and limits that affect the argument

In aligned V5's complete-design set, median FUDS80 RMSE at 300 calls is
5.71/5.70 mV for scalar/ARD BO and 7.36/7.31 mV for
resampling/projected CMA. FUDS50 medians are 5.38/5.23 and
6.73/6.55 mV respectively. These are descriptive secondary
comparisons from [aligned checkpoints](../results/v5-aligned/checkpoints.csv),
not paired assessment-effect intervals. The profiles were inspected during
development and are not untouched validation of later design choices.

Prediction accuracy also differs from physical parameter identification.
Exchanging the two RC branches leaves their summed voltage contribution
unchanged. Distinct parameter vectors can therefore produce identical model
outputs, which further limits interpretations based on parameter distance
alone. Fixed SOC/capacity, a transferred OCV curve and constant circuit
parameters are modelling approximations shared by the optimisers. Lower loss
identifies a better fit within that model and domain, not uniquely recovered
battery physics.

The feasible ball reference has approximately 5.90 mV RMSE and is worse than
some BO recommendations. It is not an optimum, lower bound or basis for global
regret. Its ambient local Gauss–Newton curvature is highly anisotropic, but
that one-point calculation is neither the battery's global conditioning nor
the constrained tangent curvature. The quadratic controls provide known
conditioning separately. Preserve these distinctions when connecting local
geometry to the rotation argument.

The project's useful explanation is conditional. There is one dynamic cell,
selected profile windows and rotations, fixed parameter bounds and controlled
algorithm settings. The original V5 design tested a hypothesis developed from
V3 before its own outcomes. Aligned V5 reuses V4 designs and is an exploratory
matched extension. Neither provides independent application validation. Additional
numerical seeds improve the assessment of search variability without expanding
the population of battery systems represented.

## 8. How to use this material in the report

The main results should move from coordinate alignment to the numerical
intervention, presenting absolute quality beside sensitivity. The short-budget
BO–CMA gap then motivates the supporting boundary result, followed by the
accuracy/time trade-off. Historical versions explain how evidence was developed;
they need not become five report sections. The broader [evidence map](REPORT_EVIDENCE.md)
and [reproduction guide](REPRODUCING.md) retain the source inventory and page constraint.

| Report claim | Decisive evidence | Qualification to preserve |
| --- | --- | --- |
| Coordinate-specific adaptation introduces alignment dependence | V4 `rotation_intervals.csv`; `quality_and_timing.csv` | Sensitivity does not imply worse absolute performance; intervals are orientation-specific |
| Scalar GP symmetry does not ensure matching numerical BO paths | V1 frozen-GP control; v2 replay record; v3 `development_replays.csv` | Diagnostic starts do not reconstruct the original random state; a large point distance need not mean a large acquisition loss |
| Corresponding pools restore scalar path agreement | V3/v4 path tables; v4 `final_path_summary.csv` | Candidate coverage changes alongside search procedure; ARD effects persist |
| Lower sensitivity can accompany worse quality | V4 `pool_quality_intervals.csv` with `rotation_intervals.csv` | The strongest deterioration is on the difficult quadratic, not uniformly on the battery |
| Ball handling changes finite-budget quality | Aligned V5 `contrasts.csv` and `checkpoints.csv` | Both intervals favour projection; 300 calls is secondary; accepted proposals and adaptation remain entangled |
| Evaluation efficiency and practical runtime favour different choices | Aligned V5 `checkpoints.csv` and `target_summary.csv` | Nineteen complete designs; disclose failure, target attainment, budget and environment; no equal-time winner established |

For figures, pair v4's rotation-effect view with the absolute-quality comparisons;
use path agreement as evidence for the numerical intervention, not as a quality
score. A compact v5 handler contrast with boundary proximity supplies the
supporting mechanism. Target attainment and time can share a table. An additional
figure devoted to the later projection contrast is not a current priority.

### Evidence provenance

V1–V4 evidence is retained in `results/`. Original V5 exports are in
`results/v5/`, and aligned V5 exports are in `results/v5-aligned/`. Their
implementations and configurations accompany the report. Export manifests and
hashes identify the stored execution state for verification or resumption.

Aligned V5 records 83,700 completed-run search calls plus 166 from the failed
attempt, 1,116 assessments, zero objective setup calls and 17,724.67 seconds
across invocations. Its twelve export hashes match the execution-side audit.
The [report-stage audit](../results/v5-aligned/report_checks.json) reconstructs
coverage, checkpoint minima, fourteen bootstrap contrasts and target summaries.
The [validation record](IMPLEMENTATION_VALIDATION.md) distinguishes these checks
from historical execution checks and records the local test results.

Raw V5 collection manifests, environment/data provenance and candidate parameter
arrays remain external. Exported initial-design hashes were verified against V4
construction by the execution-side exporter, not against historical raw arrays
in this local audit. Loss-curve agreement does not establish candidate feasibility
or independently reproduce simulator outputs. The [failure note](../results/v5-aligned/failure_note.md)
is an attributed transcription of the supplied log. These limits remain explicit
alongside the complete-design exclusion and cross-machine endpoint differences.

### Literature used for interpretation

- [Project brief][brief]: the behavioural-explanation requirement and comparison conditions.
- [Rasmussen and Williams, *Gaussian Processes for Machine Learning*, Chapter 4][gpml]: covariance structure and isotropy; bibliography key `rasmussen2006gpml`.
- [Hansen, *The CMA Evolution Strategy: A Tutorial*, revised 2023][cma]: sampling, invariance and constraint handling; key `hansen2016tutorial`.
- [Frazier, *A Tutorial on Bayesian Optimization*][frazier]: evaluation efficiency and expected improvement; key `frazier2018tutorial`.
- [CMA-ES practical guidance][cma-practical]: constraint proximity and resampling; key `cmapractical`.
- [BoTorch acquisition optimisation documentation][botorch]: multistart numerical acquisition search; key `botorch`. The repository's adapter defines the actual experiment settings.

These sources support the explanations and their limits; the numerical findings
come from this repository. Source inspection for this synthesis does not imply
that every group member has completed the corresponding reading.

[brief]: https://alexxthiery.github.io/teaching/bo-vs-es/bo-vs-es.html
[gpml]: https://gaussianprocess.org/gpml/chapters/RW4.pdf
[cma]: https://arxiv.org/pdf/1604.00772
[frazier]: https://arxiv.org/html/1807.02811v1
[cma-practical]: https://cma-es.github.io/cmaes_sourcecode_page.html
[botorch]: https://botorch.readthedocs.io/en/stable/optim.html
[v5-protocol]: FOLLOWUP_PROTOCOL_V5.md
[v5-analysis]: ../src/bo_vs_es/v5_analysis.py

# V4 results and limitations

Status: **completed and validated**, 11 November 2026. This document retains
the V4-specific findings. The [current synthesis](REPORT_INTERPRETATIONS.md)
also incorporates the separate original and aligned V5 studies. Design: [frozen protocol](FOLLOWUP_PROTOCOL_V4.md).
Evidence: [notebook 06](../notebooks/06_v4_mechanism_and_confirmation.ipynb),
[curated tables](../results/followup_v4/validation.json) and
[execution audit](../results/followup_v4/execution_audit.json).

## Execution and scope

All 3,160 confirmation trajectories completed: 400 battery BO, 800 ball CMA,
1,600 box CMA, 200 quadratic BO and 160 bound-sensitivity runs. They contain
796,000 charged search calls and 10,560 separate FUDS assessment calls. Development
adds 36 trajectories, 6,000 search calls and 72 assessment calls; 72 shared-history
replays add no simulator calls. No failed trajectories or retries occurred.
The complete scientific invocation used 8,736.05 s (2 h 26 min) of 14,400 s.
The gate forecast was 10,977.23 s for confirmation including 25% headroom.
Original analysis/notebook/audit completed about 95 s after science, followed by
a presentation-only refresh. See [execution record](../results/followup_v4/execution_record.json).

Primary battery blocks have 20 initial designs (20–39); supporting quadratic
and bound blocks have ten (20–29). Rotations are identity/Q404/Q505/Q606.
CMA has five independent search replicates within each design. All intervals below
are pointwise 95% percentile intervals from 5,000 complete-design bootstrap
resamples (seed 4129), conditional on these orientations. Search replicates are
averaged within designs. Multiple intervals do not constitute family-wide inference.

## BO: path matching, rotation sensitivity and quality differ

For signed rotation effects, R = log10(L_Q/L_I); positive means higher loss
under rotation. Battery loss is voltage MSE, not RMSE. The table gives estimates
and intervals in Q404/Q505/Q606 order.

| Method | Q404 | Q505 | Q606 |
| --- | --- | --- | --- |
| Battery ARD continuous | 0.134 [0.057, 0.220] | 0.127 [0.044, 0.236] | 0.169 [0.075, 0.287] |
| Battery ARD pool | 0.133 [0.055, 0.220] | 0.109 [0.042, 0.181] | 0.085 [0.020, 0.156] |
| Battery scalar continuous | −0.018 [−0.059, 0.021] | 0.010 [−0.025, 0.046] | −0.027 [−0.111, 0.063] |
| Quadratic ARD continuous | 1.965 [1.647, 2.266] | 2.061 [1.826, 2.283] | 2.095 [1.838, 2.365] |
| Quadratic ARD pool | 0.357 [0.232, 0.472] | 0.420 [0.274, 0.556] | 0.369 [0.215, 0.504] |

Scalar-pool final physical candidate distances remain below 1e−15 for both
objectives and both bounds. Ordinary scalar BO paths diverge substantially despite
scalar-continuous signed-loss intervals spanning zero. ARD-pool paths diverge and
its rotation effects remain positive: continuous acquisition optimisation is not
necessary for these effects. This does not uniquely separate representation,
hyperparameter fitting, bounds and their numerical interactions.

The battery kernel-by-search interaction is −0.019 [−0.133, 0.109],
−0.008 [−0.115, 0.101] and −0.111 [−0.169, −0.053]. Only Q606's interval
excludes zero. All three supporting quadratic interactions are negative, around
−1.64 to −1.67. Crucially, the quadratic pool intervention worsens absolute loss
for both kernels in every orientation: all eight paired pool/continuous intervals
are positive. ARD identity loss worsens by log10 ratio 1.945 [1.674, 2.216],
about **88× geometrically**. Its smaller rotation contrast partly reflects losing
its strong identity performance; it is not evidence of improved optimisation.
Battery scalar-pool loss intervals all include zero; ARD-pool is worse at identity
and Q505 with intervals above zero, while Q404/Q606 remain uncertain.

Battery descriptive median search times over the 80 runs per method are 20.65 s
scalar-continuous versus 2.37 s scalar-pool, and 25.45 s ARD-continuous versus
3.11 s ARD-pool. Median objective time is about 0.14 s. Pool search reduces
overhead on this cheap objective; this is not an equal-time-budget comparison.
Random search remains the baseline in all primary/control tables.
Sources: `rotation_intervals.csv`, `kernel_by_search_intervals.csv`,
`pool_quality_intervals.csv`, `checkpoints.csv`, `final_path_summary.csv` in
`results/followup_v4/`.

## Bounds and numerical diagnostics

All upper-20 minus upper-5 rotation-effect intervals include zero, for both pool
kernels/objectives. This leaves uncertainty, not equivalence. On the quadratic,
ARD-pool rotated absolute losses improve at upper 20: log ratios −0.264, −0.309
and −0.185, each with its interval below zero. Battery quality improvement is
supported only for Q505 among these pointwise ARD bound comparisons.

ARD-pool fits touch the upper bound (within 1%) in 2,254/6,400 battery fits at
bound 5 versus 816/3,200 at bound 20; for the quadratic the counts are 3,116/3,200
versus 1,520/3,200. The battery block sizes differ, so these are descriptive
occupancy counts, not the matched-design effect estimate. Fits along a trajectory
are dependent. Of 51,200 confirmation fits, 69 have STOPPED status; recorded
warning lists are empty. All runs complete and remain included. Detailed fitted
vectors and statuses are retained in raw diagnostics and replay-fit exports.
Upper bounds are a settings intervention with possible numerical consequences,
not a pure change of representation.

## CMA: replication resolves a signed effect, not equivalence

These are controlled pycma variants with modified diagonal learning rates
(`cc_sep`, `c1_sep`, `cmu_sep` matched to full constants), population eight,
initial sigma 0.3 and isotropic initial covariance. Native update mechanics remain
distinct. They do not benchmark every default CMA implementation.

At 100 calls, ball signed rotation intervals span zero for both variants, and
absolute rotation differences resemble ordinary identity-search variability.
Identity-pair overlaps are retained descriptively, not counted as independent
replications. At 300 calls, diagonal-CMA effects are 0.067 [0.034, 0.104],
0.079 [0.044, 0.115] and 0.081 [0.042, 0.124]: about **17–20% higher geometric
mean loss** after rotation. Full-CMA estimates are 0.003, −0.002 and −0.001,
with intervals spanning zero. These support a consistent diagonal effect in the
chosen cases; neither ordinary variability nor spanning zero proves equivalence.

Every planned paired log10(L300/L100) interval is negative. Ball full-CMA
estimates range from −0.482 to −0.512; diagonal estimates are −0.509 at identity
and −0.373 to −0.401 after rotation. More evaluations improve loss while step
size and covariance change. Growing spectral ratios are descriptive and do not
show alignment with useful directions or isolate a causal adaptation benefit.

On the **same physical box**, all B100, B300 and directly paired B300−B100
intervals include zero. Diagonal budget-change estimates are 0.041
[−0.066, 0.146], 0.005 [−0.090, 0.105] and 0.003 [−0.084, 0.102]. V4 does not
establish the V3 Q202 attenuation on these new rotations/designs. This is not
proof of no boundary effect, and is not a percentage of BO/CMA rotation sensitivity
explained by boundaries. V3's post-confirmation Q202 result remains a qualified,
orientation-specific finding, distinct from these predeclared contrasts.

## What remains uncertain and unfinished

The supporting quadratic has kappa=10,000; it is not the battery's global
conditioning. Finite-pool size/ties and continuous-search settings are fixed.
Bound changes are pool-only. The 300-call CMA extension is not a 300-call BO
comparison. There is no BO full-box control, coupled CMA sampling, new battery
cell/window or temperature. One cell/window, cross-cell OCV, fixed capacity/SOC,
isothermal modelling, selected domains and already inspected FUDS profiles still
limit external validity. Target attainment and non-attainment are retained in
`target_summary.csv`; predictive errors in `heldout.csv` do not establish
independent battery replication.

The report source, bibliography, figures and PDF now exist under [report/](../report/).
Submission requirements are maintained in [the project brief](PROJECT_BRIEF.md);
[authors.md](../authors.md) contains both members’ submission identities. Marking access
and Canvas submission are not established by the experimental results.

## Later report-stage comparison — 13 November 2026

A separate [post hoc analysis](../results/report_review/README.md) now directly
compares ARD/scalar and diagonal/full signed rotation effects. It preserves the
V4 exports and protocol. All ARD contrasts and CMA-300 contrasts are positive;
CMA-100 Q606 is negative and the other two intervals span zero. See the linked
complete table, bootstrap definition and provenance before generalising H1.

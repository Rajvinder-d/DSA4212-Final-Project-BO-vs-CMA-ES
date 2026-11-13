# Project contributions

Updated 13 November 2026. This record describes research design, decisions,
execution and editorial contributions. Full submission names and student numbers
are maintained in [authors.md](../authors.md). Technical reasoning and execution
checks are documented in the [research progression](RESEARCH_PROGRESSION.md),
[literature review](LITERATURE_REVIEW.md) and [validation record](IMPLEMENTATION_VALIDATION.md).

## Recorded student contributions — Mihir

Mihir's contributions centred on the initial data exploration and the experimental
design and decisions across V1–V4. The table connects six specific inputs to the
findings and methodological conclusions they supported.

| Contribution | Mihir's input | Finding and significance |
| --- | --- | --- |
| Establishing a consistent battery-calibration problem | Examined the CALCE data and defined the data-selection and preparation choices, including DST calibration, separate FUDS assessment and the declared cross-cell OCV approximation | The fixed measured traces define a deterministic optimisation objective: repeated runs measure search variability, rather than newly sampled measurement noise. Cross-cell OCV and model assumptions remain limits on physical interpretation. [Data exploration](../notebooks/01_data_inspection_and_preparation.ipynb) |
| Separating rotation from changes to the problem | Designed comparisons that preserve the physical feasible domain and common initial observations under rotation, with quadratic controls providing independently specified conditioning | Rotation preserves the Hessian spectrum, so an orientation penalty cannot be explained by an increase in intrinsic conditioning. The pilot's scalar-BO path divergence despite agreeing frozen-GP predictions motivated investigation of the numerical search. [Geometry and pilot findings](REPORT_INTERPRETATIONS.md#41-from-the-pilot-anomaly-to-the-intervention) |
| Testing the source of scalar-BO path divergence | Developed fixed-history fitting/acquisition diagnostics and the follow-up intervention using corresponding finite candidate pools | Increasing the acquisition iteration cap from 80 to 240 did not change the six diagnostic selections. Corresponding pools subsequently restored scalar path agreement to floating-point precision, supporting a numerical acquisition-search explanation without identifying a unique solver fault. [Diagnostic-to-intervention evidence](REPORT_INTERPRETATIONS.md#4-main-interpretation-invariance-of-the-model-does-not-ensure-invariance-of-the-search) |
| Distinguishing GP representation from acquisition procedure | Designed the scalar/ARD × continuous/pool comparison and length-scale-bound sensitivity checks, with absolute quality assessed alongside rotation effects | ARD rotation effects persisted under pool search, so continuous acquisition optimisation was not their sole cause. Reduced sensitivity also did not guarantee better search: ARD pool loss was about 88 times higher at identity on the difficult quadratic. Representation, fitting and bounds remained partly entangled. [BO mechanism and quality results](V4_RESULTS.md#bo-path-matching-rotation-sensitivity-and-quality-differ) |
| Separating CMA orientation effects from ordinary search variability | Added same-orientation search replication and a 300-call extension, then strengthened the design to five CMA replicates per initial design | The 100-call comparisons did not resolve a signed orientation effect. At 300 calls, diagonal CMA showed modest geometric-mean MSE penalties of about 17–20%, while full-CMA intervals included zero. This qualified the adaptation explanation rather than establishing a large or universal advantage. [CMA evidence](V4_RESULTS.md#cma-replication-resolves-a-signed-effect-not-equivalence) |
| Checking whether boundary findings generalised | Designed same-physical-box resampling/projection comparisons and follow-up checks on new rotations and initial designs | The orientation-specific V3 Q202 finding did not replicate in V4; the new boundary interaction intervals included zero. These experiments did not establish a universal boundary explanation and distinguished box rotation sensitivity from the later V5 ball-quality question. [Boundary findings and limitations](V4_RESULTS.md) |

## Recorded student contributions — Raj

Raj’s contributions centred on the comparison’s scope, the V5 investigation,
interpretation and report review. The entries below record his documented project
work and the resulting repository material through 13 November 2026.

| Contribution | Raj's work | Resulting project material |
| --- | --- | --- |
| Research scope and evaluation | Clarified that the comparison, rather than an unused dataset, was the project; set priorities for rigour, explanation and exploration | The report tests behavioural explanations rather than claiming a novel battery application |
| Investigation direction | Developed a separate V5 investigation, identified areas for improvement, and explicitly challenged possible cherry-picking | The boundary comparison and its limitations are retained in the [interpretation reference](REPORT_INTERPRETATIONS.md) and [v5 results](../results/v5/README.md) |
| V5 run preparation and reported execution | Added laptop/desktop portability, JAX support and revised runtime handling; ran V5 on the Windows PC and shared the exported results | Completed V5 exports exist. The saved audit records the completed execution; it does not establish an additional independent reproduction |
| Interpretation | Identified BO's lower calibration error and greater search time, BO's ball-boundary evaluations, and CMA's improvement under projection | These comparisons form the supporting accuracy, boundary and cost argument |
| Scientific emphasis | Set rotation sensitivity and the distinction between invariance, numerical paths and quality; included external methodological checks; set aside a separate theme around the smaller later-budget projection effect | [REPORT_INTERPRETATIONS.md](REPORT_INTERPRETATIONS.md) records this priority and the limits of the claims |
| Report design and editorial review | Used the previous report as a style/density reference, organised results by scientific question rather than chronology, and set a ten-page allocation | The report presents context and design, rotation and conditioning controls, numerical mechanisms, boundary/cost results, discussion and references within ten pages |

Raj also prepared the V4-aligned rerun after we identified differences in the
original comparisons, ran the tests and experiments on the Windows PC, and
supplied the acquisition-failure traceback for interpretation. His subsequent
review covered the report’s hypotheses, equal-budget comparisons, uncertainty,
figure readability and reproducibility. He prepared the final repository cleanup
and checked the marking guidelines directly. These contributions span research
design, execution and editorial review.

## Aligned V5 rerun and report review

We identified differences between V4 and original V5 in executor, random streams,
replication and backend. Raj prepared a rerun using V4's settings. He ran the
integration tests and collection on the Windows PC and supplied the results used
in the aligned analysis. He also provided the scalar-BO failure log during the
review of incomplete coverage.

The [aligned export](../results/v5-aligned/README.md) contains 279 completed
trajectories and one failed trajectory. The report uses nineteen complete design
groups; the failed attempt and possible selection bias remain part of the
interpretation. Original V5 results remain a separate collection.

## Group responsibility

We developed the research design, implemented and ran the studies, interpreted
the evidence and prepared the report. Mihir’s work centred on data exploration
and the V1–V4 design; Raj’s work centred on V5, execution, interpretation and
report review. We reviewed the submitted methods, numerical results and final
text. Saved evidence checks and fresh experimental reproduction remain distinct
activities.

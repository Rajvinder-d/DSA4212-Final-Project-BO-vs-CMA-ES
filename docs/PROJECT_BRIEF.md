# Project brief and research scope

Current status: V1–V4 and original V5 are complete. Aligned V5 completed 279/280
trajectories; the report uses nineteen complete designs. The [documentation guide](README.md)
and [research progression](RESEARCH_PROGRESSION.md) connect the stages and evidence.

Source: [Alexandre Thiéry, Bayesian optimisation versus evolutionary strategies](https://alexxthiery.github.io/teaching/bo-vs-es/bo-vs-es.html),
read 6 November and rechecked 13 November 2026. Submission requirements and
marking weights below follow the [assignment guidelines](https://alexxthiery.github.io/teaching/guidelines/guidelines.html),
also checked 13 November 2026.

## Requirements from the brief

Explain and experimentally test a difference in the methods' behaviour on chosen
problems. Define performance and budget, disclose counted costs, use common
problems with comparable starting information and tuning effort, and repeat
runs. For noise, evaluate final recommendations independently or against the
noise-free objective. The smallest noisy observation is insufficient evidence.

Suggested directions concern dimension, geometry, noise, multimodality and
computational cost/parallelism. They are options, not a requirement to study
all five. The separately supplied assignment instructions require a PDF of at
most **10 pages including figures, references and appendix**, one Canvas submission
per group, at most four members, the repository URL and every member's full name
and student number clearly at the report's start. No code in the main report.
The repository root must contain `readme.md`, `authors.md` and `report/` with
complete machine-readable report source, bibliography and figures. A private
repository must give GitHub user `alexxthiery` marking access. The deadline remains unspecified here; both members’ submission identities are in
[authors.md](../authors.md). [Project contributions](STUDENT_ENGAGEMENT.md)
records the member activities supplied for this project.

The supplied marking weights are understanding/problem formulation 25%,
experimental design/exploration 35%, analysis/report quality 20%, and
implementation/reproducibility 20%.

## Research direction — established by V4

Retain the measured CALCE battery application and BO–CMA comparison. Strengthen
the study through a controlled explanation of rotation behaviour, supported by
evaluation-count and elapsed-time analysis. The [research plan](RESEARCH_PLAN.md)
specifies numerical BO replays, a same-orientation CMA variability control and a
reference fit within the primary domain. The [development diagnostic runner](DIAGNOSTIC_PROTOCOL_V2.md)
completed its bounded BO replay and domain-reference execution locally. The
[v3 protocol](FOLLOWUP_PROTOCOL_V3.md) now defines the implemented confirmation
runner and analysis. V3 and the separate [V4](FOLLOWUP_PROTOCOL_V4.md)
confirmation grids are complete; see [V4 results](V4_RESULTS.md).

The [v1 protocol](EXPERIMENTAL_PROTOCOL_V1.md) documents the implemented 84-run
pilot. Its large main recipe remains deferred in favour of focused V3/V4. The
scientific contribution is the comparison and tested explanation; prior use of
the battery dataset is not itself a reason to change applications. Do not claim
the exact comparison is novel on the basis of a limited literature search.

## V5 proposal — historical, 11 November 2026

The separate [v5 proposal](FOLLOWUP_PROTOCOL_V5.md), prepared on 11 November,
tests a remaining explanation for the BO–CMA performance gap: boundary handling
on the primary ball domain. Common 100- and 300-call checkpoints address the
unequal endpoint budgets in v3's CMA extension. At proposal time V5 was
unexecuted and untested; it tests a hypothesis developed from v3 on the same battery objective, with
the design fixed before its own outcomes are inspected. It targets geometry and
computational-cost questions without claiming application novelty.

The original and aligned V5 collections have since been executed; see
[V5_ALIGNED.md](V5_ALIGNED.md) and [the result coverage](../results/v5-aligned/README.md).
Implementation and tests accompany this report.

## Initial scope proposals — historical

Choose one mechanism and a controlled intervention before broad benchmarking.
An initial candidate is sensitivity to coordinate geometry: compare original
and rotated versions of a controlled objective family while keeping its intrinsic
difficulty and evaluation budget fixed. This is a proposal, not a selected
experiment or a prediction that either method wins.

A useful hypothesis would name the precise implementations, predict an observable
change under rotation, and identify a diagnostic that could contradict it. Check
the treatment of search bounds: rotating a function inside an unchanged box can
change the feasible problem. Control or explicitly study that change. Keep
conditioning separate from rotation in the first comparison.

Alternative proposals, if better suited to the team's interests or resources:

| Question to investigate | Controlled change | Diagnostic beyond the final score |
| --- | --- | --- |
| Does additional ambient dimension matter when informative structure stays fixed? | Embed the same lower-dimensional objective in larger spaces | Progress against calls and time; sensitivity to the embedding |
| Does an apparent improvement survive independent evaluation? | Introduce a specified observation-noise process | Recommended-point quality versus the lowest recorded observation |
| When does decision overhead change the preferred method? | Vary controlled objective cost and worker count | Objective time, decision time and total elapsed time |

These were initial planning suggestions. They need method-specific reading,
a pilot and an agreed protocol before use. Start with one primary comparison;
add a secondary method or hybrid only if it tests a distinct explanation.

## Initial planning decisions — historical

Select the primary question, exact Bayesian and evolutionary implementations,
objective family, baseline and initialisation policy. Agree pilot limits before
running anything expensive. Use the protocol checklist to freeze evaluation
accounting, repetitions, recommendation rules and uncertainty summaries.

A pilot should establish that both methods obey the same evaluation contract
and that the intended intervention actually changes the targeted property.
It should not be used to choose whichever reporting condition favours a method.

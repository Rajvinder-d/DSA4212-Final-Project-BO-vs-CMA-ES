# Documentation guide

The project investigates rotation sensitivity, conditioning and boundary handling
in Bayesian optimisation and CMA-ES. V1–V4 and the original V5 collection are
complete. Aligned V5 completed 279/280 trajectories; its current comparisons use
nineteen complete initial-design groups. The two V5 collections remain separate. A later
[report-stage analysis](../results/report_review/README.md) adds twelve post hoc
paired rotation-method contrasts from unchanged V4 exports.

For the final report, use the [evidence map](REPORT_EVIDENCE.md) to locate each
claim’s experiment and numerical source. The [current submission audit](IMPLEMENTATION_VALIDATION.md#current-submission-audit--13-november-2026)
records completed checks and outstanding limits. `python scripts/reproduce.py --verify-results` checks the saved evidence without the numerical environment.

## Research progression and decisions

The recommended starting point is [Research progression](RESEARCH_PROGRESSION.md):
the questions, designs, findings and changes connecting EDA and V1–V5.
[Shared decisions](DECISIONS.md) preserves the dated decision record.
[Project brief](PROJECT_BRIEF.md) maps the work to the assignment.

The historical [problem-selection review](PROBLEM_SELECTION_REVIEW.md),
[research plan](RESEARCH_PLAN.md), [V4 scope review](V4_SCOPE_REVIEW.md) and
[V5 alignment review](V5_ALIGNMENT_REVIEW.md) explain alternatives and changes.
Their dated proposals describe what was known at the time, not outstanding work.

## Experimental protocols

[Protocol index](EXPERIMENT_PROTOCOL.md) identifies the executed designs and the
deferred V1 main grid.

| Stage | Design specification | Execution/result evidence |
| --- | --- | --- |
| V1 pilot | [V1 protocol](EXPERIMENTAL_PROTOCOL_V1.md) | [Notebook 03](../notebooks/03_pilot_experiments.ipynb), [notebook 04](../notebooks/04_results_and_interpretation.ipynb) |
| V2 diagnostics | [V2 protocol](DIAGNOSTIC_PROTOCOL_V2.md) | Completed-execution section in the protocol; [notebook 05](../notebooks/05_followup_diagnostics_and_confirmation.ipynb) |
| V3 confirmation | [V3 protocol](FOLLOWUP_PROTOCOL_V3.md) | [Saved audit](../results/followup_v3/validation.json), notebook 05 |
| V4 confirmation | [V4 protocol](FOLLOWUP_PROTOCOL_V4.md) | [V4 results](V4_RESULTS.md), [notebook 06](../notebooks/06_v4_mechanism_and_confirmation.ipynb) |
| Original V5 | [V5 protocol](FOLLOWUP_PROTOCOL_V5.md) | [Original results](../results/v5/README.md): 100/100 trajectories, ten designs |
| Aligned V5 | [Aligned design](V5_ALIGNED.md) | [Aligned results](../results/v5-aligned/README.md): 279/280 trajectories, nineteen complete designs |

## Interpretation and literature

[Report interpretations](REPORT_INTERPRETATIONS.md) is the maintained scientific
synthesis. [V4 results](V4_RESULTS.md) retains the V4-specific conclusions.
[Reading index](READING.md) locates sources; [literature review](LITERATURE_REVIEW.md)
connects them to experimental choices and contains the geometric derivations.
The [report source](../report/report.md) and [final PDF](../report/report.pdf)
are separate submission artifacts.

## Contributions and reproduction

[Contributions](STUDENT_ENGAGEMENT.md) records member activities; [authors](../authors.md)
holds submission identity fields. [Reproducing the experiments](REPRODUCING.md)
is the entry point for a fresh clone; [Operations](OPERATIONS.md) retains detailed
native commands and recovery records. [Implementation validation](IMPLEMENTATION_VALIDATION.md) retains
dated checks and their environments. The [reproduction guide](REPRODUCING.md#native-windows-and-original-jax-v5) covers
native Windows and the original JAX runner; [aligned V5](V5_ALIGNED.md) specifies the aligned runner.

Saved exports support inspection without rerunning optimisation. Raw V5 manifests
and candidate parameter arrays remain external, so export consistency checks do
not independently establish candidate feasibility or simulator-output correctness.

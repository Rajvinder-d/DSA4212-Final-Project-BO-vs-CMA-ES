# Failed scalar-BO trajectory

Source: the user's pasted Windows `error.txt`, supplied during the report review.
This is an attributed transcription of the relevant error, not a freshly read
raw artifact or a new execution. Machine-specific path prefixes are omitted.

- Run: `ca417cb2e45606ac`.
- Initial design: 21; scalar continuous BO; replicate 0.
- Exported status: failed after 166 objective calls; no FUDS assessments.
- Recorded attempt time: 177.2187519 s.
- Failure path: `followup_v4.execute_job` → `propose` → `bo_candidate` →
  BoTorch `optimize_acqf` / `_optimize_acqf_batch`.

```text
botorch.exceptions.errors.CandidateGenerationError: The optimizer produced infeasible candidates. 8 out of 8 batches of candidates were infeasible. Please make sure the constraints are satisfiable and relax them if needed.
```

This occurred under the production protocol, which allows 80 acquisition
iterations and eight restarts. It is distinct from the earlier synthetic test
failure with two iterations and two restarts. The message establishes failure
to return a feasible candidate; it does not establish that the ball is infeasible
or identify the solver's detailed numerical cause. No protocol change or rerun
has been made as part of this report analysis. The next proposal failed before
an additional objective evaluation was recorded.

The declared analysis excludes design 21 entirely at both budgets. Its 100-call
checkpoint is not exported from the failed run, so the available export does
not support reconstructing the complete twenty-design 100-call BO comparison.
The report retains the failure and qualifies complete-design inference.

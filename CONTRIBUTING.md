# Contributing

Contributions should preserve a reproducible comparison across methods. The
[experimental protocol](docs/EXPERIMENT_PROTOCOL.md) defines the comparison;
the [decision log](docs/DECISIONS.md) records agreed choices and their rationale.

## Changes and review

Keep branches and pull requests focused on one change. Coordinate changes to
shared interfaces before implementation. Each pull request should explain the
reason for the change, its effect on existing experiments and the validation
performed. Include the commands and outcomes needed for a reviewer to check it.

Changes to scientific settings should update the relevant configuration and
decision record. Identify which earlier results remain comparable.

## Experiments

Store shared comparison settings in `configs/protocols/` and individual run
recipes in `configs/experiments/`. Bounded development diagnostics have separate
settings in `configs/diagnostics/` and an explicit protocol in `docs/`. The
separate v5 design lives in `configs/followup_v5/`; its preparation and evidence
limits are recorded in [protocol v5](docs/FOLLOWUP_PROTOCOL_V5.md). Before a
substantial run, agree the hypothesis,
baseline and resource budget, then check the implementation with a small pilot.

Preserve each run's configuration, source revision and evaluation trace under a
unique ID, including failed or inconclusive runs. Record the command, runtime
and artifact location so another contributor can reproduce or inspect the result.
Report figures and tables should identify their source runs and generation code.

## Code and outputs

Keep reusable computation in `src/bo_vs_es/` and command entry points in
`scripts/`. Use notebooks to explain exploration and analyse saved outputs.
Document environment and execution requirements in [REPRODUCING.md](docs/REPRODUCING.md); the operations guide retains detailed
recovery records. Run `python scripts/reproduce.py --verify-results` before and
after changes to saved evidence. Use a separate checkout for fresh experiments.

Version code, configurations, small test fixtures and curated report outputs.
Bulk data and run artifacts remain outside Git, with access or regeneration
details recorded in the operations guide. Credentials, local environments and
personal working notes remain excluded. Decisions needed by the team belong in
the shared documentation.

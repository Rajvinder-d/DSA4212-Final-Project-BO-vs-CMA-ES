# Operations and reproducibility

For a fresh clone, start with [Reproducing the experiments](REPRODUCING.md).
It provides the unified runner, current installation command, stage dependencies
and output map. This document retains the detailed native commands and historical
execution/recovery records.

## Environment

Run commands from the repository root. The verified local environment is CPython 3.13.6 on macOS arm64, CPU. `requirements-lock.txt` records all installed distributions (no machine-specific editable path); `pyproject.toml` declares the package and dependencies. The adapter relies on pycma 4.5.0 internals, tested explicitly. Do not silently upgrade dependencies and pool new runs with old ones.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproduce.txt
.venv/bin/python -m pip check
.venv/bin/python -m ipykernel install --prefix .venv --name bo-vs-es --display-name 'BO vs ES (.venv)'
```

A second execution environment is Windows 10 x64 with CPython 3.13.17, the same locked distributions and CPU JAX/JAXlib 0.12.0.1 in float64 mode. Windows CSV regeneration uses CRLF; normalise `data/processed/*.csv` to LF so their hashes match the recorded inputs before running v5. The v5 run and its audit are recorded in `results/v5/execution_validation.json`.

If the platform lacks a pinned wheel, resolve a compatible environment separately, save a new lock and run the checks. The dependency list is not a promise of identical numerical results across BLAS implementations/platforms. CLI and notebook execution scripts limit OpenMP/OpenBLAS threads; the BO runner uses one Torch thread. Runtime comparisons should run sequentially on an otherwise idle machine.

The notebook executor selects the environment-local kernel through `JUPYTER_PATH`, stores Matplotlib/IPython/Jupyter runtime files under ignored `artifacts/`, starts a fresh kernel for each notebook and saves genuine outputs. Local Jupyter kernels require loopback sockets; restricted execution environments may need permission for those local sockets. Kernel-registration files contain local paths and remain inside ignored `.venv/`.

## Data acquisition and manual placement

```bash
.venv/bin/python scripts/prepare_data.py
```

The exact original files, all from `https://web.calce.umd.edu/batteries/data/`, are:

- `SP2_25C_DST.zip`
- `SP2_25C_FUDS.zip`
- `SP3_25C_LC_OCV_11_16_2015.zip`

If automatic download fails, download these three archives manually to `data/raw/` without extracting or renaming and rerun the command. `configs/protocols/data_sources.json` pins hashes. Existing originals are never overwritten. Downloads use verified TLS through urllib, with system curl as an alternative CA-store implementation; certificate checking is not disabled. A changed source hash is an explicit blocker requiring investigation, not permission to accept arbitrary replacement data.

`scripts/analyse_results.py` validates the saved pilot geometry, common initial designs and trace accounting and regenerates the numerical summaries without simulator calls.

`.xls` members are actually OOXML, read from bytes with openpyxl. All channel sheets are inspected. Preparation writes `data/processed/{dst80,fuds80,fuds50,ocv}.csv` and a manifest containing source records, preprocessing audits, capacity spans and DST-rest offset. Repeat preparation is deterministic apart from its timestamp. V1 search provenance excludes that timestamp, but later collection provenance can retain it. Reuse verified prepared data when resuming; keep data and metadata fixed while a collection is running.

## Tests, smoke, pilot and notebooks

```bash
MPLCONFIGDIR=artifacts/.matplotlib .venv/bin/python -m pytest -q
.venv/bin/python scripts/run_experiments.py --config configs/experiments/smoke.json
.venv/bin/python scripts/run_experiments.py --config configs/experiments/pilot.json
.venv/bin/python scripts/execute_notebooks.py
```

The smoke collection has 14 runs × 28 search calls. The pilot has 84 × 100 search calls, plus 108 final assessment calls. Notebook 02 separately performs three 300-call reference fits, six assessment calls and ten local-curvature calls. These are not hidden inside black-box budgets. Notebook 03 calls `run_collection` and skips verified completed pilot runs. Notebook 04 requires the pilot index and re-generates curated figures and tables.

Use `--start N --end N` to execute a single notebook in a fresh kernel. Notebook 01 reconstructs measured data; 02 needs prepared data; 03 needs prepared data and dependencies; 04 needs saved pilot runs. Each independently imports shared modules and establishes its root. To view interactively, open an `.ipynb` in a Jupyter-capable editor and select `.venv`/the registered kernel; there is no computation hidden in notebook-only helper code.

Main-study command, **not part of automatic reproduction of the pilot**:

```bash
.venv/bin/python scripts/run_experiments.py --config configs/experiments/main.json
```

This means 2360 runs, 708000 search calls, 2320 assessment calls. Notebook 04 estimates runtime from measured pilot times, with an explicit warning that GP fitting grows faster than a simple evaluation-count extrapolation. No main run is launched by notebooks or smoke/pilot commands.

## Run records and recovery

`artifacts/<recipe_name>/index.csv` identifies the selected attempt for every job. `collection.json` stores the recipe, protocol, provenance and exact run IDs. A run directory name derives from its resolved job, protocol and provenance; retries append `-retryN`.

| Record | Meaning |
| --- | --- |
| `manifest.json` | Run key, timestamps/status, code/lock/data/preprocessing fingerprints, source revision/dirty state, package/hardware records, output hashes |
| `config.json` | Fully resolved job/protocol/provenance |
| `initial.json` | Q and actual initial points in z and physical log coordinates u |
| `evaluations.csv` | Every attempted simulator call, status/error, loss, z/u/θ, proposed point and timings |
| `proposals.jsonl` | Search proposals including CMA rejections and projected coordinates |
| `diagnostics.jsonl` | BO fits/acquisition warnings and hyperparameters, or CMA distribution state |
| `recommendation.json` | Best observed training point, parameters and final held-out assessments |
| `summary.json` | Status, budgets, timings, loss and boundary/assessment counts |
| `error.txt` | Failure traceback, when a run fails |

Reissue the same command after interruption. Only completed runs whose config/provenance key, file hashes, evaluation count and best loss agree are skipped. Mid-run state is not checkpointed: incomplete or failed runs are preserved and explicitly retried in new directories. Failure never creates fake successful observations. CLI exit status is nonzero if any current run failed. Investigate both the current index and retained attempts when reporting failure rates; a later successful retry does not erase the earlier failure.

Do not edit a completed run to “fix” its result. If any file changes, verification refuses reuse. A code/data/config/environment change yields a new identity. The Git dirty flag records the state at execution; code hashes identify actual simulator/optimiser/runner content. For archival publication, commit the verified implementation and rerun/freeze an identified release.

## Curated outputs and source mapping

Generated PNGs are under `results/figures/`; numerical tables under `results/tables/`. Notebook 04 writes `pilot_runs.csv`, paired-effect/boundary tables and `provenance.json` listing the exact run set. Notebook 02 reads/writes `artifacts/references/` and identifies reference settings. Data plots derive from the hashed preparation manifest. Saved notebook cells show their execution and source collection.

Ignored data/artifacts are not included in a clone. Reproduce with the commands above, or transfer the exact artifact collection and original archives separately with their manifests/checksums. Do not claim availability at a nonexistent shared storage link. The saved notebook outputs and curated tables are viewable without downloading the data, but regenerating them requires the originals.

## Historical follow-up handoff — 10 November 2026

The [research plan](RESEARCH_PLAN.md) motivated this follow-up. Existing commands
above reproduce v1. The [development diagnostic protocol](DIAGNOSTIC_PROTOCOL_V2.md)
uses a separate runner and configuration:

```bash
.venv/bin/python scripts/run_diagnostics.py
.venv/bin/python -m pytest -q tests/test_diagnostics.py
.venv/bin/python scripts/run_diagnostics.py --run
```

The first command checks readiness without downloading or running anything. The
GP integration tests must execute, not skip, in the supported experiment
environment before interpreting diagnostic output. The final command runs only
the bounded development replay/reference stages (at most 1,280 simulator calls).
Select `--stage replay` or `--stage reference` to run a component. Each execution
writes a new `artifacts/diagnostics_v2/` directory and does not overwrite v1.

An earlier 10 November environment lacked packages/data and could not resolve
PyPI or CALCE: ten isolated tests passed and two GP tests skipped there. On the
current supported local environment, all 27 pre-follow-up tests passed with no
skips, and v2 measured-data diagnostics completed. The expanded suite now passes
36 tests, including the v3 controls. Preserve these distinct dated records.
The large `configs/experiments/main.json` remains executable but is deferred as a
research step. Freeze and version confirmation before expanding the run grid.

## Known scope and submission work

V1–V4 and original V5 are complete. Aligned V5 completed 279/280 trajectories;
the report uses nineteen complete designs with the failure disclosed. Report
source, bibliography, figures and PDF are available under [report/](../report/).
The [project brief](PROJECT_BRIEF.md) lists submission requirements, including
full names/student numbers, the ten-page limit and marking access. Existing
artifacts do not establish Canvas submission or private-repository access.
[Research progression](RESEARCH_PROGRESSION.md) explains the experimental sequence.


## Follow-up execution, recovery and long-running jobs

The [v3 protocol](FOLLOWUP_PROTOCOL_V3.md) is the executable follow-up specification.
Default readiness is non-mutating:

```bash
.venv/bin/python scripts/run_followup.py
.venv/bin/python scripts/run_followup.py --stage development --run
.venv/bin/python scripts/run_followup.py --stage confirmation --run
.venv/bin/python scripts/execute_notebooks.py --start 5 --end 5
```

Pass `--v2-artifacts artifacts/diagnostics_v2/20261110T141950872864Z` to pin the
original completed v2 run explicitly. By default the newest v2 directory is
selected and must be complete. V3 development verifies its hashes, replays the
original pilot histories, and records twelve extra reference diagnostic calls.
Confirmation verifies completed development provenance and rejects changed
scientific code/configuration. The original v1 adapters and fingerprints remain
unchanged. Do not run the legacy main recipe alongside this study.

Run confirmation in a persistent process with stdout/stderr redirected to
`artifacts/followup_operations/confirmation.log`; record its PID. The runner also
records PID, timestamps, current job, completed count, elapsed allowance and final
exit code in `artifacts/rotation-followup-v3/status.json`. Invocation history,
including failures, is retained under `invocations/`; per-run checkpoints and
traces live under `confirmation/`. A process lock prevents duplicate writers.

The four-hour allowance charges original v2 runtime and every v3 invocation,
including failed retries. It excludes authoring, tests and analysis. Deadlines
are checked between calls/solvers; a running solve may overrun. The original
v2 deadline was twenty minutes; v3 development has its own twenty-minute cap
within the shared allowance. Completed runs can be reused by reissuing the same
confirmation command; partial attempts are retained and retried from scratch.
If a process was killed without finalising its invocation record, inspect the
PID, log and partial artifacts before reconciling its elapsed time. Do not reset
budget records or relabel an interrupted run as successful.

Comparable timings require no concurrent CPU-heavy jobs and unchanged scientific
source during active runs. PID, logs, status paths and estimated remaining time
provide the handoff record; recovery starts by checking that recorded state.

Initial planning estimate: v2/v3 diagnostics 2–20 minutes each, confirmation
roughly 1.5–3 hours, analysis/verification 5–15 minutes. Update from observed
method timings, retaining the hard study allowance. Estimates are not measured
results or evidence of completion.

Notebook 05 is analysis-only and uses complete planned initial-design groups for
contrasts. Partial/failed groups stay in coverage tables. Its output is a snapshot:
rerun it after completion, then verify all recorded hashes and inspect plots.
Keep v1 notebooks unchanged; rerunning them can regenerate their outputs and,
after provenance changes, launch new pilot runs rather than reuse old records.

## V4 execution and handoff

V4 has a separate scientific module/runner/configuration and artifact ID. Its
readiness command is `scripts/run_v4.py` with no flags. `--stage development --run`
executes only development and replays; `--stage confirmation --run` requires
verified development and the timing gate; `--stage all --run` chains science and
the final analysis/notebook/audit. `--stage analysis --run` spends no experiment
budget and may run after a partial or timing-gate-blocked study. Notebook 06
also regenerates only V4 analysis. Never invoke legacy notebooks for V4.

The live process writes `artifacts/rotation-followup-v4/status.json`, per-invocation
charged elapsed time, per-stage collections, complete run manifests, retry
directories and `timing_gate.json`. `artifacts/followup_v4_operations/launch.json`
identifies the detached PID; `pipeline.log` captures stdout/stderr. Four hours
covers development/replays/confirmation/assessments and failed retries, excluding
tests, authoring and analysis. A solve can overrun the deadline checked between
calls. The gate uses matched development timings plus measured collection
overhead and 25% headroom; random uses scalar-pool time conservatively. Gate
failure is an explicit outcome and starts no confirmation experiments.

Inspect status/log/PID before resuming. Reissue the appropriate stage only after
the prior process exits. Completed matching results are reused after hash and
accounting checks; invalid/incomplete runs receive a new retry suffix. An orphaned
`running` invocation blocks restart until its elapsed time has been investigated
and reconciled; never reset the ledger. Code/config/environment changes require
a new study revision. The frozen scientific hashes exclude analysis/documents,
allowing independent authoring without invalidating running science.

The baseline preservation inventory is generated before the first V4 invocation
under `artifacts/followup_v4_operations/preserved_before.json`. The final audit
checks every inventoried older artifact and notebook. Before the first V4 launch
on a fresh clone, run `.venv/bin/python scripts/record_v4_baseline.py` to create
this inventory from the older files actually present; it does not imply
that ignored artifacts were distributed. Saved run outputs can be regenerated
only with the measured inputs and recorded environment.

The historical handoff policy used persistent jobs, permitted independent
lightweight work and excluded competing CPU-heavy checks during timing.
Stage, progress, PID, log and ETA identified the state at each handoff.

### Completed V4 handoff

The detached pipeline finished on 11 November 2026 at 23:15 Singapore time
(15:15 UTC), exit code 0; PID 77037 is historical, not an active job to resume.
All 3,160 confirmation runs and 36 development runs completed. Scientific time
was 8,736.05 s; the original analysis/notebook/audit took about 95 s afterward.
A later presentation-only analysis refresh retains the same scientific run set.
Use `results/followup_v4/execution_record.json` for the committed timing/provenance
snapshot and `execution_audit.json` for the latest notebook/hash checks.
`visual_review.json` records the separately inspected figure hashes. Local raw
artifacts and downloaded inputs remain ignored; the curated outputs are committed.
The report source and PDF were prepared subsequently; current submission
requirements are summarised above.

Two older exploratory landscape images are not present in this checkout. They
have no linked generating code or notebook reference and are not required to
reproduce the reported results.

## V5 collections

The current rerun is documented in [V5_ALIGNED.md](V5_ALIGNED.md). Use
`run_v5_aligned.py` for its 280-trajectory, V4-matched design. The completed
collection contains 279 successful trajectories and one failed scalar-BO run;
the report uses nineteen complete designs. See [the result record](../results/v5-aligned/README.md).

The [original v5 protocol](FOLLOWUP_PROTOCOL_V5.md) defines the earlier
100-trajectory boundary comparison, executed on Windows on 11 November 2026.
Its exported evidence is in `results/v5/`; the commands below refer to that
original design. V5 uses the project environment
and prepared CALCE inputs; it does not require ignored v2/v3 run artifacts or
install dependencies.

Entry points, for later execution in the configured environment:

```bash
.venv/bin/python scripts/run_v5.py
.venv/bin/python -m pytest -q tests/test_v5.py
.venv/bin/python scripts/analyse_v5.py
```

The first command prints the plan without running experiments. Add `--run` to
execute without a time limit. Optional `--max-seconds N` pauses after the active
trajectory when N seconds have elapsed in this session. Another invocation can
use a longer limit or none; completed matching runs are reused and every session's
time is retained. The first Ctrl+C requests a pause after the active trajectory;
a second interrupts it. An unclosed invocation requires investigation before
restart. These are session controls, not changes to the objective-call budget.

The analysis command requires a saved v5 collection and never starts experiments.
It exports a new timestamped table directory under
`artifacts/boundary-comparison-v5/analysis/`, including coverage, the declared
contrasts and target non-attainment. It rejects concurrent execution and verifies
run hashes. The planned full design has 100 trajectories, 30,000 search calls and
400 assessment calls. These are planned counts, not a completed-run record.

For Arch Linux or Windows, use the [reproduction guide](REPRODUCING.md#native-windows-and-original-jax-v5).
Install `requirements-reproduce.txt` and add the recorded JAX versions for the
original V5 collection. `--backend jax` compiles the battery recurrence on CPU in a separate
collection; its 22 fixed-point setup checks are logged outside search calls.
Analysis accepts `--plots --plot-backend jax` for saved-result figures. No GPU
configuration is needed. Existing optimisers and FUDS assessments stay unchanged.

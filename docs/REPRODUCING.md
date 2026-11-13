# Reproducing the experiments

Start with the [root README](../readme.md) for environment installation and the
complete execution command. This guide maps each experiment to its prerequisites
and outputs. Use a fresh Git clone for a new execution: analyses deliberately
regenerate tracked notebook outputs, numerical tables and scientific plots.
Inspecting the committed evidence requires no optimisation.

`report/` contains the submitted document and its source assets; `results/`
contains numerical exports and experiment plots. Historical execution records
and saved notebook outputs retain their original `reports/` paths. Those paths
now correspond to `results/`; the archived output bytes and hashes are unchanged.
Notebook source paths have been updated for fresh execution without rerunning
or replacing the recorded outputs.

## Environment and data

The primary studies used CPython 3.13.6 on macOS arm64. The package supports
Python 3.11–3.13; the aligned comparison checks the eight numerical package
versions recorded in [the V4 reference](../configs/followup_v5/v4_reference.json).
Use `requirements-reproduce.txt`, which constrains installation to the original
lock while allowing platform-specific dependency selection. No GPU is required.
The historical original V5 used CPU JAX/JAXlib 0.12.0.1 on Windows; these are
optional for the final-report suite and required by `--suite all`.

The V3/V4 launchers use `fcntl`; the full suite therefore needs Linux, macOS or
WSL2. Native Windows can run the V5 stages with `.venv\Scripts\python.exe` in
place of `.venv/bin/python`. The new orchestration is tested separately from
the numerical experiments; no fresh full-suite execution is claimed here.

Preparation downloads three archives from
[CALCE](https://web.calce.umd.edu/batteries/data/). Their SHA-256 values are pinned
in [data_sources.json](../configs/protocols/data_sources.json):

| Archive | Role |
| --- | --- |
| `SP2_25C_DST.zip` | DST80 calibration window |
| `SP2_25C_FUDS.zip` | FUDS80 and FUDS50 held-out assessment windows |
| `SP3_25C_LC_OCV_11_16_2015.zip` | Cross-cell OCV approximation |

If downloads are unavailable, place the unchanged archives in `data/raw/` and
run preparation again. Do not extract or rename them. The original preparation
code inspects the workbook sheets, selects the 3,600-second windows and records
capacity, SOC and voltage-offset assumptions. The exact methodology remains in
[the V1 protocol](EXPERIMENTAL_PROTOCOL_V1.md).

```bash
.venv/bin/python scripts/reproduce.py --stage prepare --run
.venv/bin/python scripts/reproduce.py --stage tests --run
```

Preparation registers the `bo-vs-es` kernel in the active Python environment and
executes notebook 01 when data are absent. Existing verified data are reused,
including their timestamp, so restarting setup does not invalidate later run
provenance. Windows CRLF output is converted to LF only if that conversion
recovers the exact recorded CSV hashes; other differences stop execution.
Existing mismatched data are never silently regenerated. The tests include
synthetic optimiser/control fixtures; no full experimental grid is launched.
Inspect skips: JAX tests may skip without the optional backend, but numerical
integration tests must run in the supported experiment environment.

## Native Windows and original JAX V5

For native Windows V5 execution, create the environment with
`py -3.13 -m venv .venv`, then use `.venv\Scripts\python.exe` for the installation,
data preparation and test commands. The full V1–V4 workflow still requires WSL2
or a POSIX system. Aligned V5 uses NumPy and needs no JAX installation.

To reproduce the original V5 backend, install the recorded JAX versions after
`requirements-reproduce.txt` and run its parity tests:

```bash
.venv/bin/python -m pip install -c requirements-lock.txt 'jax==0.12.0.1' 'jaxlib==0.12.0.1'
.venv/bin/python -m pytest -q tests/test_v5.py tests/test_v5_jax.py
.venv/bin/python scripts/reproduce.py --stage v5-original --run
```

JAX tests must execute rather than skip for a missing backend. CPU JAX compiles
the same battery recurrence; GP fitting and acquisition search remain in
PyTorch/SciPy. The [V5 protocol](FOLLOWUP_PROTOCOL_V5.md#optional-jax-implementation-and-timing)
defines float64 parity, the 22 setup evaluations, compilation accounting and
synchronised timing. Analysis can additionally use `--plot-backend jax` with
`scripts/analyse_v5.py --plots`; this operates on saved losses only.

The recorded Windows execution used Python 3.13.17 and CPU JAX/JAXlib 0.12.0.1:

```powershell
.\.venv\Scripts\python.exe scripts\run_v5.py --backend jax --collection-id boundary-v5-windows-jax --run --max-seconds 10800 --execution-label raj-windows
```

It completed 100/100 trajectories in 8,154.71 s. An externally interrupted
invocation and its 262-call partial attempt remain in resource accounting.
The [execution audit](../results/v5/execution_validation.json) preserves the
hardware, versions, parity checks and attempt records. This historical command
is separate from the default reproduction collection; no JAX speed-up was measured.

## Execution order and outputs

`scripts/reproduce.py` prints its plan by default; `--check` only checks
prerequisites. `--run` explicitly starts computation. The default suite runs
`prepare → tests → v1 → v2 → v3 → v4 → v5-aligned`. `--suite all` adds
`v5-original` before aligned V5. The historical and aligned collections are
never pooled. Each stage can also be selected using `--stage NAME --run`.

| Stage | Existing entry points called | Inputs and generated outputs |
| --- | --- | --- |
| `prepare` | Kernel registration; `execute_notebooks.py --start 1 --end 1` | Three archives → four CSVs and `data/processed/manifest.json`; notebook 01 |
| `tests` | `python -m pytest -q` | Prepared data and pinned environment; inspect test and skip summary |
| `v1` | `execute_notebooks.py --start 2 --end 4` | Reference fits and 84 pilot trajectories; `artifacts/pilot_v1/`, `results/tables/`, `results/figures/`, notebooks 02–04 |
| `v2` | `run_diagnostics.py --stage all --run` | Fixed-history replays and ball reference; a new timestamped directory under `artifacts/diagnostics_v2/` |
| `v3` | `run_followup.py` development then confirmation; notebook 05 | Needs V1 raw pilot histories and a completed V2 directory; 885 confirmation trajectories under `artifacts/rotation-followup-v3/`, exports in `results/followup_v3/` |
| `v4` | `record_v4_baseline.py`; `run_v4.py --stage all --run` | Prepared data, preservation inventory, development/timing gate, 3,160 confirmation trajectories; `artifacts/rotation-followup-v4/`, `results/followup_v4/`, notebook 06 and audit |
| `v5-original` | `run_v5.py --backend jax --run`; `analyse_v5.py --plots` | 100 planned trajectories with original settings; `artifacts/boundary-comparison-v5-jax/` |
| `v5-aligned` | `run_v5_aligned.py --run`; `analyse_v5.py --plots` | 280 planned trajectories with V4-matched settings; `artifacts/boundary-comparison-v5-v4-aligned/` |

The full suite takes many hours; measured timings from different machines are
not a runtime guarantee. Runs execute sequentially with one numerical thread.
V4 alone recorded about 2 h 26 min for development and confirmation. V3/V4 retain
their original four-hour allowances and V4's forecast gate. A slower machine may
stop before completing the prescribed grid; do not bypass that gate and describe
the modified protocol as an unchanged reproduction.

The wrapper logs each command, output and exit code under
`artifacts/reproduction/`. Native runners retain their own status, invocation,
trace and provenance files. V5 analysis writes a timestamped export directory
inside its collection and prints the path. It does **not** overwrite the curated
`results/v5/` or `results/v5-aligned/` evidence. Read the new `coverage.csv`,
`run_status.csv`, `contrasts.csv` and `validation.json` there.

For complete V4 confirmation, its audit checks 796,000 search calls and 10,560
assessment calls. Aligned V5 plans 280 trajectories, 84,000 search calls and
1,120 final assessments. The submitted collection completed 279 trajectories;
the paired report analysis uses nineteen complete designs. A fresh run need not
fail at the same point or produce the same final numbers. Report actual coverage
and use the supplied complete-group rules; never discard failed records.

## Recovery and partial runs

The full-suite wrapper refuses to restart over existing experiment collections.
Repeating V2 would create a new input source for V3, and regenerating early
notebooks could invalidate V4's preservation inventory. Resume the affected
native runner instead, after reading its status/error record:

```bash
# Resume V3 confirmation after its development stage has completed.
.venv/bin/python scripts/run_followup.py --stage confirmation --run
.venv/bin/python scripts/execute_notebooks.py --start 5 --end 5

# Resume V4 through its existing provenance, budget and preservation checks.
.venv/bin/python scripts/run_v4.py --stage all --run

# Resume aligned V5 and export the resulting coverage, including failures.
.venv/bin/python scripts/run_v5_aligned.py --run
.venv/bin/python scripts/analyse_v5.py --collection artifacts/boundary-comparison-v5-v4-aligned/collection.json --plots
```

These commands use the `.venv` kernel registered during preparation. V5 supports
`--max-seconds SECONDS` to pause between trajectories; the allowance applies to
that session. A normal pause can be resumed with a different allowance. A killed
process may leave an open invocation that needs inspection. Keep error files,
attempts and charged time; do not delete them to force a successful status.
See [operations](OPERATIONS.md) for the original recovery and accounting rules.

The wrapper stops on a nonzero exit. It also checks V5 collection status and
planned-row coverage: a native runner’s successful pause is not reported as
a completed experiment. If V5 has already created a collection, it
also exports partial coverage before returning the failure. A numerical failure
is not automatically repaired or retried. The historical aligned BO failure and
its analysis implications are documented in
[the aligned evidence](../results/v5-aligned/README.md).

## Report checks and limits

The final submission files are `report/report.pdf`, `report/report.md`,
`report/references.bib` and `report/figures/`. There is one final PDF; editing
copies and document-formatting intermediates are excluded from the repository.
The [evidence map](REPORT_EVIDENCE.md) links claims to saved numerical results.

The saved-evidence audit needs only standard-library Python:

```bash
python scripts/reproduce.py --verify-results
```

It checks 100 recorded export hashes, run coverage/unique IDs and the execution
status of all six saved notebooks. It independently recomputes aligned V5’s
fourteen contrasts, incumbent histories and target summaries. Results are written
to ignored `artifacts/report-checks/verification.json`; the committed exports are
not changed. Recomputed finite floating-point values use relative and absolute tolerances of 1e-12; types, counts, identifiers and structure must agree exactly. Committed-file SHA-256 checks remain exact. This verifies saved evidence, not the missing raw executions.

With the numerical environment installed, also regenerate the twelve direct
rotation contrasts and their bootstrap intervals:

```bash
.venv/bin/python scripts/analyse_rotation_contrasts.py
```

Its default output is `artifacts/report-checks/rotation/`; `--output-dir PATH`
selects another directory. Compare the resulting CSV values with
`results/report_review/`. Hash identity is not required for a fresh CSV rendering
on a different platform, but numerical differences require investigation.
Run saved-evidence checks before replacing tables with fresh experiments.

Figures 2, 4 and 5 can be regenerated from their numerical tables using base R
(Quartz on macOS or Cairo on Linux):

```bash
Rscript scripts/plot_report_figures.R .
```

Figure 1 is a fixed crop of the archived voltage/residual plot; its finished
image is included. Notebook 02 reproduces the underlying measured-data/model-fit
plot and reference-fit table. Figure 3 and Table 4 use the saved V3 conditioning
and V4 same-box interaction exports. Validate their coverage and the source table,
then rebuild the conditioning figure and final PDF:

```bash
.venv/bin/python scripts/prepare_report_evidence.py --check
MPLCONFIGDIR=artifacts/.matplotlib .venv/bin/python scripts/prepare_report_evidence.py
.venv/bin/python scripts/build_report.py --soffice soffice
```

The figure command requires the numerical environment's Matplotlib. PDF export
requires LibreOffice (`soffice` on PATH); pass its absolute executable path with
`--soffice` when needed. The builder uses Python's standard library, the committed
Markdown and figures, and writes ODT/manifest intermediates to ignored
`artifacts/report-build/`. Only `report/report.pdf` is exported as a document.
The PDF must be visually checked and remain at most ten pages after edits;
automatic page count alone does not establish a readable layout. No optimisation
or data preparation is performed by these commands. The submission-file audit
is `results/report_review/submission_validation.json`.

These saved-evidence checks are separate from fresh experimental execution.
A fresh run can change paths, optimiser failures and
timings because operating systems, solvers and hardware differ; its manifest
records the actual environment. Do not combine newly generated V4 tables with
historical V5 exports and call them one new matched experiment.

The repository supplies experiment code, frozen configurations, notebook
analyses, saved numerical exports and the complete report source. CALCE inputs
can be downloaded and new raw runs generated. Historical raw run directories are not included. Consequently, the original
per-evaluation execution cannot be fully audited from this checkout; V5 candidate
arrays and manifests are among the missing records. Export-level consistency and fresh reproducibility
are distinct checks.

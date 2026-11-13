# Bayesian optimisation versus evolutionary strategies

DSA4212 project by **Mihir Sinha (A0273662Y)** and
**Rajvinder Dheepak Singh (A0277441B)**. See [authors](authors.md) and
[project contributions](docs/STUDENT_ENGAGEMENT.md).

We investigate how GP length scales, CMA covariance adaptation and numerical
search affect optimisation under rotations, and which boundary effects can be
resolved. The main objective calibrates a five-parameter two-RC battery model to
CALCE measurements. Domain-preserving rotations, shifted quadratics with known
conditioning, fixed-history replays, and acquisition/boundary interventions test
explanations. This is a controlled five-dimensional study, not a universal
ranking of BO and CMA-ES.

[Final report](report/report.pdf) · [Report source](report/report.md) ·
[Evidence map](docs/REPORT_EVIDENCE.md) · [Documentation guide](docs/README.md)

## Current evidence

| Study | Recorded status | Evidence and interpretation |
| --- | --- | --- |
| V1 pilot | 84 successful trajectories | Data/model checks and initial rotation diagnostics; three primary battery designs |
| V2 | Completed fixed-history diagnostics | Fitting/acquisition replays investigate scalar-BO path divergence |
| V3 | 885 confirmation trajectories | Pool, CMA sampling and conditioning controls; [protocol](docs/FOLLOWUP_PROTOCOL_V3.md) |
| V4 | 3,160 confirmation trajectories | 20 primary designs, 10 supporting designs, three rotations, five CMA replicates; [results](docs/V4_RESULTS.md) |
| Original V5 | 100 trajectories, 10 designs, three CMA replicates | Separate Windows/JAX study; [historical exports](results/v5/README.md) |
| Aligned V5 | **279/280 completed; 19 complete designs** | NumPy and V4-aligned streams/settings; scalar BO failed on design 21 after 166 calls; [exports and limitations](results/v5-aligned/README.md) |
| Report-stage extension | 12 post hoc method contrasts | Paired differences of V4 rotation effects; [all estimates and provenance](results/report_review/README.md) |

The report uses V4 for rotation and aligned V5 for matched-budget BO/CMA quality.
Original and aligned V5 are not pooled. ARD has larger signed rotation effects
than scalar BO; diagonal CMA has larger effects than full CMA at 300 calls, with
mixed comparisons at 100. Scalar pools restore path agreement but can worsen
quality. Same-box boundary interactions remain unresolved; projection's benefit
on identity-ball quality is a different result. See
[interpretations](docs/REPORT_INTERPRETATIONS.md) for uncertainty and limitations.

## Reproduce the study

Use a separate Git clone for fresh experiments: the notebook analyses regenerate
tracked tables, figures and notebook outputs. The commands below are for Linux
or macOS; Windows users can run the full workflow in WSL2. Native Windows supports
V5, but the original V3/V4 launchers use POSIX file locks. Computation uses the CPU;
no GPU setup is required.

Create a Python 3.13 environment (the primary recorded version is 3.13.6):

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproduce.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/reproduce.py --check
```

The requirements file installs the project with the recorded dependency versions
as constraints, without forcing macOS-only packages onto other platforms. The
preflight checks the numerical versions and frozen scientific inputs. Missing
CALCE data are reported as a preparation step; an existing mismatched input is
an error. Do not remove version constraints to bypass an installation problem.

```bash
.venv/bin/python scripts/reproduce.py
.venv/bin/python scripts/reproduce.py --run
```

The first command prints the plan. The second registers the environment's
notebook kernel, downloads and prepares the three hash-checked CALCE archives,
runs the tests, then executes **V1 → V2 → V3 → V4 → aligned V5**, including their
notebook analyses and V5 export. This covers the experiments used in the final
report and can take many hours. Logs go to `artifacts/reproduction/`.
The frozen V3/V4 time allowances still apply; a slower machine may reach a gate.

To also reproduce the separate, historical JAX V5 study, install its recorded CPU
backend and select the full suite **before starting a fresh run**:

```bash
.venv/bin/python -m pip install -c requirements-lock.txt 'jax==0.12.0.1' 'jaxlib==0.12.0.1'
.venv/bin/python scripts/reproduce.py --suite all --check
.venv/bin/python scripts/reproduce.py --suite all --run
```

[Reproduction guide](docs/REPRODUCING.md) gives individual stage commands, data
sources, output locations, expected counts and recovery instructions. Existing
scientific runners retain their configurations and failure handling. The deferred
`configs/experiments/main.json` is an unrun V1 proposal and is excluded.

## Check the submitted results without optimisation

The committed CSVs and six executed notebooks can be inspected directly. A
standard-library Python installation is enough for the saved-evidence audit:

```bash
python scripts/reproduce.py --verify-results
```

This checks 100 recorded export hashes, coverage and unique run IDs across the
five trajectory collections, and executed outputs in all six notebooks. It also
recomputes the fourteen aligned V5 contrasts, incumbent histories and target
summaries. Floating-point recomputations allow 1e-12 relative/absolute roundoff; counts, identities and committed export hashes remain exact. It downloads no data, evaluates no objectives and writes checks only
to ignored `artifacts/report-checks/`. It does not replace a fresh experiment or
audit the absent historical raw runs.

The [evidence map](docs/REPORT_EVIDENCE.md) connects each report result to the
relevant configuration, code and saved table. Recomputing the rotation bootstrap
requires the project environment; figure-generation commands and output locations
are in the [reproduction guide](docs/REPRODUCING.md#report-checks-and-limits).

To rebuild the added conditioning display and check all same-box boundary
intervals, then export the report (project environment and LibreOffice required):

```bash
.venv/bin/python scripts/prepare_report_evidence.py
.venv/bin/python scripts/build_report.py --soffice soffice
```

These commands use saved exports only. The report includes the exploratory V3
conditioning trend (Figure 3) and all twelve unresolved V4 boundary interactions
(Table 4). The build guide also gives the R command for Figures 2, 4 and 5.


| Location | Contents |
| --- | --- |
| `src/bo_vs_es/`, `scripts/` | Battery model, optimisers, experiment runners and analysis |
| `configs/`, `tests/` | Versioned experimental settings and verification tests |
| `notebooks/` | Executed data/model inspection, pilot and V3/V4 analyses |
| `results/` | Saved numerical results, plots and execution/analysis records |
| `report/` | Final PDF, complete Markdown source, bibliography and figures |
| `docs/` | Protocols, scientific interpretations, literature and project contributions |
| `data/`, `artifacts/` | Local downloaded/prepared inputs and new raw runs; Git-ignored |

## Reproducibility boundary and submission

Saved exports support checking the reported quantities. Fresh experiments require
the CALCE archives and pinned environment; the runners save full traces and
provenance for new runs. Historical raw run directories are not distributed, so the checkout cannot
fully audit the original per-evaluation execution. In particular, V5’s original
candidate arrays and manifests are absent. Matching
seeds and settings across machines does not guarantee identical trajectories,
failures or timings. The original and aligned V5 collections remain separate.

The final [PDF](report/report.pdf) has ten pages including references; author
identities and the repository URL appear at its start. The repository retains
complete report source and scientific plotting code, as required by the
[assignment guidelines](https://alexxthiery.github.io/teaching/guidelines/guidelines.html).
Group review, private-repository access for `alexxthiery`, and the single group
Canvas submission remain the authors' final steps.

## AI disclosure

AI agents were used as tools, in line with the course's AI-assistance guidelines, to generate figures, assist with literature searches for relevant articles, and assist with coding.

Reproducibility of the experiments was verified by the team.

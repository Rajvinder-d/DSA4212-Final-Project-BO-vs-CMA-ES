# Implementation validation — pilot through aligned V5

Current coverage: V1–V4 and original V5 are complete; aligned V5 completed
279/280 trajectories, with nineteen complete designs used for inference. The
final section records the local export audit and its verification limits. Earlier
checks and handoffs retain their dates, environments and historical test counts.

This records implementation checks and study scope, not a submission report. The current machine-readable audit is [`submission_validation.json`](../results/report_review/submission_validation.json). Reproduction commands are in [REPRODUCING.md](REPRODUCING.md).

## Current revision checks — 13 November 2026

The revised report contains five figures and five tables in ten A4 pages. The
added V3 conditioning display retains both rotations and a scalar-BO baseline;
the V4 same-box table retains all twelve intervals, including unresolved results.
All ten rendered pages were visually inspected after the final PDF build, with
identities, repository hyperlink, equations and figure/table numbering checked.
The restored build command uses the Markdown source and included figures.

The full numerical test suite passed on Python 3.13.6: **98 passed, one optional
JAX test skipped, nineteen subtests passed**. The saved-evidence command passed
all 100 committed export hashes, collection coverage and six executed notebooks,
and recomputed the fourteen aligned-V5 contrasts. JSON numerical comparisons
now tolerate finite floating-point differences at 1e-12 relative/absolute scale;
counts, types, identifiers and file hashes remain exact. Controlled-clock tests
cover deadlines before execution, during setup and after four evaluations.

The twelve rotation-method intervals and 240 design-level differences were
recomputed and matched. The new displays were checked against saved evidence,
including coverage and malformed-input tests. Historical numerical exports and
all six notebooks’ code, execution counts and outputs remain unchanged. Narrative
cells and project records now describe our completed work and findings.

The existing numerical environment and prepared data pass preflight. No fresh
environment installation, new optimisation experiments, GitHub publication or
Canvas submission was performed. Historical missing candidate arrays/manifests
remain an audit limitation. The earlier checks below retain their historical
scope and do not describe the current revision’s test status.

## Earlier submission audit — 13 November 2026

Reviewed against the [assignment guidelines](https://alexxthiery.github.io/teaching/guidelines/guidelines.html)
and [project brief](https://alexxthiery.github.io/teaching/bo-vs-es/bo-vs-es.html).
The controls address the central behavioural-explanation requirement: rotation
sensitivity, numerical selection and path agreement, and a boundary intervention
with matched-call quality/cost comparisons. No new experiments are needed merely
to add another result. The remaining scientific limits are explicitly scoped:
selected rotations and one battery window, pointwise intervals, post hoc contrasts,
representation/fitting entanglement, changed adaptation under projection and
complete-case BO–CMA comparisons. The report does not claim equal-time performance
or independent confirmation from the aligned extension.

| Area | Finding and action | Remaining limit |
| --- | --- | --- |
| Required submission files | One ten-page PDF, complete Markdown, bibliography and figures; names/student numbers and repository URL verified | Group approval, marking access and Canvas submission are external actions |
| Report-to-evidence traceability | Replaced the mixed historical/current evidence map with a claim–control–code–export map for the final report | Historical raw run directories are absent |
| Saved-result verification | Added `python scripts/reproduce.py --verify-results`: 100 output hashes, five collection coverages, six executed notebooks and fourteen independently recomputed aligned contrasts passed | Hash/summary checks do not audit absent raw candidates or reproduce numerical solver paths |
| Safe analysis output | Both report-analysis scripts now default to ignored output rather than overwriting submitted evidence; inference is unchanged | Its NumPy/pandas bootstrap was not re-executed in this environment |
| Execution status | Wrapper now checks V5 collection status and planned-row coverage; a native runner’s successful pause cannot be reported as completion | Full-suite execution has not been revalidated from a fresh supported environment |
| Student contributions | Removed obsolete identity-pending text; recorded Raj’s explicit aligned-rerun, execution and later report-review work | AI tool use is disclosed in the README; this record does not assess completeness of background reading |
| Source/PDF agreement | Removed the internal attribution disclosure; aligned Mihir-first authorship; rebuilt the PDF and editable Word from Markdown; ten A4 pages retained, parsed and visually checked | This is a document correction, not a new experimental result |

**Verification performed:** fourteen standard-library orchestration/audit tests
passed on Python 3.12.10 and 3.14.7. The saved-evidence command passed on the current
machine. The R plotting script parses; its plotting code and numerical inputs
are unchanged. The rotation inference function was compared structurally with its
previous version and is unchanged; only its output handling was edited. Frozen
V4-alignment source/configuration/package-lock hashes remain intact. Local links,
report structure and tracked-file whitespace were checked.

**Not completed locally:** a clean numerical-environment installation and full
scientific test/run sequence. Attempting to obtain Python 3.13 failed because this
shell could not resolve GitHub’s hostname. This is an environment/network blocker,
not evidence that the recorded dependencies are invalid. No download, numerical
suite or long experiment is represented as successful. No historical raw collection
was found in the local DSA4212 project directories. Access for `alexxthiery` has
not been verified.

The records below are dated history. Their old file layouts, test counts and
renderer descriptions must not be read as the current submission state.

## V1 pilot validation — historical

- Verified original CALCE ZIPs and prepared all selected measured traces. The OCV workbook had one identical overlapping channel record, removed with an audit; the selected dynamic workbooks had none. Native times and current-sign conversion are retained.
- Executed **84/84** current pilot runs successfully: 30 primary battery-ball runs, 14 battery-box/boundary runs, 30 quadratic controls and 10 uncorrected-OCV sensitivity runs.
- Verified **8400** search calls and **108** final held-out assessment calls. Separate reference fits and their diagnostics are identified in notebook 02.
- Executed all four notebooks in order from fresh kernels, then refreshed notebook 04 after making method colours consistent. Saved outputs contain no cell errors. The final notebook pass reused only verified pilot results.
- Passed **15 tests**, including measured-data integration, analytic RC response and local synthetic parameter recovery, branch symmetry, geometry/covariance/kernel transformations, seeded candidates, failure accounting, verified reuse and assessment only after search.
- Verified every selected run's output hashes, index-to-summary agreement, candidate domains/transforms, matched initial observations and proposal counts. Current simulator/optimiser/runner fingerprint, dependency lock and prepared data/metadata match run provenance.
- Generated 12 curated scientific figures plus numerical tables, run IDs, optimiser-health records and parameter diagnostics. Inspected the data/OCV, reference voltage/residual, convergence, held-out, internal-state, path-divergence and quadratic figures. `pip check` and `git diff --check` passed.

The recorded search times sum to approximately **14.45 minutes** on this local CPU, excluding data loading/preparation and notebook startup. Notebook 04's simple method-specific extrapolation is approximately **19.88 hours** for the configured main study; larger GP fits may increase that substantially. This estimate is not a measured main-study runtime.

## Implementation issues investigated

The initial notebook launch was blocked by sandbox restrictions on local kernel sockets; execution succeeded with local Jupyter permission. The downloader supports the platform curl CA store when Python's certificate store fails, retaining TLS verification. Workbook extensions were checked against actual OOXML content rather than assumed to imply legacy Excel.

An initial test exposed pandas' read-only NumPy view behaviour; the isolation test now replaces its test vector rather than mutating that view. Inspection of pycma revealed a small full-CMA axis perturbation and different default diagonal learning rates. Both were corrected to match the approved comparison and covered by tests before the final pilot was accepted. Earlier implementation-check runs remain under ignored artifacts, with their previous index/collection preserved in `artifacts/implementation_checks/pre_cma_audit/`; the current collection identifies only the accepted 84 runs.

## What the pilot establishes, and does not

The measured-data pipeline and controlled experiment execute reproducibly with recorded provenance. BO, CMA and random search produce feasible recommendations and independently assessed FUDS predictions. The full-box reference is outside the restricted ball, so its score is not a feasible ball target.

The three-seed pilot does not establish the expected ranking of rotation sensitivity. In particular, isotropic BO's small initial mapped-candidate differences can develop into different sequential search paths. A frozen-history control gives posterior/LogEI agreement near floating-point precision, isolating a mathematical invariance that the full numerical search does not necessarily preserve. Random-search paths remain physically matched. This supports investigating numerical fitting/acquisition behaviour; it does not uniquely identify a solver cause. Some ARD length scales approach their declared upper bound, another useful sensitivity question for a future protocol revision.

CMA absolute paired changes also include sampling variability because common integer seeds do not couple physical Gaussian draws pathwise. The single-seed boundary and quadratic demonstrations cannot establish general effects. Cross-cell OCV transfer, constant circuit parameters, fixed SOC/capacity and one selected dynamic cell limit predictive and physical interpretation.

The **2360-run V1 main configuration remains deferred and unexecuted**. Focused
follow-up replaced it as the research direction. The report source and PDF now
exist; submission identity and access requirements are tracked in the
[project brief](PROJECT_BRIEF.md) and [authors record](../authors.md).


## Subsequent validation — 10 November 2026

A separate local review independently verified all 84 pilot artifact sets, their
8,400 search calls, 108 assessment calls, source/data/notebook/collection hashes
and zero saved notebook errors. All 27 then-current tests passed, including GP
integration; diagnostic preflight was ready. This updates current readiness,
not the historical 15-test delivery record above.

The v2 measured-data diagnostic completed and retains its own manifest and
callback traces. V3 adds original-history and finite-pool replays, CMA search
replicates, checkpoint assessment, deadline/retry handling and grouped analysis.
The expanded suite passes 36 tests with one upstream Torch deprecation warning.
Its exact execution log is `artifacts/followup_operations/tests.log`.
Confirmation status and analysis validation are separate: see
`artifacts/rotation-followup-v3/status.json` and
`results/followup_v3/validation.json`. Do not infer completed confirmation from
passing tests.

Notebook 05 executed all four code cells in a fresh local kernel with zero cell
errors; the mechanism figure was visually inspected. Original notebook hashes
and the v1 source fingerprint remain unchanged. Markdown link checks and
`git diff --check` passed. Confirmation was launched as a persistent process;
its completion and final analysis remain pending. The exact handoff audit is
`results/followup_v3/implementation_validation.json`.


## Confirmation completion — 11 November 2026

All 885 planned v3 trajectories completed successfully on 10 November at
23:57 Singapore time. Confirmation elapsed time was 4,590.42 seconds (76.51
minutes), within the four-hour allowance. The complete grid contains 196,500
search calls and 2,400 separately counted assessment calls. Final analysis is
regenerated by notebook 05; its hash/accounting audit is
`results/followup_v3/validation.json`. Earlier running/pending statements describe
the dated handoff, not the current execution state. Current interpretation is
maintained in [Report interpretations](REPORT_INTERPRETATIONS.md).

## Analysis presentation review — 11 November 2026

The revised notebook 05 executed all nine code cells in a fresh local kernel,
with zero saved cell errors and no truncated HTML tables. Seven figures are
displayed, including the new sequential-path and CMA-variability plots; the two
new figures were visually inspected. Execution needed local Jupyter loopback
permission; headless plot generation uses the Agg backend.

All 37 tests passed (one upstream Torch deprecation warning). The added test
checks pairing before bootstrap, unequal replicate counts, missing extensions
and duplicate checkpoint rejection. All 35 curated output hashes and the current
analysis-source hash match the regenerated audit. The original v3 collection hash
is unchanged; historical CSV outputs and notebooks 01–04 are unchanged.
`git diff --check` passed. No new optimisation experiment was run. Paired budget
contrasts added in this review are explicitly post-confirmation exploratory.

## V4 implementation and launch — 11 November 2026

All 48 tests passed before V4 launch, including baseline continuous/scalar-pool
parity, scalar rotation consistency, ARD metric dependence, pool indexing/ties,
bound settings, exact development/confirmation counts, failure/deadline/retry
handling, timing-gate arithmetic, kernel-by-search and boundary-budget contrasts,
and cross-block complete-group coverage. One upstream Torch deprecation warning
remains. A repository-wide recipe test was extended to recognise the separate V4
grid; original scientific adapters were not edited.

Readiness passed on Python 3.13.6 with the existing dependencies/prepared inputs.
An inventory captured 9,305 pre-existing files for preservation checking. The
detached process launched under PID 77037, with the log and launch record in
`artifacts/followup_v4_operations/`. The actual stage, timing gate and completion
are in `artifacts/rotation-followup-v4/status.json`. Notebook 06 is implemented
and will execute automatically after the scientific stage; its output and visual
verification are pending. This record establishes implementation and launch,
not completed confirmation at that date.

## V4 final validation — 11 November 2026

All 3,160 confirmation trajectories and 36 development trajectories completed
without failed trajectories/retries. Confirmation accounting is exactly 796,000
search and 10,560 assessment calls; development adds 6,000 and 72 respectively.
Seventy-two replay records add no simulator calls. The original invocation used
8,736.05 seconds of the four-hour allowance and passed its timing gate.

The final suite passed **48 tests**, with one upstream Torch deprecation warning.
The final analysis refresh executed notebook 06's nine code cells in a fresh
kernel, with zero saved errors and no truncated HTML tables. All 44 generated
output hashes were verified. All five V4 scientific figures were inspected;
the rotation figure uses separate y-scales and legends outside the data panels
so battery uncertainty remains visible. Scientific source hashes still match
the recorded 14-file dependency set, and all 9,305 files in the pre-V4 inventory
remain unchanged. The separately requested notebook-05/V3 presentation changes
predate that V4 preservation inventory and do not change the original raw runs.

The final analysis includes fitted replay-scale/status tables, numerical warning
and STOPPED-fit counts, all planned paired intervals and coverage. The committed
`results/followup_v4/execution_record.json` preserves scientific timing/provenance;
`execution_audit.json` and `visual_review.json` identify the final presentation.
The latter records the visual inspection. A future
analysis regeneration requires a fresh visual review before replacing its record.
Local Markdown links and `git diff --check` passed at that review. Current
submission requirements are listed in [the project brief](PROJECT_BRIEF.md).


## V5 execution validation — 12 November 2026

The v5 additions were executed on a second machine: Windows 10 x64 (AMD Ryzen 5 5500,
CPU), CPython 3.13.17, the locked numerical packages and CPU JAX/JAXlib 0.12.0.1 in
float64 mode. Before search, the 11-point JAX/NumPy loss comparison passed (maximum
absolute gap 2.6e-18). The full suite passes **49 tests**, including 13 v5/JAX tests;
`pip check` and `git diff --check` pass.

CALCE ZIP hashes match `configs/protocols/data_sources.json`; regenerated processed CSVs
matched the recorded v1 hashes after CRLF→LF normalisation (the documented Windows
line-ending issue). The v1 readiness preflight reports `ready: true` with no blockers,
and the source fingerprint still matches `results/tables/delivery_validation.json`.

The fixed 100-trajectory design completed **100/100 runs across all ten design groups
with zero exclusions**: 30,000 counted search calls and 400 assessment calls in
**8,154.71 s (135.91 min)**, within the 10,800 s session limit. The analysis verifier
reports `status: complete`, `excluded_design_groups: 0` and matching output hashes.
Across two invocations, accounting includes 44 setup calls and one 262-call unclosed
partial attempt from an externally stopped process; the attempt is retained and is not
one of the planned runs. Full details:
[`results/v5/execution_validation.json`](../results/v5/execution_validation.json). The
verified result tables are committed in [`results/v5/`](../results/v5/README.md) with the
verifier's hashes; the raw collection stays ignored.

This historical entry validates original V5 execution and accounting. Scientific
interpretation and the later aligned results are maintained in
[Report interpretations](REPORT_INTERPRETATIONS.md).

## V5 integration validation — 12 November 2026

The V5 implementation, configurations and tests were integrated without
changes to their scientific logic. Nine aligned-design tests passed locally;
four numerical integration tests were skipped because Torch, BoTorch and CMA
are absent. Pytest and JAX are also unavailable, so the full numerical/JAX suite
was not rerun. No measured-data experiment was executed during the integration.
The passing checks include V4 source fingerprints, shared design/search streams,
paired handler counts, configuration guards and the planning-only CLI.

Duplicate aligned exports were compared after newline normalisation and retained
with the report's verified CRLF bytes. All recorded report/evidence hashes match;
the ten-page PDF and both Word drafts are unchanged by this integration. Historical
V4 and V5 validation counts above refer to their recorded environments, not a
new combined-suite execution.

## Aligned V5 test-fixture history — 12 November 2026

Historical pre-execution validation on 12 November 2026 passed nine planning, seed-parity,
source-integrity, contrast and input-drift tests. Four synthetic integration
checks are supplied for executor selection/resumption, candidate-prefix parity,
initial/acquisition streams and handler/export verification. They were skipped
locally because the numerical stack was absent; dependency installation failed
at DNS resolution. No CALCE experiment was run in that validation environment.

The subsequent Windows test run reported 18 passes and one failure in the
acquisition-prefix fixture: BoTorch rejected both final restart candidates as
infeasible. That fixture had reduced GP/acquisition iterations to 2/2 and used
only 16 initial candidates with two restarts. It therefore did not exercise
the production solver settings. The corrected fixture retains the full 60/80
iteration limits, 256 initial candidates, eight restarts and original pool size;
only trajectory lengths are shortened. It compares both post-initial proposals
at stream steps 20 and 21 and reports the BO variant on failure. This correction
does not modify production code. The subsequent Windows test report recorded a pass for all three BO variants in 12.09 s. The later production run still encountered an eight-restart infeasibility failure on design 21; the result record distinguishes that failure from the earlier fixture issue.

## Documentation audit and record consolidation — 13 November 2026

Our implementation, execution and analysis checks are recorded here. Research design and member roles are summarised in [Project contributions](STUDENT_ENGAGEMENT.md).

The former engagement workbook's execution references are consolidated here:
V2 development records are under `artifacts/diagnostics_v2/`; the historical
follow-up test log is `artifacts/followup_operations/tests.log`. V3 status and
analysis are recorded in `artifacts/rotation-followup-v3/status.json` and
[the V3 audit](../results/followup_v3/validation.json). These local artifact paths
are not promised as files available in a clean clone.

The pre-cleanup local audit passed 67 tests with one skipped test and nine
passing subtests. It verified 98 recorded export hashes: 35 V3, 44 V4, seven
original V5 and twelve aligned V5. All fourteen aligned contrasts and bootstrap
intervals, 279 saved loss sequences, checkpoint minima, target summaries and
coverage/accounting were recomputed from exports without running optimisation.
The current aligned analysis, contrast and alignment source hashes matched the
export audit. Raw V5 manifests, candidate arrays and the original failure log
were unavailable locally; export consistency does not independently verify
feasibility or simulator outputs. The [failure note](../results/v5-aligned/failure_note.md)
is an attributed transcription.

Aligned V5 completed 279/280 trajectories. Nineteen complete designs are used in
paired inference; the omitted design is 21. The collection records 83,866 search
calls and 1,116 assessments. The [aligned audit](../results/v5-aligned/validation.json)
retains `status: partial`; [report-stage checks](../results/v5-aligned/report_checks.json)
verify the qualified analysis without changing that status.

Documentation cleanup checks passed: the reading guide links all 22 other
Markdown documents, local Markdown links and script/test command paths resolve,
and Raj's original contribution section is byte-for-byte unchanged. The updated
quality/time table matches complete-design export medians. All 98 export hashes
remain unchanged, and `git diff --check` passes. This cleanup changed only
Markdown documentation; no experiments or notebooks were executed.

## Report and reproduction revision — 13 November 2026

This revision adds a report-stage analysis of twelve direct V4 battery
rotation-effect differences, explicitly post hoc. Historical experiment code,
exports and executed notebooks are preserved. The new analysis validates the
full 20-design arm grid, pairs before averaging five CMA replicates, and uses
5,000 design-bootstrap resamples with seed 4129. Tests exercise known arithmetic,
replicate aggregation, row-order independence, deterministic intervals and
rejection of missing, duplicate, invalid or inconsistent evidence.

The local suite passed **77 tests and nine subtests**, with **one optional JAX
module skipped** because JAX is absent and one Torch deprecation warning. The
aligned saved-export audit recomputed all fourteen contrasts, 279 incumbent
histories/checkpoints and target summaries without objective evaluation. All
98 hashes listed in the V3, V4, original V5 and aligned V5 export audits match.
A separate before/after inventory verified 151 existing report-result/notebook
files and the older Word draft were unchanged. The two older exploratory landscape
images referenced in earlier handoffs are not present in this checkout.

The canonical report and editable Word copy include both member names, student
numbers and the repository URL. The rebuilt PDF and rendered Word have ten A4
pages; every page was visually inspected. Source and Word body text agree after
normalising whitespace, and the first-page repository link and identities were
checked structurally in both exports.
`results/report_review/submission_validation.json` records current hashes and
remaining limitations.

The new portable LibreOffice exporter was exercised on macOS using an isolated
profile and preserves the older Word draft. A temporary R figure-rebuild check
exposed the old script's silent failure when Cairo/X11 libraries were unavailable.
The corrected script uses Quartz on macOS, Cairo elsewhere, stages each output
and rejects a failed device or missing file. All six figure 2–4 PNG/PDF outputs
then rebuilt in a temporary directory; the three PNGs were visually inspected.
Committed figure assets were retained. Figure 1's source/asset hashes were
preserved; its crop command was not rerun here because ImageMagick 7 is absent.
The README lists that dependency and a build path using committed figures.

No full optimisation collection or notebook execution was launched for this
revision. These checks verify saved-result analysis and document consistency,
not the missing historical candidate arrays or raw execution manifests.

## Final repository preparation — 13 November 2026

The final-report cleanup synchronised the corrected author identity, retained
one PDF and one editable Word report, and removed superseded drafts, outlines,
unused figure variants and empty-directory placeholders. The custom document
builder/exporters were then removed; the complete Markdown report, bibliography,
current figures and scientific analysis/plotting scripts remain. Both submission
exports were checked at ten A4 pages before the formatting helpers were removed.
Earlier build/validation entries above describe their historical state.

A standard-library orchestration entry point, `scripts/reproduce.py`, now exposes
plans and read-only preflight checks, then delegates explicit execution to the
existing runners. It orders preparation, tests and V1–V4 before aligned V5; the
optional full suite adds the separate historical JAX V5. It checks locked
numerical versions, frozen input hashes and native-platform constraints, registers
the notebook kernel, preserves existing prepared data and exports partial V5
coverage when a run fails. No scientific runner, optimiser, objective, configuration
or recorded numerical result was changed. `requirements-reproduce.txt` constrains
project installation without installing every platform-specific distribution from
the original macOS freeze. [The reproduction guide](REPRODUCING.md) documents
inputs, outputs and recovery.

Verification for this cleanup:

- **11 orchestration tests passed** on CPython 3.12.10 and 3.14.7. These standard-library
  tests cover execution guards, dependency order, prerequisite failures, preservation
  of prepared data, exact-hash CSV newline conversion and partial V5 export handling.
  Passing on 3.14 does not make that interpreter supported for the experiments.
- The actual preflight correctly rejects this workstation's Python 3.14.7 and
  missing/mismatched numerical dependencies. The plan prints without evaluating
  objectives or importing the scientific stack.
- All **98 exported-file hashes** in the V3/V4/original-V5/aligned-V5 audits match;
  **200 retained scientific/evidence files** match the pre-cleanup inventory.
  The removed inventory entries are placeholders and the obsolete report outline.
- All **15 frozen source/protocol/lock hashes** in the aligned V5 reference match.
  The saved-export analysis reproduces the committed aligned report-check JSON.
- The PDF remains ten A4 pages, with both corrected author identities and repository
  URL at the start. Retained local Markdown links resolve and `git diff --check` passes.

This is an orchestration and saved-evidence audit, not a new end-to-end scientific
reproduction. No dataset download, optimiser collection or notebook execution was
launched during repository preparation. A fresh supported environment must still
execute the numerical suite and the experiments; the historical raw V5 artifact
limitation and marking-access check remain explicit.

### Submission-folder cleanup

`report/` now contains only the final PDF, complete Markdown source, bibliography
and figure assets. The editable Word file was moved outside the repository; local
render previews/intermediates and the image-cropping helper were removed. Scientific
analysis and plotting scripts moved unchanged to `scripts/`; their commands and
the rotation-analysis test import were updated. The evidence map is now
`docs/REPORT_EVIDENCE.md`, and the submission audit is
`results/report_review/submission_validation.json`. Historical numerical manifests
retain their original source paths; the relocated script bytes still match their
recorded hashes. The scientific data, optimiser settings and reported values did
not change during this layout cleanup.

## PDF-only report refresh — 13 November 2026

Applied five targeted report clarifications, corrected the boundary-study
provenance wording and updated the contribution record. The PDF was rebuilt from `report/report.md` with the archived ODT builder
and LibreOffice PDF export; it remains ten A4 pages. The editable Word copy was
not rebuilt for this PDF-only revision. Submission-audit hashes were refreshed;
no experiment or numerical result changed.

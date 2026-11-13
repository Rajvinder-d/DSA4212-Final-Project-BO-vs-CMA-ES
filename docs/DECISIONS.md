# Shared decisions

This dated record preserves decisions and their rationale. Earlier open or
pending entries describe their historical state and are superseded by later
entries. [Research progression](RESEARCH_PROGRESSION.md) connects the decisions
to findings; [the documentation guide](README.md) identifies current evidence.
Contribution attribution is maintained in [Project contributions](STUDENT_ENGAGEMENT.md).

## Preparation record — 6 November 2026

| Date | Decision | Basis |
| --- | --- | --- |
| 2026-11-06 | Prepare a DSA4212 Bayesian/evolutionary optimisation project | Group selection of the assignment topic |
| 2026-11-06 | The group planned a private remote and teammate access | Preparation-stage repository decision; remote creation remained pending |
| 2026-11-06 | Use the linked teaching page as the project brief | Course project brief selected by the group |
| 2026-11-06 | Separate shared code, configuration, analysis, report and operations documentation | Prepared repository convention |

## Initial decisions (historical; superseded where noted below)

| ID | Decision | Historical status |
| --- | --- | --- |
| D01 | Primary explanation and controlled intervention | Proposed geometry direction; team to decide |
| D02 | Algorithms, libraries and baseline | Open / unassigned |
| D03 | Objective family and problem instances | Open / unassigned |
| D04 | Budgets, initialisation, seeds and tuning allowance | Open / unassigned |
| D05 | Final recommendation, measures and uncertainty analysis | Open / unassigned |
| D06 | Compute environment and shared artifact location | Open / unassigned |
| D07 | Member responsibilities and review assignments | Open / unassigned |
| D08 | Deadline, deliverables and submission requirements | Await course information |

When accepting an experiment design, link the agreed protocol and config.
Do not silently reinterpret an old result under a new definition. Record the
reason when replacing a decision, and update affected README/report claims.

## Accepted implementation decisions — 9 November 2026

| ID | Accepted decision | Basis / location |
| --- | --- | --- |
| D01 | Rotation sensitivity with preserved domains, independent quadratic conditioning, CMA boundary ablation | Group research design; [protocol v1](EXPERIMENTAL_PROTOCOL_V1.md) |
| D02 | Matérn-5/2 scalar/ARD LogEI, full/diagonal pycma, random; residual least squares reference | [Literature review](LITERATURE_REVIEW.md) |
| D03 | Measured CALCE SP20-2 DST/FUDS plus SP20-3 low-current OCV, 25°C; two-RC and synthetic controls | Verified archives and the group’s declared cross-cell OCV approximation |
| D04 | 84-run CPU pilot, 100 calls/20 initial; separate 2360-run main recipe, 300 calls/20 seeds, unrun | The group selected an executable local pilot and a separate full-study recipe |
| D05 | Calibration loss, paired rotation effects and frozen held-out RMSE; no known battery global regret | Protocol sections 1 and 8 |
| D06 | Local isolated Python environment, exact lock, ignored data/artifacts and curated notebook/plot/table outputs | [Operations](OPERATIONS.md) |
| D07 | Member responsibilities were not yet recorded at this stage | Superseded by the completed contribution record |
| D08 | Follow provided course submission requirements; final report/membership/access tasks remain | Preparation-stage scope; later entries record completed reporting |

Implementation audit found pycma's small axis-dependent initial covariance perturbation and diagonal learning-rate defaults. Before accepting the final pilot, the adapter was corrected to start isotropically and match path/covariance learning-rate constants. Initial implementation-check artifacts are retained; only the current fingerprinted collection is analysed. Native diagonal update mechanics still differ and are disclosed. This is an implementation correction to the approved comparison, not post-result tuning for a preferred winner.

V1 freezes the source files, stage/window rules, OCV transfer/offset, parameter bounds and search settings. Subsequent scientific changes require a new protocol/configuration revision and new run identities. Earlier general planning proposals are preserved, not interpreted as completed experiments.

## Follow-up direction — 10 November 2026

The group reviewed the brief and pilot and selected further investigation through
rigour and explanation; novelty of the application is not the selection criterion.

| ID | Decision | Status and consequence |
| --- | --- | --- |
| D09 | Retain battery calibration, the five-method comparison and supporting quadratic controls | Accepted research direction; preserve v1 evidence |
| D10 | Prioritise fixed-history BO fitting/acquisition replays and a targeted numerical intervention | Planned follow-up; exact intervention depends on diagnostic evidence and must be frozen before confirmation |
| D11 | Separate CMA sampling variation from orientation effects; obtain a ball-constrained reference | Ball-reference runner now implemented; measured-data execution and CMA sampling controls pending |
| D12 | Report quality against calls and elapsed time; optionally estimate target-specific cost crossovers | Secondary analysis; equal-time superiority and crossover values are not established |
| D13 | Defer the 2,360-run v1 main grid until focused diagnostics inform the confirmation design | Supersedes treating the existing main recipe as the immediate next step; retain the recipe unchanged |
| D14 | Separate development histories/seeds/rotations from confirmation and version changed scientific settings | Planned design requirement; no new numerical budget, split or run result is asserted |

The [research plan](RESEARCH_PLAN.md) defines the evidence, experiment sequence
and remaining design choices. This initial direction-setting update changed
documentation only; the subsequent diagnostic implementation is recorded below.
Original pilot records, configurations and notebook outputs are preserved.

## Diagnostic implementation — 10 November 2026

The group selected the bounded
[development diagnostic protocol v2](DIAGNOSTIC_PROTOCOL_V2.md) to implement the
BO replay and ball-reference portions of D10/D11. Development cases and reserved
confirmation cases are explicit in `configs/diagnostics/development_v2.json`.
The group’s development allowance is at most 80 history-generation calls and
1,200 reference callbacks, with a 1,200-second deadline checked between solver
calls. This is a development allowance, separate from the deferred large main grid.

Ten numerical/control tests pass; two GP integration tests skip because their
packages are missing. Shell DNS failures also prevent package/data retrieval.
No measured-data diagnostic run or confirmation has completed. The unchanged v1
adapter is used for short, newly generated identity histories; these cannot be
labelled the original macOS trajectories. The CMA sampling-control stage remains
pending. V1 implementation and notebook fingerprints are preserved.


## Accepted focused follow-up — 10 November 2026

The group selected a four-hour compute allowance, CMA-specific boundary attribution,
and report evidence/outline rather than a final report draft. Implementation is
specified in [protocol v3](FOLLOWUP_PROTOCOL_V3.md). The predeclared intervention
replaces continuous acquisition search with a 4,096-point mapped interior/surface
pool. Confirmation uses reserved designs/orientations, separate CMA search
replicates and an explicit 100-call primary / 300-call CMA-extension distinction.
This supersedes unresolved D10/D11/D14 settings; the v1 main grid remains deferred.

The supported local environment passed 27 pre-follow-up tests including GP
integration and completed v2 diagnostics. The expanded implementation passes
36 tests. Earlier missing-package/data notes above are historical environment
records. V2's computed reference is feasible but worse than some pilot points;
its result is retained without calling it an optimum. Current run/coverage status
is in the v3 manifests and analysis audit.

The later contribution record identifies Mihir's EDA and V1–V4 experimental
design and decisions, alongside Raj's contributions. Technical implementation
provenance remains in [validation](IMPLEMENTATION_VALIDATION.md); member roles
are recorded in [Project contributions](STUDENT_ENGAGEMENT.md).

## Accepted V4 implementation — 11 November 2026

The group selected and executed the focused design in
[FOLLOWUP_PROTOCOL_V4.md](FOLLOWUP_PROTOCOL_V4.md): BO mechanism plus CMA,
the existing kappa=10^4 control, a four-hour experiment cap, and effect estimation
without equivalence claims. New rotations are 404/505/606 and designs 20–39;
supporting controls use 20–29. Bound sensitivity changes only pool models' upper
bound from 5 to 20. New data/domains, coupled CMA and 300-call BO are excluded.

The implementation passed 48 tests before launch, including old-adapter parity,
seed/pool coupling, counts/failures/recovery, and paired interaction arithmetic.
The full pipeline was launched detached; actual progress and the timing gate,
not this implementation record, determine whether confirmation ran or completed.
The group retained the declared compute cap and timing gate; any reduction in scope would require a documented design change.

## V4 completion and interpretation — 11 November 2026

All planned runs completed within the four-hour allowance; no reduction, extra
rotation, bound selection or significance-based stopping was used. The final
[results note](V4_RESULTS.md) distinguishes path agreement, absolute performance,
rotation contrasts and numerical diagnostics. In particular, quadratic pool-ARD's
smaller rotation effect accompanies worse absolute loss, and V4's same-box
boundary-budget intervals do not establish V3's Q202 attenuation. Keep these
negative qualifications in the report. The original scientific sources and
V1–V3 inventory remained unchanged during V4. Proposal-stage and launch entries
above are dated history; the latest validated outputs supersede their pending status.

## V5 preparation — 11 November 2026

The group created a separate V5 follow-up. This preparation stage covered files and configuration only; execution and publication remained subsequent steps.

The [v5 protocol](FOLLOWUP_PROTOCOL_V5.md) proposes testing whether CMA's boundary
handler contributes to the BO–CMA gap on the primary ball domain. It preserves
the battery objective and existing adapters, compares resampling with projection,
and gives all methods 100- and 300-call checkpoints. The primary contrast is the
100-call CMA handler difference; BO comparisons and longer-budget findings are
secondary. The design retains null and adverse outcomes rather than selecting
settings for a preferred ranking.

Configuration, execution/analysis entry points and targeted tests are prepared.
None has been executed or validated. Proposed seeds 30–39 avoid the listed
earlier cases. The hypothesis was developed from v3 and will be tested on the
same objective; new runs do not establish generalisation to other batteries. V1–v3
results remain historical evidence, not v5 results or validation.

The group subsequently added laptop execution and JAX support while preserving the shared model and optimiser choices. The optional JAX CPU
backend reproduces the recurrence in float64 using `lax.scan` and `jit`; BO
remains BoTorch/PyTorch and CMA remains pycma. A fixed-point parity check runs
before search when this backend is selected. Compilation/check calls are logged
separately, and backend/environment changes require distinct result collections.

The original fixed cumulative runtime cap is replaced by an optional session
limit, checked between trajectories. A new session can use a different limit or
none, without discarding completed runs. All session durations and failed
attempts remain recorded. Cross-platform locking and optional JAX-derived plots
are prepared for Arch Linux and Windows. No installation, tests or experiments
were executed during these edits.

## Consolidate the completed project — 12 November 2026

The group consolidated V5 with the report, preserving both V5 result collections and their verified export bytes, the V4 executor and the report. The V5 preparation entries above record historical state; the aligned
execution and report are documented in [V5_ALIGNED.md](V5_ALIGNED.md) and the
[result record](../results/v5-aligned/README.md). Implementation, tests and
execution guides now accompany the report in this repository.

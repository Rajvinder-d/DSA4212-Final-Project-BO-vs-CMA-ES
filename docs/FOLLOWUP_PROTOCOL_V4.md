# Focused follow-up V4 — frozen before execution

Accepted design, 11 November 2026. ID: `rotation-followup-v4`; machine settings:
`configs/experiments/followup_v4.json`. The pre-execution specification below
is retained; the completed outcome appears at the end and in [V4 results](V4_RESULTS.md).
V1–V3 source adapters, data, notebooks and results were preserved. Protocol
design and decisions are recorded in [Project contributions](STUDENT_ENGAGEMENT.md).

## Questions and interventions

Test scalar/ARD GP representation crossed with continuous/finite-pool acquisition
search, plus a pool-only upper-length-scale-bound intervention. Strengthen CMA
rotation and same-box boundary comparisons with five independent searches per
design. A disappearing effect does not uniquely identify numerics; persistence
under pool search does not isolate representation from fitting/bounds. CMA
ordinary variability is descriptive, not an equivalence threshold. Covariance
spectral ratios do not demonstrate directional alignment or causal benefit.

The measured battery objective, 3,600 s windows, corrected cross-cell OCV,
capacity/SOC assumptions, log parameter ranges and domains are unchanged from
v1. The sole supporting quadratic is the existing shifted, normalised kappa=10^4
objective. It is a selected stress control, not a battery-matched landscape.

## Confirmation grid and accounting

All blocks use identity/Q404/Q505/Q606 and 20 charged initial observations.
Initial-design seeds 20–39 are primary; 20–29 support quadratics/bounds. Search
replicates 0–4 apply to CMA only. Runs are sequential on one CPU worker, with one
Torch/BLAS/OpenMP thread.

| Block, execution order | Methods | Designs | Calls/run | Runs |
| --- | --- | ---: | ---: | ---: |
| Battery ball BO | scalar/ARD × continuous/pool, random | 20 | 100 | 400 |
| Battery ball CMA | full/diagonal, resampling, five replicates | 20 | 300 | 800 |
| Battery box CMA | full/diagonal × resampling/projection, five replicates | 20 | 300 | 1600 |
| Quadratic ball | four BO variants and random | 10 | 100 | 200 |
| Battery bound sensitivity | scalar-pool/ARD-pool, upper bound 20 | 10 | 100 | 80 |
| Quadratic bound sensitivity | scalar-pool/ARD-pool, upper bound 20 | 10 | 100 | 80 |
| **Total** | | | **796,000 search calls** | **3160** |

The original length-scale upper bound is 5; lower bound .03, initial .7. Bound-20
runs pair with the already planned bound-5 pool trajectories for the same designs.
They do not estimate a full bound-by-acquisition factorial. BO fits use the v1
Matérn-5/2, output scaling, fixed variance 1e-6, output-scale constraints, LogEI,
float64, 60 fitting iterations and 80 acquisition iterations with unchanged
start count/tolerances. Pool size remains 2048 interior plus 2048 surface points,
scored in chunks of 256 with first-index exact ties. Continuous BO directly calls
the unchanged v1 adapter. The pool model/fit parity is tested against v3.

CMA retains population eight, sigma .3, isotropic initial covariance, disabled
active updates/mirrors/additional decoding, and diagonal cc_sep/c1_sep/cmu_sep
matched to full constants. Native update mechanics remain distinct. Checkpoint
100 is after ten generations, 300 after 35. This is not default-CMA benchmarking
or an equal-adaptation claim. Every battery checkpoint gets two FUDS assessments
after the entire trajectory, totalling **10,560** confirmation assessment calls.
No assessment informs selection. Failed objective attempts consume search budget;
rejected proposals consume time only. No partial CMA generation is evaluated.

## Streams and diagnostics

Use NumPy SeedSequence [4107, design, replicate, evaluation_index, stream_kind].
Kinds: 0 initial design (replicate/index zero); 1 CMA/Torch seed; 2 acquisition
starts; 3 candidate pool; 4 random search. Method, kernel, bound, objective and
orientation do not enter these keys. Each step creates its own start/pool stream,
so divergent histories do not shift subsequent corresponding physical pools.
All methods receive matching physical initial designs within a domain. Different
domains still define different initial point sets.

Save actual Q, physical/optimiser coordinates, all charged attempts, proposals,
repairs/rejections, GP vector length scales/status/warnings, pool hashes/indices
and top gaps, timing components, CMA state and assessment records. Hash scientific
modules/configuration/dependencies/data separately from analysis and documents.
Freeze and check scientific hashes during execution. New run keys and retry
suffixes preserve failure records; completed results require hash/accounting
validation. No in-flight optimiser-state resume is implemented.

## Development and timing gate

Development uses seeds 0/1, identity: all four baseline BO methods on both
objectives; both bound-20 pool methods on both objectives; full/diagonal CMA on
ball resampling and box resampling/projection. Total **36 runs, 6,000 search calls,
72 assessments**. None is confirmation evidence.

At histories 20/40/80 from development scalar-continuous runs, replay the same
physical history in identity/Q101. Independently fit both kernels at upper bounds
5 and 20; compare posterior/LogEI at 100 common physical queries (seed 4930),
and acquisition scores/indices on a common 4096-point pool (seed 4941). Add scalar
common-fit controls. Record fits, score gaps, candidate distances, source run IDs
and trace hashes. There are 72 replay records and zero extra simulator calls.
These are fixed-history controls, not recovery of original acquisition RNG states.

The shared experiment allowance is **14,400 seconds**, including development,
replays, confirmation, assessments and failed/retried attempts. Authoring/tests
and analysis/rendering are excluded. Use mean development attempt time by
objective/domain/method/bound/handler plus measured collection overhead, sum over
the pending confirmation grid and multiply by **1.25**. Random-search timing uses
scalar-pool timings conservatively because the fixed pilot has no random runs.
If the forecast exceeds remaining allowance, start no confirmation runs and
report the forecast. A possible separate revision reduces supporting designs
from ten to five; it is not automatically adopted. Do not enlarge or shrink the
grid based on outcomes. Deadlines are checked between calls/solves; an in-flight
solve can overrun. Interrupted ledger entries need investigation before resuming.

## Estimands and uncertainty, predeclared

For every method/orientation/objective, R = log10(L_Q/L_I). Lower final calibration
MSE is better. Compute within-kernel R_pool-R_continuous, then the kernel-by-search
interaction (ARD's difference minus scalar's). Retain signed orientation effects,
absolute paired effects, absolute loss and path distances. Pool-versus-continuous
quality and timing comparisons use matching designs/orientations. For bounds,
report log10(L20/L5), R20-R5, occupancy and fit health; bound changes can themselves
have numerical consequences.

Average CMA search replicates within designs before inference. Show independent
identity-search pairs descriptively; their overlapping pairs are not independent
replications. Report paired log10(L300/L100). On the same physical box,
B_n = R_project,n - R_resample,n; estimate B100, B300 and paired B300-B100 directly.
These budget contrasts are predeclared for V4 (unlike the post-confirmation V3
review). Never infer budget differences by comparing two separate zero-crossings.

Bootstrap whole independent designs, **5000 resamples, seed 4129**, preserving
all associated runs for each contrast. Report pointwise 95% percentile intervals
conditional on the selected rotations, no multiplicity-adjusted family claims,
equivalence claims or significance-based stopping. Supporting quadratic and
bound analyses use ten designs. For cross-block bound comparisons require both
complete planned groups, not just one available matched run. Coverage/failures
remain visible, and incomplete groups cannot silently enter paired inference.

Report 6 mV and 10 mV targets, failures to reach them, conditional calls/time to
hit, measured search/objective/overhead time, held-out errors, model diagnostics
and provenance. Best-known battery reference loss is not certified global regret.
Zero/nonfinite loss ratios must be explicitly diagnosed, never hidden by an
arbitrary logarithm floor. Plotting distance floors are labelled separately.

## Execution and handoff

Commands are run from the root in the verified existing environment:

```
.venv/bin/python scripts/run_v4.py
.venv/bin/python scripts/record_v4_baseline.py
.venv/bin/python scripts/run_v4.py --stage all --run
.venv/bin/python scripts/run_v4.py --stage analysis --run
```

`--stage development` and `--stage confirmation` select charged stages explicitly.
The complete pipeline chains development, gate, confirmation, analysis, notebook
06 in a fresh kernel, and a final output/preservation audit. It uses a process
lock and records PID, stage, progress, elapsed allowance, estimate and exit state.
Long jobs run detached with logs under `artifacts/followup_v4_operations/`.
Competing CPU-heavy work is excluded during scientific timing. PID, log location
and ETA identify the execution state; recovery procedures are in
[Operations](OPERATIONS.md). Partial/budget-limited evidence remains reportable.

## Execution outcome — 11 November 2026

The frozen grid completed without scope reduction: 36 development trajectories,
72 zero-call replays and 3,160 confirmation trajectories. Charged scientific time
was 8,736.05 s, within 14,400 s. The timing gate passed with 25% headroom.
See [V4_RESULTS.md](V4_RESULTS.md) for post-execution interpretation and the
machine-readable audits under `results/followup_v4/`. This appended outcome does
not alter the predeclared design above.

# V4 proposal review — 11 November 2026

Status: **historical proposal review**, written before scope approval. Its planning
recommendations below are preserved. The subsequent accepted scope used a four-hour
study, frozen in [FOLLOWUP_PROTOCOL_V4.md](FOLLOWUP_PROTOCOL_V4.md), and all V4
runs have completed. See [V4_RESULTS.md](V4_RESULTS.md) for evidence and limits.
The review itself is not the frozen executable specification.

## What V3 leaves unresolved

- ARD representation, fitting bounds and numerical search remain entangled.
  The upper-bound count is 4,265/6,000 fits with at least one scale within 1%:
  1,009/2,400 battery fits and 3,256/3,600 quadratic fits. This is not a count
  of independently replicated experiments or proof that the bound caused loss.
- At 100 calls, CMA absolute rotation differences are comparable in scale to
  independent-search identity differences. That is not an equivalence test.
  At 300 calls, diagonal-CMA battery-ball effects are positive under both Q202
  (0.065 [0.012, 0.119]) and Q303 (0.088 [0.030, 0.142]) in log10 MSE units:
  approximately 16% and 23% higher geometric-mean MSE than identity. Both
  intervals exclude zero; describing this simply as a null or "weak effect"
  misses the distinction between effect size and uncertainty. More precision may resolve effects
  of practical interest; it cannot establish an exact zero effect.
- The diagonal-CMA same-box boundary interaction attenuates. The newly added,
  post-confirmation paired budget contrast supports attenuation for Q202 more
  clearly than Q303. Saying it "disappears" at 300 calls is too strong.
- One cell/window, transferred OCV and fixed capacity/SOC/temperature assumptions
  limit external validity. More optimiser repetitions do not address this.
- Quadratic inference has five independent designs per condition. The finite
  pool is one acquisition-search intervention with fixed size and tie handling.
- The common budget is 100 calls. A 300-call BO extension would be an
  **equal-evaluation-budget** comparison, not "equal adaptation": generations
  and BO updates are not interchangeable units of adaptation.

## Recommended smallest useful V4

Prioritise **scalar versus ARD × continuous versus finite-pool acquisition**,
with matched initial designs, domains, settings and indexed physical pools for
the two pool methods. Include a small, separately specified upper-length-scale
bound sensitivity block, using the same settings across kernels where applicable.
Keeping the original bound as a reference is essential. Choose the alternative
bound and fitting allowance before examining its confirmation results.

Use fixed-history replays as well as sequential outcomes: compare fitted
predictions and acquisition scores at corresponding points, selected pool
indices, path distances, final loss, fit status and timing. A coarse pool can
select the same point even when fitted acquisition functions differ. A pooled
signed effect can also be near zero because orientations cancel. Neither
outcome proves a purely numerical explanation.

Twenty new initial designs are a reasonable proposed precision improvement,
not a power guarantee. Keeping 20 initial evaluations holds the intervention
focused; V3 does not establish that 20 is universally adequate or that
initialisation cannot be a bottleneck. Use V3 variability to plan the number of
designs against a scientifically chosen effect size, not against obtaining an
interval that excludes zero. Keep random search as a cheap baseline/control.

Do not run every proposed block as one large factorial grid. If CMA/boundaries
are a second primary question, add a separate 100/300-call CMA replication
block, retaining both variants and both same-box handlers. Five search
replicates per design is a plausible starting proposal; independent design
count and within-design replication answer different precision questions.

## Replace categorical decision rules

| Control | Defensible interpretation and criterion |
| --- | --- |
| Pool-ARD | Estimate the change in rotation effects between continuous and pool search, within each kernel and orientation, and the kernel-by-search interaction. Attenuation implicates acquisition search or its interaction with modelling. Persistence shows that continuous acquisition optimisation is not necessary for the effect; it does not isolate representation from fitting/bounds. |
| Length-scale bounds | Estimate changes under the predeclared alternate bound, including bound occupancy and fit diagnostics. Robustness strengthens a claim about this range of settings; sensitivity limits it. Avoid selecting the better bound after confirmation. |
| CMA replication | Report signed orientation effects and design-bootstrap intervals. Predeclare a practical equivalence margin in log-loss units if the goal is evidence for a negligible effect. An interval inside that margin supports practical equivalence under the design; an interval overlapping both meaningful and negligible effects is inconclusive. Ordinary variability is not a decision threshold. |
| CMA coupling | Treat this first as an implementation/equivariance diagnostic. Validate the marginal sampling law, transformed initial state, boundary handling and random innovations. Full CMA supplies a useful positive control. Coupling can reduce comparison variance, but it does not certify a null or make all diagonal-CMA populations physically identical. |
| Domain change | Compare rotation effects within each preserved physical domain, then report a domain interaction. Changing domains changes volume, initial designs, attainable solutions and boundary exposure. Disappearance of an effect supports domain dependence, not proof that the original ball was a confound. |
| Local-GN quadratic | Test a surrogate geometry at a specified reference point. Agreement supports transfer of a local mechanism to that control; disagreement does not prove that the battery effect is uniquely model-specific. Curvature variation, nonquadratic terms, constraints and symmetry remain alternatives. |
| More designs/rotations | Assess precision and heterogeneity; neither narrower intervals nor unchanged estimates are guaranteed. Report all planned cases rather than stopping when a preferred conclusion appears. |
| Data robustness | Report separate outcomes by window/cell/temperature with explicitly rebuilt inputs. Persistence extends evidence to those cases, not all batteries. |

## CMA coupling is not just sharing Gaussian samples

Equal integer seeds already share some random inputs but do not produce
corresponding physical samples. A valid coupling must preserve each algorithm's
own distribution. Once diagonal-CMA physical covariances differ across
orientations, forcing exactly identical physical candidates would generally
change at least one marginal distribution. Use shared innovations through each
appropriate covariance transformation, with checks for factorisation and
rejection-induced RNG differences. Coupled identity runs with identical state
and innovations should match and therefore are not an ordinary-noise baseline.
Independent identity searches remain useful descriptive controls.

Given this engineering cost, extra independent searches are the simpler first
step unless pathwise equivariance is itself the research question. Neither
option identifies the causal benefit of covariance adaptation; that requires a
separate intervention or a carefully qualified directional diagnostic.

## Domain and curvature cautions

All 90 v3 battery BO recommendations, including the pool variant, lie on the
unit-ball boundary to numerical precision. This motivates a domain robustness
check, but is not proof that it explains the ARD effect. Prefer the already
defined physical box over an enlarged ball as the first robustness condition.
A ball of radius greater than one admits coordinates outside the declared log
parameter box; intersecting it with the box creates another domain. Containing
the computed reference also does not establish an interior global optimum.

The spectral ratio about **555** belongs to the local ambient Gauss–Newton
matrix at the **box reference**. At the feasible ball reference it is about
**113,000**. Neither is "the battery's condition number". They depend on the
point and coordinate parameterisation and are not constrained tangent Hessians.
If constructing a control, use the full specified spectrum and eigenvectors,
declare its centre and scaling, and place its known optimum in the chosen
interior. A generic kappa=555 quadratic matches only the extreme-eigenvalue
ratio, not the spectrum or coupling. It remains a local approximation, with no
claim to reproduce the battery's nonlinear landscape or branch symmetry.

## Rotations, costs and versioning

I/202/303/404 is **four orientations, three nonidentity rotations**, and reuses
two already examined rotations. New designs 20–39 are reasonable; distinguish
replication on 202/303 from confirmation on genuinely unexamined rotation seeds.
Choose new rotations before inspection and retain orientation-specific results.
If inference over rotations is intended, design for that additional source of
variation; bootstrapping designs alone remains conditional on the chosen Qs.

Do not extrapolate pool-ARD from scalar-pool's 2.36 s/run or 300-call continuous
BO linearly from 100 calls. ARD fitting costs differ and GP work grows with
history length. Time a small declared development pilot at the largest proposed
budget/domain before freezing the run count and allowance. The proposed 4–6 h
is a planning envelope, not a demonstrated cost. Count failed attempts, tuning
and separate assessments explicitly. More domain/replicate factors multiply
the grid quickly.

After scope selection, create `FOLLOWUP_PROTOCOL_V4.md`, a validated
`followup_v4.json` and a new `rotation-followup-v4` artifact root. Keep V1–V3
raw artifacts, configs and scientific adapters frozen. In particular, directly
adding V4 branches to the frozen V3 runner would alter its hashed provenance;
prefer a small V4 module or explicitly versioned shared implementation. Reuse
the tested sampling, accounting and analysis helpers where compatible, while
preserving a reproducible V3 checkout. No V4 configuration is created by this
review because the proposal's factors, thresholds and compute plan are not yet
frozen.

## Methodological sources

These recommendations apply the covariance-metric distinction in
[GPML, chapter 4](https://gaussianprocess.org/gpml/chapters/RW4.pdf), the sampling
and invariance discussion in
[Hansen's CMA tutorial](https://arxiv.org/html/1604.00772v2), and the importance
of acquisition numerics in
[Ament et al., LogEI](https://proceedings.neurips.cc/paper_files/paper/2023/hash/419f72cbd568ad62183f8132a3605a2a-Abstract-Conference.html).
The proposed controls and interpretation rules are project design recommendations,
not experiments established by those sources.

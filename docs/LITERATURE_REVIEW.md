# Rotation, conditioning and battery calibration: literature review

Reviewed 9 November 2026. This is a targeted methodological review, not an exhaustive novelty search or a claim that these papers' experiments have been reproduced. “Read” below means the stated sections were inspected, not that every proof or appendix was independently checked. The implementation decisions belong to this project. The executable specification is [protocol v1](EXPERIMENTAL_PROTOCOL_V1.md).

## 1. Question and scope

The study asks how coordinate-specific GP length scales and CMA covariance adaptation affect optimisation under rotations, and whether boundary handling changes the observed rotation effect. A five-parameter battery calibration loss provides an interpretable measured-data application; shifted quadratics provide controlled curvature spectra. These serve different purposes: the battery has model discrepancy, symmetry and unknown optima, whereas the quadratics isolate a known mathematical mechanism.

A rotation is a reparameterisation only when the physical feasible set, objective, initial observations and other prior information are preserved. Rotating a function inside a newly imposed axis-aligned cube changes the problem. Moreover, an orthogonal transformation preserves Hessian eigenvalues: poor performance after rotation is not evidence that the objective became more ill-conditioned. This distinction is central to the project's design.

## 2. GP geometry: isotropy, ARD and numerical optimisation

[Rasmussen and Williams (2006), *Gaussian Processes for Machine Learning*, Chapter 4, particularly §4.2](https://gaussianprocess.org/gpml/chapters/RW4.pdf) develops stationary covariance functions and distance metrics. For the project's Matérn kernel the distance is

\[
r^2=(z-z')^T\operatorname{diag}(\ell_1^{-2},\ldots,\ell_d^{-2})(z-z').
\]

Equal length scales make this distance invariant under orthogonal transformations. Unequal ARD scales describe an axis-aligned metric; a generic rotation produces a dense metric that this kernel family cannot represent. ARD does not automatically discover an arbitrary linear low-dimensional subspace. **Decision:** compare scalar and diagonal metrics with the same Matérn smoothness, likelihood, fitting budget and acquisition optimiser. No axiswise input normalisation after rotation, which would introduce an additional metric change. **Limitation:** a representational property does not guarantee better finite-budget search; the isotropic model may underfit strong anisotropy.

[Frazier (2018), *A Tutorial on Bayesian Optimization*, GP modelling, acquisition and noisy-evaluation sections](https://arxiv.org/html/1807.02811v1) explains the surrogate–acquisition loop and the distinction between choosing an experiment and recommending a final point. **Decision:** sequential exact-GP BO, observed deterministic calibration minimum as the recommendation, and separately assessed predictive error. The fixed measured trace defines a deterministic function even though the original measurements contain error. **Limitation:** GP uncertainty here is a surrogate's modelling uncertainty; it is not a calibrated posterior over battery physics or freshly sampled voltage noise. Acquisition maximisation and marginal-likelihood fitting are themselves numerical optimisation problems and can break idealised equivariance.

[Ament et al. (2023), *Unexpected Improvements to Expected Improvement for Bayesian Optimization*, numerical analysis and LogEI construction](https://proceedings.neurips.cc/paper_files/paper/2023/file/419f72cbd568ad62183f8132a3605a2a-Paper-Conference.pdf) identifies underflow and poor numerical gradients in conventional EI and proposes stable logarithmic formulations. **Decision:** analytic LogEI for a single candidate, double precision, explicitly feasible multistart acquisition optimisation. **Limitation:** LogEI improves numerical behaviour; it neither removes local acquisition optima nor supplies a rotation-invariance theorem for the project's constrained numerical pipeline. The implementation retains fitting statuses and warnings rather than treating every iteration cap as successful convergence.

[Hvarfner et al. (2024), *Vanilla Bayesian Optimization Performs Great in High Dimensions*, model/prior experiments](https://arxiv.org/html/2402.02229v4) and [Papenmeier et al. (2025), *Understanding High-Dimensional Bayesian Optimization*, optimisation and initialisation discussion](https://arxiv.org/html/2502.09198v1) show why conclusions about GP-BO depend on surrogate settings and fitting behaviour. **Decision:** explicitly freeze and record length-scale bounds, initial hyperparameters, target scaling, fitting iterations and acquisition starts instead of relying on evolving library defaults. **Limitation:** their dimensional regimes and benchmark suites differ from this five-dimensional battery problem; their reported rankings are not predictions for this study. No claim is made that the project's fixed settings are optimal for either kernel.

[Eriksson et al. (2019), *Scalable Global Optimization via Local Bayesian Optimization*, trust-region method and experiments](https://arxiv.org/html/1910.01739v4) motivates fitting local surrogate models when global modelling becomes difficult. **Decision:** keep TuRBO outside v1 so the kernel comparison has one global BO pipeline. **Limitation/follow-up:** a global-BO failure here does not establish a failure of BO generally. Axis-aligned trust regions would introduce another geometry intervention; adding TuRBO requires its own domain treatment and protocol revision.

## 3. CMA-ES: covariance, invariance and implementation details

[Hansen (2016, revised 2023), *The CMA Evolution Strategy: A Tutorial*, covariance adaptation, step-size control and invariance discussion](https://arxiv.org/html/1604.00772v2) explains rank-based updates and adaptation of a Gaussian search distribution. A full covariance can represent a rotated ellipsoid; a diagonal covariance generally cannot. Orthogonal equivariance requires correspondingly transformed initial state and sampling, and compatible constraint treatment. **Decision:** compare full and diagonal covariance, population eight, isotropic initial covariance, common step size, common path/covariance learning rates, no restarts and no value-based stopping. **Limitation:** identical integer random seeds need not produce rotated copies of Gaussian populations. The controls test distribution/covariance transformations, not exact pathwise equality across rotations. Boundary rejection and projection affect which distribution is actually evaluated.

[Ros and Hansen (2008), *A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity*, author-maintained summary](https://cma-es.github.io/) motivates diagonal/separable CMA for reduced computational cost and faster adaptation on suitable problems. Reading status: author summary and bibliographic record only; the full paper was not accessible during this review. **Decision:** include diagonal CMA as a coordinate-dependent control. **Limitation:** v1 equalises learning-rate constants to focus the comparison; it is therefore not a benchmark of optimally tuned sep-CMA against optimally tuned full CMA.

[pycma official options documentation](https://cma-es.github.io/apidocs-pycma/cma.options_parameters.html) was checked alongside installed **4.5.0** source in `sampler.py`, `options_parameters.py` and `evolution_strategy.py`. Full CMA adds a small initial diagonal perturbation; diagonal CMA normally uses different adaptation constants. **Decision:** cancel the initial perturbation through `CMA_teststds`, equalise `cc_sep`, `c1_sep`, `cmu_sep` to the full constants, and disable active updates, mirrors and additional diagonal decoding for both. **Limitation:** diagonal and dense covariance updates remain their native pycma implementations, including diagonal decoding's exponential update. This is a controlled practical representation comparison, not an algebraically identical update with off-diagonals deleted. Version-specific adapter tests guard these choices.

For constraints the V1 design compares feasible resampling with Euclidean projection onto the same transformed physical box. Both operators commute with an orthogonal coordinate change when correctly implemented. Therefore, **the hypothesis is an interaction in practical optimisation**, not a theorem that projection necessarily destroys rotation equivariance. A larger observed difference may reflect changed boundary occupancy, finite-run stochastic variation, or numerical implementation. Proposal, rejection, repair and evaluated-point records are needed to distinguish them.

## 4. Benchmarking and what paired runs establish

[Hansen et al. (2016, revised 2020), *COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting*, experimental design and performance measurement](https://arxiv.org/pdf/1603.08785) motivates controlled instances, explicit evaluation accounting and interpretable performance targets. **Decision:** common physical initial designs, fixed call budgets, saved trajectories, paired seeds and rotation interventions, plus a random-search control. **Limitation:** the project's fixed-budget application study is not a COCO/BBOB benchmark. The study does not import target-hitting statistics without justified targets or label a local reference fit as a known global optimum.

In V1, the inferential unit was an optimiser seed, conditional on selected profiles and rotations. Within each seed, compare log final loss after rotation with identity and average over the selected rotations before bootstrap resampling seeds. This is a design choice, not a claim that timestamps, multiple rotations or held-out profiles supply independent battery replications. The three-seed pilot is descriptive. Later protocols use their declared
design-level estimands and replication, as documented in [Research progression](RESEARCH_PROGRESSION.md). For CMA, common integer seeds do not give pathwise coupled physical draws: absolute paired differences can remain nonzero under invariance in distribution. They are descriptive sensitivity measures, not by themselves evidence of systematic orientation bias. Failures and warnings are outcomes to investigate, not runs to silently discard. Objective-call counts and measured wall time answer different questions; a cheap circuit simulator can make GP fitting dominate total time.

## 5. Battery models, OCV and identifiability

[Tian, Wang, Chen and Fang (2017), *On Parameter Identification of an Equivalent Circuit Model for Lithium-Ion Batteries*, §§II–III](https://www.merl.com/publications/docs/TR2017-123.pdf) derives a Thevenin model, sensitivity-based local identifiability and constrained nonlinear least squares. Its joint OCV/parameter treatment and constant-current experiment differ from the project's fixed-OCV dynamic-profile fit. **Decision:** use time constants rather than capacitances as search variables, a residual-based bounded least-squares reference, and local sensitivity diagnostics. **Limitation:** their identifiability argument does not certify identifiability for the project's selected trace, bounds or cross-cell OCV transfer. Finite-difference simulator calls must be counted explicitly for the reference optimiser.

[Grandjean, McGordon and Jennings (2017), *Structural Identifiability of Equivalent Circuit Models for Li-Ion Batteries*, §§1–2 and discussion](https://www.mdpi.com/1996-1073/10/1/90) distinguishes structural from practical identifiability and illustrates interchangeable RC branches. **Decision:** retain both branch orderings during optimisation, sort only for presentation, compare one- and two-RC fits, and inspect the residual Jacobian and time-constant separation. **Limitation:** low voltage error does not establish unique physical parameters. Nearly equal time constants or a weak branch can yield flat directions even when the circuit form is locally identifiable elsewhere. Local Gauss–Newton eigenvalues are diagnostic, not exact global curvature.

[Thevenin model documentation, governing equations](https://thevenin.readthedocs.io/stable/user_guide/model_description.html) provides a useful independent equation/sign check. **Decision:** discharge-positive current, Coulomb-counted SOC, ohmic voltage drop and first-order polarisation branches. The implementation is isothermal, with constant resistances/time constants, fixed capacity and no explicit hysteresis. **Limitation:** these omissions can explain systematic residuals and held-out errors; better optimisation cannot remove all model discrepancy.

The [CALCE primary data catalogue](https://calce.umd.edu/data) identifies dynamic DST/FUDS tests, initial SOC/temperature conditions and separate low-current OCV characterisation. Its linked files were downloaded and their workbook contents inspected. **Decision:** SP20-2 DST/FUDS at 25°C and SP20-3 low-current OCV, with original-byte hashes, all channel sheets, native timestamps and documented stage selection. **Limitation:** the OCV cell differs from the dynamic cell. The V1 design explicitly accepted this transfer as a tested approximation, not as proof of cell equivalence. A constant offset is estimated solely from the DST preparatory rest and frozen for assessment; an uncorrected sensitivity block tests dependence on that adjustment.

[Zheng et al. (2016), *Influence of different open circuit voltage tests on state of charge online estimation for lithium-ion batteries*, Applied Energy 183, 513–525](https://web.calce.umd.edu/articles/abstracts/2016/16_Influence_of_different_open_circuit_voltage_tests_on_state_of_charge_online_estimation_for_lithium-ion_batteries.html) compares low-current and incremental OCV tests. **Reading status:** the authors’ CALCE abstract and catalogue description, not full text. The abstract reports more robust SOC tracking from incremental-OCV characterisation in their comparison. **Decision:** the project's available low-current curve is a declared approximation, not the literature-established best OCV procedure. **Limitation:** their SOC-estimator comparison does not quantify the project's calibration error or validate transfer between SP20-3 and SP20-2. This strengthens the case for the uncorrected sensitivity check and eventual same-cell/incremental-OCV follow-up.

[Xing et al. (2014), *State of Charge Estimation of Lithium-Ion Batteries Using the Open-Circuit Voltage at Various Ambient Temperatures*, Applied Energy 113, 106–115](https://web.calce.umd.edu/articles/abstracts/2014/14_State%20of%20charge%20estimation%20of%20lithium-ion%20batteries%20using%20the%20open-circuit%20voltage%20at%20various%20ambient%20temperatures.html) uses temperature-dependent OCV and a UKF; its author-hosted abstract describes DST identification and FUDS validation. **Reading status:** abstract/catalogue only. **Decision:** use separate profile roles and keep temperature fixed in v1. **Limitation:** its adaptive SOC estimator and temperature model differ from the project's fixed-SOC, fixed-parameter voltage fit; its results cannot justify the project's cross-cell correction. Only the abstract/catalogue evidence is used here.

## 6. Related battery BO and limits of novelty

[Pi et al. (2024), *Parameter Identification for Electrochemical Models of Lithium-Ion Batteries Using Bayesian Optimization*, model and numerical/experimental sections](https://arxiv.org/html/2405.10750v1) studies an electrochemical equivalent-circuit model with electrochemical parameters, with gradient-descent and particle-swarm comparators. Its three identified parameters, model, evaluation setup and measured experiments differ from the project's five-parameter two-RC CALCE study. **Decision:** cite this as evidence that battery calibration with BO is established, not as a matched baseline or a novelty claim. **Limitation:** its comparisons do not establish BO versus CMA-ES performance or the effects of domain-preserving rotations. The project neither reproduces its results nor adopts its reported ranking as a conclusion.

[Tu, Lin, Wang and Fang (2024), *System Identification for Lithium-Ion Batteries with Nonlinear Coupled Electro-Thermal Dynamics via Bayesian Optimization*, §§II–IV](https://arxiv.org/html/2405.20219v1) uses a ten-parameter nonlinear double-capacitor/thermal model, likelihood optimisation and progressively reduced ellipsoidal search regions. Its demonstration generates synthetic voltage/temperature data under US06, UDDS and LA92 profiles, and compares search-space reduction with standard BO. **Decision:** distinguish this work from measured CALCE calibration and keep adaptive domain shrinkage outside v1. **Limitation:** it does not compare CMA or domain-preserving rotation effects; synthetic parameter recovery cannot establish predictive validity on the project's measured cell. It nevertheless makes clear that battery BO with non-axis-aligned search geometry predates this project. Reading status: model, method and simulation sections inspected.

The initial proposed contribution was a controlled explanation connecting
surrogate/covariance representation to orientation effects, distinguishing
conditioning from rotation, and examining boundary treatment while preserving
the physical problem. At the design stage, weak rotation effects or dominant
numerical/boundary effects were possible outcomes. The subsequent
[research progression](RESEARCH_PROGRESSION.md) records how those possibilities
were investigated.

## 7. Initial V1 evidence-to-experiment map

| Question | Published basis | Project intervention and falsifiable expectation | Important alternative explanation |
| --- | --- | --- | --- |
| Does isotropic BO rotate more consistently than ARD? | Kernel metric geometry; GPML | Same points/domain, scalar versus diagonal metric; smaller average absolute paired changes is a hypothesis | Acquisition numerics, fitting caps, finite data, isotropic underfitting |
| Does full CMA cope with oriented curvature? | CMA tutorial; diagonal representation | Fixed common settings and learning rates, full versus diagonal covariance | Only ten pilot generations; stochastic variation; native update differences |
| Does boundary handling alter the rotation contrast? | Constraint treatment plus the geometric derivation | On the same box compare rotation effects under rejection and projection | Both handlers are geometrically equivariant; sample variation can dominate |
| Does conditioning amplify the effect? | Quadratic Hessian algebra | Independently set κ=1,100,10000; preserve eigenvalues across Q | Normalisation, finite budget and saturation at initial best |
| Does good calibration transfer? | ECM/OCV and identifiability literature | Evaluate frozen recommendations on FUDS-80 and FUDS-50 | Cross-cell OCV, fixed SOC/capacity, model discrepancy |

## 8. Reading and implementation record

All links above were checked during this implementation. Full text/selected sections: GPML ch.4; Frazier modelling/acquisition; Hansen adaptation/invariance; Ament numerical LogEI; Eriksson method; COCO experimental design; Hvarfner settings; Papenmeier fitting/initialisation; Tian §§II–III; Grandjean §§1–2/discussion; Pi methods/experiments. Official documentation/source: CALCE data, Thevenin equations, pycma options and installed source. Partial only: Ros/Hansen author summary, Zheng/Xing author-hosted abstracts and catalogue references. No full-paper reading is implied for partial entries.

[BoTorch acquisition optimisation documentation](https://botorch.readthedocs.io/en/stable/optim.html) informs explicit feasible initial conditions and linear/nonlinear constraints. [SciPy least_squares documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) informs bounded residual fitting and the need to count numerical derivative calls outside reported optimiser iteration counts. Library versions and successful environment checks are recorded in `requirements-lock.txt` and run provenance.

The reading record describes source coverage. Experimental outcomes and subsequent
changes are documented in [Research progression](RESEARCH_PROGRESSION.md);
member contributions are recorded in [Project contributions](STUDENT_ENGAGEMENT.md).

## 9. Geometric derivations

The following derivations support the protocol comparisons and were transferred
from the former engagement workbook. GPML and the CMA tutorial cited above
provide the mathematical background.

Let u=Qz with QᵀQ=I. For a scalar kernel length scale ℓ,
`||Q(z-z')||²/ℓ² = ||z-z'||²/ℓ²`, so corresponding observations and queries have
the same kernel matrices. With the same targets, mean, noise regularisation and
hyperparameters, the GP posterior and analytic LogEI therefore agree in exact
arithmetic. Numerical fitting and acquisition maximisation remain separate
algorithms; their finite results need not follow identical paths.

For ARD, write the metric as D=diag(ℓ₁⁻²,…,ℓ_d⁻²). Reparameterisation gives
`(z-z')ᵀQᵀDQ(z-z')`. In general QᵀDQ is dense, outside the diagonal ARD family.
This describes representational capacity, not a guarantee that isotropic BO
achieves smaller loss or greater numerical stability.

If z has covariance C_z, u has covariance `C_u=Q C_z Qᵀ`. Full covariance can
represent this transformation; a diagonal covariance generally cannot. Drawing
the same Gaussian vector in each coordinate system is not the same as rotating
one physical draw, explaining why equal integer seeds do not establish paired
physical CMA trajectories.

For a quadratic/local Hessian, `H_z=QᵀH_uQ`, which preserves eigenvalues and
their ratio. Rotation does not itself change conditioning. The box transforms
as `-1 ≤ Qz ≤ 1`; clipping z componentwise instead would change the feasible
problem. Euclidean projection is `Qᵀ clip(Qz,-1,1)` and commutes with the
coordinate transformation. The unit ball is unchanged by Q.

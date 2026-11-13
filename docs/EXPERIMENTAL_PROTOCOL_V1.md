# Experimental protocol v1: rotation, conditioning and battery calibration

Status: completed 84-run CPU pilot; the separate 2,360-run V1 main recipe remains deferred and unexecuted. Protocol ID `rotation-battery-v1`. Date: 9 November 2026. The machine-readable scientific settings are [`rotation_battery_v1.json`](../configs/protocols/rotation_battery_v1.json); recipes are in [`configs/experiments/`](../configs/experiments/). The [literature review](LITERATURE_REVIEW.md) gives evidence and limitations. This document supersedes the unresolved choices in the earlier planning protocol, which is retained for history.

## 1. Question, hypotheses and estimands

How do coordinate-specific GP length scales and CMA covariance adaptation affect optimisation under rotations, and how much does boundary treatment change the observed sensitivity?

H1: isotropic GP-BO and full CMA may have smaller rotation sensitivity than ARD GP-BO and diagonal CMA when physical geometry and initial information are preserved. This is a finite-budget hypothesis, not an assertion of identical numerical trajectories. Natural battery coordinates are not assumed favourable to ARD or diagonal CMA.

H2: on a fixed physical box, changing CMA boundary treatment may change the magnitude or sign of its rotation effect. Resampling and correct Euclidean projection both respect orthogonal reparameterisation; no structural invariance violation is presumed.

H3: coordinate dependence may become more consequential as quadratic conditioning increases. Orthogonal rotation itself preserves conditioning. On the battery, measure local curvature rather than assigning a known condition number.

Primary outcome: lowest deterministic DST calibration MSE after the allotted simulator attempts (lower is better). Primary paired effect: log10(L_Q/L_I) for each method/seed. Also inspect absolute paired effects and the within-seed contrasts |effect_ARD|−|effect_iso| and |effect_diagonal|−|effect_full|, averaging selected orientations within seed before bootstrap. Predictive outcomes: voltage RMSE on FUDS-80 and FUDS-50 at the frozen recommendation. Runtime and objective/decision costs are separate secondary outcomes.

## 2. Exact data sources and preparation

Original files from the [CALCE catalogue](https://calce.umd.edu/data), base URL `https://web.calce.umd.edu/batteries/data/`:

| Archive | Workbook used | Role |
| --- | --- | --- |
| `SP2_25C_DST.zip` | `11_05_2015_SP20-2_DST_80SOC.xls` | SP20-2, 25°C, nominal initial SOC .8, calibration |
| `SP2_25C_FUDS.zip` | `11_06_2015_SP20-2_FUDS_80SOC.xls` | Same cell/temperature, initial SOC .8, assessment |
| `SP2_25C_FUDS.zip` | `11_09_2015_SP20-2_FUDS_50SOC.xls` | Same cell/temperature, initial SOC .5, transfer assessment |
| `SP3_25C_LC_OCV_11_16_2015.zip` | `11_16_2015_low current OCV test_SP20-3.xls` | Separate cell, low-current OCV characterisation |

SHA256 values are fixed in [`data_sources.json`](../configs/protocols/data_sources.json). Download only missing files; preserve original ZIP bytes. If download fails, manually place these same ZIPs in `data/raw/` and rerun preparation. A checksum mismatch stops preparation. No simulated-data fallback is permitted.

The `.xls` extension is misleading: these files contain OOXML ZIP workbooks. Read their bytes with openpyxl, all `Channel*` sheets, selecting explicit named columns and units: seconds, amperes, volts, ampere-hours. Required values must be finite. Reconcile overlapping record IDs only when all selected fields agree; conflicts and clock resets fail. After record reconciliation, keep the last record at an identical timestamp and count this in the audit. Validate increasing sample times. Preserve record/step/cycle columns and unmodified raw archives.

Steps 1–6 contain preparation; dynamic current begins at step 7. Require the preceding row to be step 6 with |current|≤.01 A. Set t=0 at that final preparatory-rest sample, retain it and every subsequent sample with t≤3600 s, including interleaved dynamic rests. This convention adds the final rest-to-profile interval rather than interpolating a synthetic sample at a rounded time. Require coverage to at least 3585 s. The first 3600 s is used for each selected profile, without resampling. Negate CALCE charge-positive current to discharge-positive current internally. SOC uses nominal capacity 2 Ah and the documented .8/.5 initial value, integrating left-endpoint current. Invalid SOC stops execution; never clip it.

For SP20-3 OCV use low-current discharge step 6 and charge step 8, requiring magnitudes .08–.12 A and a measured capacity span between 1.5 and 2.5 Ah. Normalise each branch's cumulative capacity by its own measured span: discharge SOC=1−fraction; charge SOC=fraction. Average duplicate SOC voltages, interpolate each branch linearly onto 1001 equally spaced SOC values in [0,1], then average the two voltages. This approximates OCV; it does not remove all hysteresis or ohmic bias. Battery model interpolation is linear inside that grid only.

**Cross-cell transfer is explicitly user-approved.** Estimate δ=mean voltage during the final 60 s of DST step-6 rest minus U_OCV(.8). Freeze δ for every optimiser and held-out trace. Never use FUDS voltage to estimate it. The uncorrected sensitivity block sets δ=0 everywhere. The measured v1 offset is approximately +12.155 mV; the manifest, not this rounded figure, is authoritative. Repeated optimiser runs use fixed measured traces and quantify search variability, not new measurement noise.

Preparation produces four CSVs and `data/processed/manifest.json` with sources, hashes, sheet/duplicate audits, capacity spans, offset and conventions.

## 3. Model, parameterisation and objective

Use a constant-parameter, isothermal Thevenin two-RC circuit. With discharge-positive I and capacity C_n=2 Ah:

\[
s_k=s_{k-1}-I_{k-1}\Delta t_k/(3600C_n),\quad
v_{j,k}=e^{-\Delta t_k/\tau_j}v_{j,k-1}+R_j(1-e^{-\Delta t_k/\tau_j})I_{k-1},
\]
\[
\widehat V_k=U(s_k)+\delta-R_0 I_k-v_{1,k}-v_{2,k},\qquad v_{j,0}=0.
\]

I[k−1] is held on [t[k−1],t[k]); the instantaneous ohmic drop uses I[k]. Exponential updates are exact for that convention on native irregular times. Zero initial polarisation is justified by the preparatory rest as an assumption; very slow residual relaxation may remain. The model does not fit capacity, initial SOC, temperature dependence, OCV, initial polarisation or hysteresis.

| Parameter order | Lower | Upper | Unit |
| --- | ---: | ---: | --- |
| R0 | .001 | .2 | Ω |
| R1 | .0001 | .2 | Ω |
| τ1 | 1 | 2000 | s |
| R2 | .0001 | .2 | Ω |
| τ2 | 1 | 2000 | s |

These are declared experimental ranges, not validated physical estimates. Let a,b be componentwise logarithms of the bounds. Map u to θ=exp((a+b)/2+(b−a)u/2). Both RC branch orderings remain feasible and yield identical voltage when exchanged. Any table ordering is cosmetic; no ordering inequality enters optimisation.

Loss L(u)=N^−1 Σ( V̂_k−V_k )², equally weighted by recorded sample, in V². Native times are irregular, so this is sample MSE, not a time-integrated loss. Assessments report sqrt of their corresponding MSE in volts (plots may use mV). The recommendation is the lowest observed calibration loss, including initial points. A battery reference-fit gap is a gap to a computed feasible reference, never known global regret.

## 4. Geometry and controlled objectives

Rotation matrices: identity or QR of a 5×5 standard normal matrix from NumPy default_rng(rotation_seed), sign-correct QR diagonal, then flip the first column if needed for determinant +1. Save Q. Use u=Qz (row-array implementation u=z Qᵀ); positive parameters always follow from u, never directly from z.

Ball domain: ||z||₂≤1, equivalently ||u||₂≤1. This is a strict restriction of the physical log box, invariant across Q. Box domain: −1≤Qz≤1 componentwise, exactly the original physical log box. Do not replace this with −1≤z≤1. Do not compare unrestricted-box reference scores as if they were feasible ball scores.

Uniform initial/random points are sampled in base coordinates and mapped into each orientation. Ball sampling uses a normalised Gaussian direction and radius U^(1/5); box sampling is uniform on [-1,1]^5. The same seed gives the same base design across methods/rotations within a domain. Different domains do not have matched initial physical points.

For κ∈{1,100,10000}, define w_i=κ^((i−1)/4), a_i=.25/√5, f(u)=Σw_i(u_i−a_i)²/Σw_i. Its known global minimum is zero at an interior point of the ball; Hessian 2 diag(w)/Σw has condition number κ. Rotating preserves that spectrum. Normalisation prevents the largest raw loss scale from growing arbitrarily with κ; targets remain condition-specific. These controls are deterministic and distinct from the measured-data application.

## 5. Algorithms and fixed computational effort

All methods receive and pay for 20 physical initial evaluations. BO trains on all 20; random search continues independent uniform draws; CMA uses the best point as its mean, without inferring covariance from the initial cloud. This interface difference is explicit and common to both CMA variants. There is no performance-based tuning between reported methods.

**GP-BO:** BoTorch/GPyTorch exact SingleTaskGP with constant mean, output scale and Matérn ν=5/2 covariance. Scalar length scale for `bo_iso`, five scales for `bo_ard`. At each step standardise negative observed loss with current sample mean/std (denominator floor 1e−12); maximise analytic LogEI against the maximum standardised observation. Fixed variance 1e−6 in standardised units is numerical regularisation, not inferred experimental noise. No input transform. Length scales constrained to [.03,5], initial .7; output scale [.01,100], initial 1. Refit from these common initial hyperparameters each step using marginal-likelihood optimisation, at most 60 iterations, ftol=1e−8, float64 and one Torch thread. These hyperparameters have no added hand-specified priors.

Generate 256 feasible acquisition-start candidates with the dedicated search RNG and keep the top eight. Optimise from these starts, sequentially (`batch_limit=1`), at most 80 iterations, ftol=1e−9, no automatic warning retry. Ball constraint is 1−Σz²≥0; box uses all ten linear inequalities ±Q_i z≥−1. Enclosing coordinate bounds are [-1,1] for ball, ±Σ_i|Q_ij| for box. Infeasibility over tolerance 1e−6 terminates the run; smaller floating-point boundary error is projected away. Record fitting status, warnings, length scales, output scale, LogEI and fit/acquisition time. Iteration-limit warnings remain visible even if a feasible candidate is returned.

**CMA:** pycma 4.5.0, population eight, initial σ=.3, isotropic covariance, full or diagonal representation. Disable active covariance updates, mirrors and additional diagonal decoding. Cancel the library's tiny full-sampler axis perturbation so the initial sampling covariance is exactly .09I within floating-point tolerance. Equalise covariance-path/rank-one/rank-μ constants in diagonal CMA to the corresponding full-CMA constants; only representation-specific native update mechanics remain. The pinned-library adapter and tests enforce this. No restarts, no optimiser `stop()` call and no value-based early termination. Every generation has eight accepted/evaluated candidates (10 generations in pilot; 35 in main).

CMA boundary resampling: ask again until feasible, at most 10000 proposals per evaluated point; on cap failure stop and preserve the run. Projection: u=Qz, clip u to [-1,1], map back with Qᵀ; for the ball use radial projection. Pass evaluated points to `tell`. Resampling is used for the ball comparisons. Box ablation runs both resampling and projection for full and diagonal CMA. Rejections cost time, not simulator calls; repeated projected points are evaluated and charged. Save both proposed and evaluated coordinates and repair indicators.

**Random:** uniform feasible independent draws, same initial data and total budget. Because sampling is in base coordinates, its paired trajectories provide a strong implementation control across rotations.

**Structure-exploiting reference:** bounded nonlinear least squares on residuals in full-box log coordinates, with one or two RC branches. Three starts (zero and two draws uniform in [-.8,.8], RNG710), 300 actual simulator calls allocated in three cumulative blocks of 100; unused allocation may carry into the next block. SciPy `least_squares`, three-point numerical Jacobian, ftol/xtol/gtol=1e−10. Count every residual callback, including derivative perturbations, in the budget; save best evaluated point. Corrected one-RC, corrected two-RC and uncorrected two-RC references are diagnostics, not members of the five-way black-box ranking. Two assessment calls per reference are additional. The local 2JᵀJ/N diagnostic uses ten extra calls at step1e−4, truncated one-sided at bounds.

## 6. Recipes, seeds and costs

| Recipe/block | Orientations | Optimiser seeds | Methods/handlers | Runs |
| --- | --- | --- | --- | ---: |
| Pilot battery ball | I, Q101 | 0,1,2 | five methods, CMA resampling | 30 |
| Pilot battery box | I, Q101 | 0 | five methods, CMA both handlers | 14 |
| Pilot quadratics | I, Q101; three κ | 0 | five methods, CMA resampling | 30 |
| Pilot uncorrected OCV | I, Q101 | 0 | five methods, ball | 10 |
| **Pilot total** | | | 100 attempts/run, including 20 initial | **84** |
| Main battery ball | I,Q101,Q202,Q303 | 0–19 | five methods | 400 |
| Main battery box | same four | 0–19 | five methods, CMA both handlers | 560 |
| Main quadratics | same four; three κ | 0–19 | five methods | 1200 |
| Main uncorrected OCV | I,Q101 | 0–19 | five methods | 200 |
| **Main total (unrun)** | | | 300 attempts/run, including 20 initial | **2360** |

Pilot: 8400 search attempts plus 108 final battery assessment calls; reference fits/curvature are separate. Main: 708000 search attempts plus 2320 battery assessment calls. Smoke: 28 attempts, 20 initial, Q101, seed9, ball/box and both CMA handlers, 14 runs. Smoke is validation, not added replication.

Initial RNG = default_rng(seed). Search RNG = default_rng(SeedSequence([seed,41])). Torch and CMA seeds = seed+10000. Rotation RNGs are separate. CPU, one worker, one Torch thread; CLI/notebook runner also limits BLAS/OpenMP threads. Package versions, platform and machine are saved. No artificial objective delays. Search timing includes objective calls, proposal rejection, fitting, acquisition, update and trace logging; setup/data loading and final assessment are outside it. Assessment time is separate. Runtime extrapolation from pilot to main is only a planning estimate because GP costs grow with training-set size.

## 7. Failure policy, provenance and restart

Every attempted simulator call, including an exception/nonfinite result, consumes budget. Record the failure then terminate that run; do not replace it with an arbitrary penalty or successful pseudo-observation. Invalid coordinates detected before simulation consume no call and cause an explicit failure. Ordinary warnings are recorded and investigated. Remaining jobs continue. Failed/incomplete directories persist; retries get a new suffix. Configuration/data-construction failures can prevent a collection from starting and must be resolved explicitly.

Save resolved config, code fingerprint, Git revision/dirty flag, package versions, hardware, source-data hashes, Q and initial points. Each run includes manifest/config/initial, evaluation CSV, proposal/diagnostic JSONL, summary and (if successful) recommendation; errors include traceback. Verify file hashes, budget counts and trace minimum before reusing a completed run. Reinvoking the collection skips only verified matching completions. There is no mid-run optimiser checkpoint. A code/config/data change generates a different run key. Outputs from earlier implementation checks remain archived and are not silently pooled into the current index.

## 8. Analysis, interpretation and follow-up

Show individual convergence curves and medians, final paired log ratios, common-seed method comparisons, final held-out errors, wall-time components, proposal rejection/repair rates and internal GP/CMA diagnostics. For each method average selected rotation effects within seed, then bootstrap seeds (2000 resamples, RNG129) for percentile 95% intervals. Three-seed pilot intervals are labelled descriptive, and one-seed controls have no inferential interval. CMA uses common integer seeds but not pathwise coupled physical Gaussian samples. Consequently, absolute paired changes include Monte Carlo variation even for an invariant algorithm; signed mean contrasts and their seed uncertainty are needed before claiming systematic orientation sensitivity. An independent-seed identity control or explicit distributional coupling would be a separate follow-up if quantifying excess sensitivity becomes central. Main uncertainty remains conditional on this cell, these profiles and selected rotations; do not treat time samples or rotations as independent cells.

Boundary interaction per method/seed/rotation is [log L_Q−log L_I]_project − [log L_Q−log L_I]_resample. Report its components and boundary encounters. This estimates an intervention on the algorithm, not a universal percentage of rotation sensitivity attributable to boundaries. Ball-versus-box differences also change domain volume and attainable optimum, so cannot alone identify a boundary-handling mechanism.

Check whether isotropic BO's acquisition fits/candidates actually remain consistent; examine optimiser statuses when they do not. A post-implementation numerical follow-up compares paired candidate-path distances and a frozen-history isotropic-GP posterior/LogEI control, without new simulator calls. This isolates kernel mathematics from the sequential fitting/acquisition pipeline; it does not by itself prove which solver detail caused divergence. Inspect whether CMA saw enough generations to adapt, whether ARD scales hit constraints, whether recommendations approach boundaries, and whether equal loss conceals branch ambiguity. Compare corrected/uncorrected OCV using identical designs, and assess one-RC versus two-RC residuals before interpreting parameters physically. A two-RC reference outside the unit ball explicitly demonstrates a domain limitation, not an algorithm failure.

Changing bounds, OCV construction, profiles, noise, budgets, kernel priors, learning rates or algorithm families after this v1 pilot requires a new protocol/config revision before additional scientific comparisons. Record why, retain old artifacts, and do not retroactively pool changed runs. A stable null result or warning-limited method is reportable evidence if investigated. Subsequent interpretation and checks are recorded separately from this frozen specification.

## 9. Acceptance and notebook flow

1. Parse real workbooks; verify overlaps, schema, sign, stage selection and OCV interpolation.
2. Verify analytic RC response, recovery on an identifiable synthetic trace, branch-swap equivalence and native-time prediction.
3. Verify orthogonality, coordinate round trips, preserved spectra/losses, physical feasibility, initial design matching, scalar/ARD kernel transformations and covariance transformations.
4. Verify counted failures, budget exhaustion, repair, seeded repeatability, file hashes and trace/summary consistency. No demand for identical cross-rotation CMA trajectories.
5. Keep assessment voltage out of fitting and acquisition. Offset uses DST rest only.
6. Execute notebooks 01–04 from fresh kernels; inspect actual plots; save run IDs and curated tables/figures. Notebook 03 uses the same runner as the CLI and reuses only verified results.
7. Never replace measured application outputs with synthetic ones, never launch main automatically, and do not claim a submission report has been written.

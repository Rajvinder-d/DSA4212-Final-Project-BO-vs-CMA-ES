# BO versus evolutionary strategies: problem selection review

Historical problem-selection review, prepared 9 November 2026. Battery
calibration was subsequently retained; [Research progression](RESEARCH_PROGRESSION.md)
records the resulting experiments. The alternatives below retain their proposal-stage context.

Historical shortlist: this records the proposals before review of the partner’s
battery pilot. The selected direction and current next steps are in the
[research plan](RESEARCH_PLAN.md). Decision-status statements below describe
the original shortlist, not the current implementation.

The project is selecting an applied problem for comparing Bayesian optimisation
(BO) with evolutionary strategies, provisionally CMA-ES. The preference is for
measured data and a recognisable practical problem. Four candidates are outlined
below. None has been selected, implemented or experimentally validated by the
team. Algorithm variants and computing budgets remain undecided.

Please assess these proposals independently. Check the assignment and cited
sources, challenge the provisional ranking, and distinguish established evidence
from assumptions that require a pilot. The review should help choose a feasible
experiment that explains optimiser behaviour.

## Assignment and methodological basis

The governing source is [Alexandre Thiéry’s project brief](https://alexxthiery.github.io/teaching/bo-vs-es/bo-vs-es.html).
Its central requirement is to “Develop and test an explanation for a difference
in their behaviour.” Suggested directions include dimension, parameter geometry,
noisy evaluations, multimodality, and evaluation cost or parallelism. The brief
requires explicit budgets, comparable information and tuning effort, repeated
runs, and independent assessment of recommendations when evaluations are noisy.
It does not state that the application must be new to the literature.

The four supplied starter papers are:

- [Frazier, *A Tutorial on Bayesian Optimization*](https://arxiv.org/abs/1807.02811): surrogate modelling, acquisition functions and noisy evaluations.
- [Hansen, *The CMA Evolution Strategy: A Tutorial*](https://arxiv.org/abs/1604.00772): covariance adaptation, step-size control and invariance.
- [Eriksson et al., *Scalable Global Optimization via Local Bayesian Optimization*](https://arxiv.org/abs/1910.01739): TuRBO and local modelling for larger search spaces.
- [Hansen et al., *COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting*](https://arxiv.org/abs/1603.08785): benchmarking design and performance measurement.

These provide methods and experimental principles; they do not establish that
the applications below are feasible or novel. A comparison of final predictive
accuracy alone would leave the brief’s explanatory requirement unanswered.

## Candidate problems

### 1. Battery model calibration

**Practical task and data.** Estimate parameters of an equivalent-circuit battery
model from measured current and voltage traces, then assess voltage prediction
on a separate load profile. [CALCE’s battery data](https://calce.umd.edu/data)
include laboratory measurements under dynamic load profiles. The exact cell,
files, model equations and parameter bounds still need selection.

**Optimisation problem.** Minimise voltage prediction error over continuous
physical parameters, such as resistances and capacitances. Model outputs are
simulated, but the fitting target is measured battery behaviour.

**Existing work.** [Pi et al. (2024)](https://arxiv.org/abs/2405.10750) used BO to
identify parameters of an electrochemical equivalent-circuit model, compared it
with gradient descent and particle swarm optimisation, and verified performance
using an experimental drive cycle. This is related precedent, not evidence that
the proposed CALCE setup has been studied. Particle swarm optimisation is not
CMA-ES.

**Proposed question.** How does parameterisation affect GP-based BO and CMA-ES
when fitting the same measured voltage responses? Investigate whether differences
in parameter coupling help explain convergence, rather than assuming either
method will win.

**What needs vetting.** Determine whether a small model has enough optimisation
difficulty to support the study. Check parameter identifiability and the need for
a least-squares baseline. Equivalent parameterisations must preserve the physical
feasible set; changes to initial sampling or priors must be accounted for.
Nonlinear transformations do not inherit CMA-ES’s affine invariance. A cheap
circuit model may make BO overhead a substantial part of total runtime.

### 2. Rainfall–runoff model calibration

**Practical task and data.** Fit a compact hydrological model to observed river
discharge using precipitation and other meteorological inputs. The
[CAMELS dataset](https://ral.ucar.edu/dataset/camels-catchment-attributes-and-meteorology-large-sample-studies-dataset-downloads)
provides catchment data for this purpose. A small set of catchments would provide
separate real problem instances, with held-out periods for validation.

**Optimisation problem.** Choose continuous storage and flow parameters to
minimise a specified discharge error. The hydrological model, objective and
calibration periods remain open.

**Existing work.** [A 2022 study](https://doi.org/10.1016/j.envsoft.2021.105235)
used BO to calibrate SWAT. The [eWaterCycle platform paper](https://gmd.copernicus.org/articles/15/5371/2022/)
demonstrates CMA-ES calibration of a seven-parameter TOPMODEL implementation.
These establish use of both methods in hydrological calibration, not a direct
comparison on the proposed CAMELS setup.

**Proposed question.** Does access to parallel evaluations change the relative
time needed by BO and CMA-ES to reach a fixed calibration quality? A suitable
batch BO variant would be needed for a meaningful parallel comparison.

**What needs vetting.** Find a maintained model implementation, check required
inputs and warm-up periods, and measure evaluation time. Domain setup could
dominate the project. Different outcomes across catchments would demonstrate
variation, but would not by themselves explain its cause. Worker count should
be varied on the same objectives, with total resource use reported.

### 3. Hydraulic fault detection

**Practical task and data.** Detect a specified component fault from pressure,
flow, temperature and other sensor measurements in the
[UCI Condition Monitoring of Hydraulic Systems dataset](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems).
It contains 2,205 measured operating cycles from a laboratory test rig with
controlled component conditions. It is not a record of faults observed in an
operational fleet.

**Optimisation problem.** Tune a fixed classifier’s continuous hyperparameters
against validation performance, using a fixed feature pipeline. The initial
suggestion was an RBF SVM, optimising regularisation and kernel width.

**Existing work.** [Tang et al. (2022)](https://doi.org/10.1016/j.isatra.2022.01.013)
used Gaussian-process BO to tune a CNN for hydraulic pump fault diagnosis. This
supports the application area, but does not establish the same UCI/SVM/CMA-ES
experiment.

**Proposed question.** If a stochastic training model is justified, how does
allocating evaluations between new configurations and repeated training runs
affect final recommendation quality?

**What needs vetting.** The original two-parameter SVM proposal may be too limited
for the intended investigation. Introducing a larger model requires a practical
reason. A fixed SVM and fixed data split generally give deterministic evaluations;
sensor measurement noise alone does not supply repeated-evaluation noise.
Check dependence between cycles and prevent leakage between training and testing.

### 4. Building energy prediction

**Practical task and data.** Predict household appliance consumption from past
consumption and available environmental measurements using the
[UCI Appliances Energy Prediction dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction).
It contains approximately 4.5 months of measured data at ten-minute intervals
from one house. A forecasting task must define its horizon and use inputs
available at prediction time.

**Optimisation problem.** Tune continuous hyperparameters of a fixed prediction
model, using chronological validation and a later test period.

**Existing work.** [A 2025 hyperparameter-optimisation study](https://ieeexplore.ieee.org/document/11293904/)
used this dataset to compare an adaptive genetic algorithm with several tuning
frameworks, including BO-based tools. This does not establish a CMA-ES comparison
or the same forecasting protocol.

**Proposed question.** Do differences in evaluation efficiency persist when
optimiser overhead and parallel workers are included in the budget?

**What needs vetting.** Identify a model with a defensible continuous search
space and manageable training cost. Integer and categorical choices complicate
a standard CMA-ES comparison. Separate prediction from genuine forecasting,
prevent temporal leakage, and restrict conclusions to the available household.

## Provisional assessment

Battery and rainfall–runoff calibration currently appear to offer the clearest
connection between practical parameters and optimiser behaviour. Hydraulic fault
detection and energy prediction remain valid alternatives, but their contribution
would come primarily from the hyperparameter-optimisation experiment.

This ranking is a proposal, not a feasibility result. The literature checks above
establish related work; they do not establish novelty through an exhaustive
search. Existing work may provide a useful reproducible starting point.

## Historical review criteria

The review considered:

1. **Brief alignment:** which candidates support a testable explanation, and
   which currently amount only to an application benchmark?
2. **Prior work:** identify the closest direct BO–CMA-ES comparisons. Distinguish
   the same dataset and objective from related applications, and distinguish
   CMA-ES from genetic algorithms or particle swarm methods. Link primary sources
   and state whether conclusions rely on abstracts or full methods.
3. **Feasibility:** check data access, usable model code, required domain work and
   plausible evaluation cost. Separate verified facts from estimates.
4. **Experimental validity:** for the strongest candidate, specify a hypothesis,
   controlled intervention, necessary baseline, fair budget and a diagnostic
   that could contradict the proposed explanation. Identify confounding changes.
5. **Recommendation:** rank the candidates and give the smallest pilot that would
   determine whether the leading option is worth pursuing. State unresolved
   decisions rather than inventing a computing budget or course requirement.

The review should retain measured data as the main application evidence. An
alternative problem is useful if it addresses a concrete weakness in this
shortlist and has identifiable data and an implementable objective.

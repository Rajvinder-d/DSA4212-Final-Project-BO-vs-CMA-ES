# Reading index

**Current:** [critical literature review](LITERATURE_REVIEW.md), reviewed 9 November 2026. It records primary-source links, inspected sections, partial readings, limitations and implementation decisions. The initial reading plan below is preserved; its “abstracts only” status describes 6 November, not the later review.

## Follow-up evidence priorities — 10 November 2026

The [research plan](RESEARCH_PLAN.md) retains the application and focuses on
explaining the comparison. Review fitting/acquisition behaviour and stochastic
invariance in the exact installed versions before attributing a pilot difference
to kernel or covariance representation.

Related work already establishes battery parameter identification using BO and
other optimisers. The [Battery-Sim-Agent author repository](https://github.com/opqrst-chen/Battery-Sim-Agent)
provides BO and CMA-ES baselines in a PyBaMM-based study. Reading status: repository
description, not a full-methods audit. It establishes related comparison work;
it does not establish the same CALCE/two-RC/rotation experiment. The exact overlap
should be checked before any novelty statement.

For the application, [Elmarghichi et al.](https://beei.org/index.php/EEI/article/view/2637)
use CALCE INR18650-20R DST data for parameter identification and voltage prediction
(abstract inspected). A [2026 two-RC study](https://www.mdpi.com/2313-0105/12/3/84)
uses CALCE DST/FUDS data with recursive least-squares identification and QPSO-tuned
state estimation (experimental setup inspected). Neither is evidence that this
specific controlled BO–CMA comparison has been reproduced. These sources inform
positioning, not a change of dataset or a new algorithm requirement.

---

# Reading and evidence notes

The [project page](https://alexxthiery.github.io/teaching/bo-vs-es/bo-vs-es.html)
was read in full on 6 November 2026. Its four starter-paper links were checked;
only their abstracts and bibliographic records have been reviewed so far.
The questions below are a reading plan, not conclusions from a full review.

| Source | Why read it for this project | Questions to resolve |
| --- | --- | --- |
| Peter I. Frazier, [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811), 2018 | Surrogates, acquisition functions and noisy evaluation | Which modelling and acquisition choices fit the selected objective? How should a final noisy recommendation be assessed? |
| Nikolaus Hansen, [The CMA Evolution Strategy: A Tutorial](https://arxiv.org/abs/1604.00772), 2016, revised 2023 | Covariance adaptation and step-size control | What assumptions underlie the relevant invariances, and which settings or boundary treatments alter them? |
| David Eriksson et al., [Scalable Global Optimization via Local Bayesian Optimization](https://arxiv.org/abs/1910.01739), NeurIPS 2019, arXiv revised 2020 | Local probabilistic models and the TuRBO approach | Would a local Bayesian method help distinguish a surrogate limitation from a limitation of the particular global implementation? |
| Nikolaus Hansen et al., [COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting](https://arxiv.org/abs/1603.08785), arXiv 2016, revised 2020 | Benchmark instances, target values and evaluation-count measurement | Which comparison measures and instance design are appropriate, without importing a full benchmark unnecessarily? |

Before making a method claim, add a note identifying the exact section/equation,
its assumptions, its relevance to the chosen experiment and what the implementation
actually does. Distinguish a paper's reported result from a prediction for this
study. Verify library behaviour against its versioned implementation or official
documentation before fixing dependencies.

For new sources, record citation, access/version date, reading status and the
decision they inform. A citation in this list does not mean its implementation
has been adopted or its results reproduced.


## Follow-up bibliography check — 10 November 2026

The bibliography check verified author metadata for LogEI against the NeurIPS paper, TuRBO
and COCO against their arXiv records, Tian et al. against the MERL publication
record, and Grandjean et al. against the publisher page. These checks populated
`report/references.bib`. These were bibliographic metadata checks; the
literature review separately identifies the sections or abstracts inspected.

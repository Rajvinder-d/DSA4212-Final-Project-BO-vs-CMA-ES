# Post hoc paired rotation-method contrasts

This report-stage extension reads unchanged V4 battery exports. It launches no
optimisation and does not amend the frozen V4 protocol. Reproduce from the root:

```bash
MPLCONFIGDIR=artifacts/.matplotlib .venv/bin/python scripts/analyse_rotation_contrasts.py
```

For each method, R = log10(rotated loss / identity loss). The contrast is R_left
minus R_right, paired within initial design and orientation. CMA replicate
contrasts are averaged over five runs within each design. Twenty designs (20–39)
are resampled with replacement 5,000 times, seed 4129, to estimate the mean and
percentile 95% interval. Intervals are pointwise and conditional on the chosen
rotations; they do not correct for multiple comparisons. Positive values mean a
larger **signed** rotation effect for ARD/diagonal, not necessarily worse absolute
quality or greater absolute sensitivity.

| Contrast | Calls | Rotation | Estimate | Pointwise 95% interval |
| --- | ---: | --- | ---: | --- |
| ARD minus scalar, continuous | 100 | Q404 | 0.152 | [0.055, 0.255] |
| ARD minus scalar, continuous | 100 | Q505 | 0.117 | [0.025, 0.235] |
| ARD minus scalar, continuous | 100 | Q606 | 0.196 | [0.144, 0.252] |
| ARD minus scalar, pool | 100 | Q404 | 0.133 | [0.055, 0.220] |
| ARD minus scalar, pool | 100 | Q505 | 0.109 | [0.042, 0.181] |
| ARD minus scalar, pool | 100 | Q606 | 0.085 | [0.020, 0.156] |
| Diagonal minus full CMA | 100 | Q404 | -0.049 | [-0.120, 0.024] |
| Diagonal minus full CMA | 100 | Q505 | -0.041 | [-0.111, 0.026] |
| Diagonal minus full CMA | 100 | Q606 | -0.084 | [-0.172, -0.005] |
| Diagonal minus full CMA | 300 | Q404 | 0.065 | [0.023, 0.106] |
| Diagonal minus full CMA | 300 | Q505 | 0.081 | [0.040, 0.120] |
| Diagonal minus full CMA | 300 | Q606 | 0.082 | [0.034, 0.133] |

All ARD contrasts and CMA-300 contrasts are positive. At 100 calls, two CMA
comparisons remain unresolved and Q606 is negative. The mixed early result
qualifies a blanket diagonal-sensitivity claim; comparing intervals across
budgets is not a direct interaction test. No percentages of rotation sensitivity
attributable to boundary handling are inferred.

`rotation_method_pairs.csv` contains 240 design-level differences;
`rotation_method_intervals.csv` contains all 12 summaries. `validation.json`
records input/output hashes, package versions, aggregation and bootstrap settings.
The generator rejects missing/duplicate arm records, invalid losses, inconsistent
identity losses and stored-effect discrepancies. Historical raw execution was
not re-audited by this analysis.

The analysis script now lives at `scripts/analyse_rotation_contrasts.py` and
writes to ignored `artifacts/report-checks/rotation/` by default. Its computation
is unchanged; only output-directory selection was added. `--output-dir PATH`
selects another destination. Compare the recomputed CSV values with the submitted
tables here. The historical `validation.json` retains the generating script’s
original path and hash; it is a record of that execution, not the current CLI.
`submission_validation.json` records the current file layout and checks.

Historical input paths in `validation.json` use `reports/`; the same numerical
files now reside under `results/`. Its recorded generator hashes identify the
original analysis revision. The current scripts use the renamed directory.


The current report also includes the V3 conditioning display and V4 same-box
boundary table. `python scripts/prepare_report_evidence.py --check` validates
all twelve conditioning intervals, twelve boundary intervals and six budget
changes against the Markdown table where applicable. Without `--check`, it also
renders the conditioning figure using Matplotlib. These are presentations of
existing exports, not new experiments or newly declared hypotheses. The final
PDF can be rebuilt with `python scripts/build_report.py --soffice soffice`.

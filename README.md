# Patient-Specific Setup Error Threshold Prediction (Stacking Ensemble)

[![DOI](https://zenodo.org/badge/1338980828.svg)](https://doi.org/10.5281/zenodo.22004922)

Analysis code for the manuscript:

> **Prediction of patient-specific setup error thresholds using stacking ensemble meta-learning in breast-conserving hypofractionated radiotherapy**
> Weiheng Zhang, Yingqi Liu, Bing Zou, Xingliu Wang, Xuehui Zhao, Juanjuan Chen*
> Department of Oncology, the Second Affiliated Hospital of Nanchang University, Nanchang, Jiangxi, China

## Overview

This repository contains the complete R analysis pipeline used in the study:

1. **Data preprocessing** — long-format reshaping of patient-axis-level dose–error data (50 patients x 3 axes x 11 simulated setup-error levels = 1650 observations); outcome defined as D95 < 95% of prescribed dose.
2. **Grouped data splitting** — patient-level (`ID`) group split, 80/20 (40/10 patients; 1320/330 observations), and repeated 5-fold cross-validation (5 repeats) grouped by patient.
3. **Candidate model tuning** — 8 base learners (DT, CT, RF, bagged C5.0 trees, glmnet logistic regression, SVM, MLP, KNN) tuned with ANOVA racing (`finetune::tune_race_anova`, 80 candidate configurations per model).
4. **Stacking ensemble** — `stacks::blend_predictions()` + `fit_members()`; non-negative lasso (glmnet) meta-learner.
5. **Test-set evaluation** — ROC-AUC, accuracy, Brier score with patient-level bootstrap (2000 resamples) 95% confidence intervals.
6. **Calibration** — Platt scaling (logistic calibration) via the `probably` package.
7. **Model interpretation** — permutation variable importance, partial dependence profiles, and SHAP values (`DALEXtra`).
8. **Decision curve analysis** — `dcurves`, including the factor-level fix for the event definition.
9. **Patient-specific threshold prediction** — illustrative worked example scanning setup error from -0.5 to 0.5 cm to locate the individualised intervention threshold (predicted probability = 0.5).

## Citation

If you use this code, please cite the archived release:
https://doi.org/10.5281/zenodo.22004922

## Requirements

- R >= 4.4 (developed under R 4.5.2)
- Packages: `tidyverse`, `tidymodels`, `readxl`, `janitor`, `future`, `bonsai`, `finetune`, `here`, `baguette`, `stacks`, `DALEXtra`, `probably`, `dcurves`

## Data availability

The analysis dataset is **not** included in this repository because it derives from
institutional treatment-planning data and is not publicly shareable. To run the
pipeline on your own data, place a de-identified file under `data/` and adjust the
import block and variable names at the top of the script.

## Reproducibility

All random seeds are fixed in the script (split: 1501; resampling: 1502; racing: 1503;
stacking: 1504; bootstrap CI: 1505; interpretation: 1511-1512).

## License

MIT License (see `LICENSE`).

## Contact

Corresponding author: Juanjuan Chen, Department of Oncology, the Second Affiliated
Hospital of Nanchang University, Nanchang 330006, Jiangxi, China.

# Julius Elemans

Student number: 20260518

## Overview

This project predicts whether a defendant will reoffend within two years using the COMPAS dataset. The pipeline loads and cleans the data, preprocesses the features, trains a classification model, and compares false-positive rates between race groups. Race is kept for fairness evaluation but is not used to train the model.

See `data/README.md` for the full problem description and data dictionary.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── data_diagnostics.py # missingness, invalid values + duplicate checks
│   ├── preprocessing.py    # cleaning, preprocessing + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 2 | Model comparison | Forked the repository, added my name and student number to the README, changed the model from logistic regression to a decision tree, and compared the results. |
| 3 | EDA and preprocessing | Replaced row deletion with imputation, cleaned inconsistent categories and invalid values, removed duplicate and redundant data, added missingness indicators, and placed preprocessing inside the model pipeline to prevent data leakage. I also checked consistency between `age`/`age_cat` and `decile_score`/`score_text`. |

## Preprocessing decisions

| Column(s) | Problem | Decision |
|---|---|---|
| `age` | Missing values and 9 values outside the valid adult range | Invalid values become missing, then median imputation is fitted on the training data. |
| `juv_fel_count` | Missing values and 5 negative counts | Negative values become missing, then median imputation is used. |
| `priors_count` | Missing values, `-` placeholders, and 6 extreme values | Convert invalid values to missing, use median imputation, and add `priors_count_was_missing`. |
| `c_charge_degree` | Missing values and inconsistent labels | Normalize to `F`/`M`, use mode imputation, and add `c_charge_degree_was_missing`. |
| `sex` and `race` | Different spelling/capitalization and placeholder values | Normalize both categories. Sex uses mode imputation; missing race is labelled `Unknown` so the fairness audit does not assign it to another group. Race remains excluded from model features. |
| `age_cat` | Six clear contradictions with valid numeric ages | Recreate the category from the more detailed numeric age. The dataset treats age 45 as `Greater than 45`. |
| `score_text` | Ten contradictions with valid `decile_score` values | Recreate it from the numeric score using 1–4 = Low, 5–7 = Medium, and 8–10 = High. It is only used for comparison with COMPAS. |
| Duplicate rows | 72 repeated records, also identified by repeated IDs | Keep the first record for each ID. |
| `prior_offenses`, `age_in_months`, `juvenile_total` | Duplicate or derived information | Drop them and keep the more detailed original features. |

Numeric features use median imputation and standard scaling. Categorical features use most-frequent imputation and one-hot encoding. The imputer, scaler, and encoder are fitted only on the training set through a scikit-learn pipeline.

## Best Model

| Pipeline | Model | Train accuracy | Test accuracy | Test F1 | Train-test gap |
|---|---|---:|---:|---:|---:|
| Week 2 baseline | Logistic regression | 0.680 | 0.678 | 0.630 | 0.002 |
| Week 2 baseline | Decision tree | 0.829 | 0.629 | 0.550 | 0.200 |
| Week 3 preprocessing | Logistic regression | 0.673 | 0.665 | 0.590 | 0.007 |
| Week 3 preprocessing | Decision tree | 0.792 | 0.611 | 0.513 | 0.181 |

Logistic regression is still the best current model. Its test accuracy is higher and its small train-test gap shows much less overfitting than the decision tree. Week 3 preprocessing did not improve accuracy, but it keeps 7,214 cleaned records instead of dropping every incomplete row and produces consistent groups for the fairness report.

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.

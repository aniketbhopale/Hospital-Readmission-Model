# 30-Day Diabetic Readmission

This project estimates whether a diabetic inpatient is likely to return within 30 days (`target_lt30`) using a LightGBM model trained on processed encounter records. Instead of keeping everything inside notebooks, `Model.py` handles training and threshold setup and lets you check incoming patients interactively by `patient_nbr`.

## How the Project Is Set Up

```text
diabetic-readmission-30d/
├── Feature engineering & Processing/
│   ├── diabetic_data.csv
│   ├── Final_diabetic_data.csv
│   ├── new_patients.csv
│   └── Processing & selection.ipynb
├── Model.py
├── Evaluation/
│   └── Model-evaluation.ipynb
└── Readme.md

```

## Why Assess 30-Day Readmission Risk?

Flagging early return risk within 30 days of discharge matters for several reasons:

* **Gaps in preventive and therapeutic care:** Even with strong evidence that preventive and therapeutic interventions improve outcomes for people with diabetes, many patients still miss out on them.
* **Inconsistent inpatient glycemic management:** Hospital care routines can be uneven and sometimes fall short on steady blood glucose control during the stay.
* **Higher hospital costs:** Early readmissions add avoidable strain and managing expenses onto hospital systems.
* **Patient health and survival risks:** Missed or incomplete care increases the risk of complications, illness burden, and mortality for diabetic patients after discharge.


Historical training data (`Final_diabetic_data.csv`) comes from ten years (1999–2008) of clinical care records across 130 US hospitals and integrated delivery networks covering diabetic inpatient stays of up to 14 days with lab tests and medications. Incoming patient records are supplied via `new_patients.csv` without needing `target_lt30`.

## Model & Threshold Choices

* **Algorithm:** `LGBMClassifier` with `n_estimators = 800` (with 50-round early stopping), `learning_rate = 0.017268689827900557`, `num_leaves = 55`, `max_depth = 8`, `min_child_samples = 72`, `subsample = 0.7826262456580011` (`subsample_freq = 1`), `colsample_bytree = 0.7505188431637575`, `reg_alpha = 0.027449208711527943`, and `reg_lambda = 2.2718670804240984`.
* **Patient-Grouped Splits:** Uses `GroupShuffleSplit` grouped by `patient_nbr` so encounters from the same patient do not leak across training and validation sets.

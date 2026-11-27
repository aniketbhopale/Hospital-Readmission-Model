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

LightGBM Model: Uses LightGBM to learn patterns across mixed clinical and encounter features with early stopping and regularization to keep generalization steady.

Patient-Grouped Splits: Keeps encounters from the same patient grouped together across splits so patient records do not leak between training and validation.

Recall-Focused Threshold: Prioritizes catching early readmissions—which are the minority class—by tuning the decision cutoff rather than relying blindly on a default split.

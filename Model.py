import argparse
from pathlib import Path
import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
MIN_RECALL = 0.50
BASE_DIR = Path(r"C:\Aniket\Hospital Model Final")
DEFAULT_ML_CSV = Path(r"C:\Aniket\Hosp model\Final_diabetic_data.csv")
DEFAULT_NEW_CSV = BASE_DIR / "Feature engineering & Processing" / "new_patients.csv"

LGB_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.017268689827900557,
    "num_leaves": 55,
    "max_depth": 8,
    "min_child_samples": 72,
    "subsample": 0.7826262456580011,
    "subsample_freq": 1,
    "colsample_bytree": 0.7505188431637575,
    "reg_alpha": 0.027449208711527943,
    "reg_lambda": 2.2718670804240984,
    "n_jobs": -1,
    "verbose": -1,
}


def best_threshold_for_recall(y_true, y_prob, min_recall=MIN_RECALL):
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    p, r, t = prec[:-1], rec[:-1], thresh
    mask = r >= min_recall
    if not np.any(mask):
        return float(t[int(np.argmax(r))])
    cand_p, cand_r, cand_t = p[mask], r[mask], t[mask]
    return float(cand_t[int(np.lexsort((cand_r, cand_p))[-1])])


def build_cat_schema(df_ml, feature_cols):
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df_ml[c])]
    cat_dtypes = {}
    for c in cat_cols:
        cats = sorted(df_ml[c].astype(str).fillna("Missing").unique())
        if "Missing" not in cats:
            cats.append("Missing")
        cat_dtypes[c] = pd.CategoricalDtype(categories=cats)
    return cat_dtypes


def apply_cat_schema(df_in, feature_cols, cat_dtypes):
    X_out = df_in[feature_cols].copy()
    for c, dtype_c in cat_dtypes.items():
        s = X_out[c].astype(str).fillna("Missing").astype(dtype_c)
        X_out[c] = s.fillna("Missing")
    return X_out


def fit_readmission_model(ml_csv_path=DEFAULT_ML_CSV, seed=SEED, min_recall=MIN_RECALL):
    df_loaded = pd.read_csv(ml_csv_path, dtype={"payer_code": str})
    feature_cols = [c for c in df_loaded.columns if c not in ["encounter_id", "patient_nbr", "target_lt30"]]
    cat_dtypes = build_cat_schema(df_loaded, feature_cols)
    X_ready = apply_cat_schema(df_loaded, feature_cols, cat_dtypes)
    y_ready = df_loaded["target_lt30"].astype(int)
    groups_ready = df_loaded["patient_nbr"]

    tv_idx, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(X_ready, y_ready, groups=groups_ready))
    X_tv, y_tv, g_tv = X_ready.iloc[tv_idx], y_ready.iloc[tv_idx], groups_ready.iloc[tv_idx]

    tr_idx, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed).split(X_tv, y_tv, groups=g_tv))
    X_tr, y_tr = X_tv.iloc[tr_idx], y_tv.iloc[tr_idx]
    X_va, y_va = X_tv.iloc[va_idx], y_tv.iloc[va_idx]

    clf = LGBMClassifier(**LGB_PARAMS, random_state=seed)
    clf.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va, callbacks=[lgb.early_stopping(50, verbose=False)])
    val_probs = clf.predict_proba(X_va)[:, 1]
    threshold = best_threshold_for_recall(y_va.values, val_probs, min_recall=min_recall)

    return clf, threshold, feature_cols, cat_dtypes, df_loaded


def predict_readmission(input_data, clf, threshold, feature_cols, cat_dtypes):
    if isinstance(input_data, dict):
        df_query = pd.DataFrame([input_data])
    else:
        df_query = input_data.copy()

    if df_query.empty:
        return np.array([], dtype=int)

    for col in feature_cols:
        if col not in df_query.columns:
            df_query[col] = "Missing" if col in cat_dtypes else 0

    X_query = apply_cat_schema(df_query, feature_cols, cat_dtypes)
    probs = clf.predict_proba(X_query)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds if len(preds) > 1 else int(preds[0])


def lookup_patient(patient_nbr, query_df, clf, threshold, feature_cols, cat_dtypes):
    sub = query_df[query_df["patient_nbr"] == patient_nbr]
    if sub.empty:
        print("No matching patient_nbr found.")
        return
    preds = predict_readmission(sub, clf, threshold, feature_cols, cat_dtypes)
    out = pd.DataFrame({
        "encounter_id": sub["encounter_id"].values,
        "patient_nbr": sub["patient_nbr"].values,
        "pred": np.atleast_1d(preds),
    })
    print(out.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-csv", type=str, default=str(DEFAULT_ML_CSV))
    parser.add_argument("--new-csv", type=str, default=None)
    parser.add_argument("--patient-nbr", type=int, default=None)
    args = parser.parse_args()

    clf, threshold, feature_cols, cat_dtypes, df_loaded = fit_readmission_model(args.ml_csv)

    if args.new_csv:
        query_df = pd.read_csv(args.new_csv, dtype={"payer_code": str})
    elif DEFAULT_NEW_CSV.exists():
        query_df = pd.read_csv(DEFAULT_NEW_CSV, dtype={"payer_code": str})
    else:
        raise FileNotFoundError(f"New patient CSV not found at: {DEFAULT_NEW_CSV}")

    if args.patient_nbr is not None:
        lookup_patient(args.patient_nbr, query_df, clf, threshold, feature_cols, cat_dtypes)
    else:
        while True:
            raw = input("Enter patient_nbr (or press Enter to exit): ").strip()
            if not raw or raw.lower() in {"q", "quit", "exit"}:
                break
            if not raw.isdigit():
                print("Enter a valid integer patient_nbr.")
                continue
            lookup_patient(int(raw), query_df, clf, threshold, feature_cols, cat_dtypes)
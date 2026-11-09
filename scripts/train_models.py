#!/usr/bin/env python3
"""Source-preserving extract of A, B1, C1, B2, C2 training steps.

Original Kaggle source cells: [106, 107, 109, 111, 112].
Requires project workspace restored at /kaggle/working/acoustic_linguistic_research.
This is a reorganized source extract, not an independently revalidated execution.
"""


# ========== ORIGINAL KAGGLE CELL 106 ==========
# %%

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT = Path("/kaggle/working/acoustic_linguistic_research")

DATA = PROJECT / "data/modeling/modeling_dataset_0001-0400.csv"
SPLIT = PROJECT / "data/splits/speaker_groupkfold5_frozen.csv"
CV_CONFIG = PROJECT / "configs/speaker_disjoint_cv_v1.json"
DESIGN = PROJECT / "configs/fluency_feature_design_v1.json"
CHECK = PROJECT / "checkpoints/research_progress.json"

RESULTS = PROJECT / "results/step05"
PREDICTIONS = RESULTS / "baseline_acoustic_oof_predictions.csv"
FOLD_RESULTS = RESULTS / "baseline_acoustic_fold_metrics.csv"
SPEAKER_RESULTS = RESULTS / "baseline_acoustic_speaker_metrics.csv"
REPORT = PROJECT / "reports/step05_baseline_acoustic_ridge_v1.json"

EXPECTED = {
    DATA: "7d20428a14d9e00b88f1a20b36b7147bf5b98234045a093172fef76b68276746",
    SPLIT: "5f87cdc1c2f69922bcb542ee9516f5f89168bf68402fbd2449b28fcf61a375e7",
    CV_CONFIG: "302c7ea0086a54d644a4ebc0b8a5d96d63575a3896b5a29d76dba7e26dacd472",
    DESIGN: "626b0050f58f35fafa4f2efcd1888fda2a195454e544c4cb094cd50e796a1b83"
}

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for path, expected_hash in EXPECTED.items():
    assert sha256(path) == expected_hash, (
        f"Source SHA256 mismatch: {path}"
    )

design = json.loads(DESIGN.read_text(encoding="utf-8"))
cv_config = json.loads(CV_CONFIG.read_text(encoding="utf-8"))

assert cv_config["status"] == "frozen"
assert cv_config["number_of_folds"] == 5
assert cv_config["speaker_disjoint"] is True

data = pd.read_csv(
    DATA,
    dtype={"audio_id": "string", "speaker": "string"}
)

split = pd.read_csv(
    SPLIT,
    dtype={"audio_id": "string", "speaker": "string"}
)

ids = ["pilot_index", "audio_id", "speaker", "fluency"]

audio = data[ids].drop_duplicates()

assert len(audio) == 400
assert audio["audio_id"].is_unique

acoustic_features = [
    col for col in data.columns
    if col not in (
        ids
        + ["hypothesis_id"]
        + design["linguistic_feature_names"]
        + design["prompt_feature_names"]
    )
]

assert len(acoustic_features) == 14
assert len(set(acoustic_features)) == 14
assert "fluency" not in acoustic_features

# Reduce to one acoustic record per audio.
acoustic = data.loc[
    data["hypothesis_id"] == "H1",
    ids + acoustic_features
].copy()

assert len(acoustic) == 400

df = acoustic.merge(
    split[ids + ["validation_fold"]],
    on=ids,
    how="inner",
    validate="one_to_one"
)

assert len(df) == 400
assert df["validation_fold"].between(1, 5).all()

prediction_rows = []
fold_rows = []

# Fixed alpha for the initial benchmark.
RIDGE_ALPHA = 1.0

for fold in range(1, 6):

    train = df.loc[
        df["validation_fold"] != fold
    ].copy()

    val = df.loc[
        df["validation_fold"] == fold
    ].copy()

    assert len(train) == 320
    assert len(val) == 80

    assert set(train["speaker"]).isdisjoint(
        set(val["speaker"])
    )

    X_train = train[acoustic_features]
    X_val = val[acoustic_features]

    y_train = train["fluency"].to_numpy(dtype=float)
    y_val = val["fluency"].to_numpy(dtype=float)

    # M0: training-fold mean only.
    pred_mean = np.full(
        len(val),
        y_train.mean(),
        dtype=float
    )

    # A: imputation and scaling learned on training only.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA))
    ])

    model.fit(X_train, y_train)
    pred_acoustic = model.predict(X_val)

    assert np.isfinite(pred_acoustic).all()

    fold_rows.append({
        "fold": fold,
        "train_audio_count": len(train),
        "validation_audio_count": len(val),
        "train_speaker_count": train["speaker"].nunique(),
        "validation_speaker_count": val["speaker"].nunique(),
        "train_mean_fluency": float(y_train.mean()),
        "validation_mean_fluency": float(y_val.mean()),
        "mean_baseline_mae": float(
            mean_absolute_error(y_val, pred_mean)
        ),
        "mean_baseline_rmse": float(
            np.sqrt(mean_squared_error(y_val, pred_mean))
        ),
        "acoustic_ridge_mae": float(
            mean_absolute_error(y_val, pred_acoustic)
        ),
        "acoustic_ridge_rmse": float(
            np.sqrt(mean_squared_error(y_val, pred_acoustic))
        )
    })

    for i, (_, row) in enumerate(val.iterrows()):
        prediction_rows.append({
            "pilot_index": int(row["pilot_index"]),
            "audio_id": str(row["audio_id"]),
            "speaker": str(row["speaker"]),
            "validation_fold": fold,
            "fluency": float(y_val[i]),
            "mean_baseline_prediction": float(pred_mean[i]),
            "acoustic_ridge_prediction": float(pred_acoustic[i])
        })

predictions = pd.DataFrame(prediction_rows).sort_values(
    "pilot_index"
).reset_index(drop=True)

fold_metrics = pd.DataFrame(fold_rows)

assert len(predictions) == 400
assert predictions["audio_id"].is_unique
assert predictions["speaker"].nunique() == 20
assert predictions["validation_fold"].nunique() == 5
assert predictions["fluency"].notna().all()

# Each speaker has predictions from exactly one fold.
assert predictions.groupby("speaker")[
    "validation_fold"
].nunique().eq(1).all()

y = predictions["fluency"].to_numpy()

overall_metrics = {}

for name, col in [
    ("M0_mean_baseline", "mean_baseline_prediction"),
    ("A_acoustic_ridge", "acoustic_ridge_prediction")
]:
    pred = predictions[col].to_numpy()

    overall_metrics[name] = {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred)))
    }

speaker_rows = []

for speaker, group in predictions.groupby("speaker"):
    speaker_rows.append({
        "speaker": str(speaker),
        "fold": int(group["validation_fold"].iloc[0]),
        "audio_count": len(group),
        "mean_fluency": float(group["fluency"].mean()),
        "mean_baseline_mae": float(
            mean_absolute_error(
                group["fluency"],
                group["mean_baseline_prediction"]
            )
        ),
        "acoustic_ridge_mae": float(
            mean_absolute_error(
                group["fluency"],
                group["acoustic_ridge_prediction"]
            )
        )
    })

speaker_metrics = pd.DataFrame(speaker_rows)

assert len(speaker_metrics) == 20

RESULTS.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

def save_csv_safely(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8")

    if path.exists():
        assert sha256(path) == sha256(tmp), (
            f"Existing result differs: {path}"
        )
        tmp.unlink()
    else:
        tmp.replace(path)

save_csv_safely(predictions, PREDICTIONS)
save_csv_safely(fold_metrics, FOLD_RESULTS)
save_csv_safely(speaker_metrics, SPEAKER_RESULTS)

report = {
    "status": "PASS",
    "stage": "initial_fixed_alpha_benchmark",
    "model_A": "Ridge",
    "ridge_alpha": RIDGE_ALPHA,
    "hyperparameter_tuning_performed": False,
    "validation": "frozen_5fold_speaker_disjoint",
    "unique_audio_count": 400,
    "unique_speaker_count": 20,
    "acoustic_feature_count": 14,
    "prediction_count_per_model": 400,
    "metrics": overall_metrics,
    "predictions_sha256": sha256(PREDICTIONS),
    "fold_metrics_sha256": sha256(FOLD_RESULTS),
    "speaker_metrics_sha256": sha256(SPEAKER_RESULTS),
    "modeling_dataset_sha256": sha256(DATA),
    "frozen_split_sha256": sha256(SPLIT),
    "note": (
        "Predictions are out-of-fold for unseen speakers. "
        "Ridge alpha is fixed at 1.0, without tuning. "
        "No official test data were used."
    )
}

if REPORT.exists():
    previous = json.loads(REPORT.read_text(encoding="utf-8"))
    assert previous == report, (
        "Existing benchmark report differs."
    )
else:
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    tmp.replace(REPORT)

progress = json.loads(CHECK.read_text(encoding="utf-8"))
progress["step5_initial_acoustic_benchmark"] = "qc_pass"
progress["step5_initial_acoustic_predictions_sha256"] = sha256(PREDICTIONS)
progress["next_task"] = (
    "Review acoustic Ridge and mean baseline results "
    "before transcript-only Ridge experiments"
)
progress["updated_at"] = datetime.now(
    timezone.utc
).isoformat()

CHECK.write_text(
    json.dumps(progress, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("=== STEP 5.3 INITIAL ACOUSTIC BENCHMARK ===")
print("Status:", report["status"])
print("Validation: frozen 5-fold speaker-disjoint")
print("Unique audios:", 400)
print("Features:", len(acoustic_features))
print("Ridge alpha:", RIDGE_ALPHA)
print("Hyperparameter tuning: NO")

print("\n=== OVERALL OUT-OF-FOLD METRICS ===")
for name, metrics in overall_metrics.items():
    print(
        f"{name}: "
        f"MAE={metrics['MAE']:.4f}, "
        f"RMSE={metrics['RMSE']:.4f}"
    )

print("\n=== PER-FOLD METRICS ===")
print(fold_metrics.round(4).to_string(index=False))

print("\n=== PER-SPEAKER MAE ===")
print(speaker_metrics.round(4).to_string(index=False))

print("\nPredictions CSV:", PREDICTIONS)
print("Predictions SHA256:", sha256(PREDICTIONS))
print("Report:", REPORT)
print("Checkpoint updated.")



# ========== ORIGINAL KAGGLE CELL 107 ==========
# %%

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT = Path("/kaggle/working/acoustic_linguistic_research")
DATA = PROJECT / "data/modeling/modeling_dataset_0001-0400.csv"
SPLIT = PROJECT / "data/splits/speaker_groupkfold5_frozen.csv"
DESIGN = PROJECT / "configs/fluency_feature_design_v1.json"
CV = PROJECT / "configs/speaker_disjoint_cv_v1.json"
ACOUSTIC_PRED = PROJECT / "results/step05/baseline_acoustic_oof_predictions.csv"
CHECK = PROJECT / "checkpoints/research_progress.json"

RESULT_DIR = PROJECT / "results/step05"
PRED_OUT = RESULT_DIR / "b1_transcript_ridge_oof_predictions.csv"
FOLD_OUT = RESULT_DIR / "b1_transcript_ridge_fold_metrics.csv"
SPEAKER_OUT = RESULT_DIR / "b1_transcript_ridge_speaker_metrics.csv"
REPORT = PROJECT / "reports/step05_b1_transcript_ridge_v1.json"

EXPECTED = {
    DATA: "7d20428a14d9e00b88f1a20b36b7147bf5b98234045a093172fef76b68276746",
    SPLIT: "5f87cdc1c2f69922bcb542ee9516f5f89168bf68402fbd2449b28fcf61a375e7",
    DESIGN: "626b0050f58f35fafa4f2efcd1888fda2a195454e544c4cb094cd50e796a1b83",
    CV: "302c7ea0086a54d644a4ebc0b8a5d96d63575a3896b5a29d76dba7e26dacd472",
    ACOUSTIC_PRED: "48407b4eac0f213e4ebf39d5b88aa131178e1aca3597cd41011cef168eac1595"
}

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for path, expected in EXPECTED.items():
    assert path.is_file(), f"Missing: {path}"
    assert sha256(path) == expected, f"SHA mismatch: {path}"

design = json.loads(DESIGN.read_text(encoding="utf-8"))
cv = json.loads(CV.read_text(encoding="utf-8"))

assert cv["status"] == "frozen"
assert cv["speaker_disjoint"] is True

FEATURES = design["linguistic_feature_names"]
assert len(FEATURES) == 7

data = pd.read_csv(
    DATA,
    dtype={"audio_id": "string", "speaker": "string"}
)
split = pd.read_csv(
    SPLIT,
    dtype={"audio_id": "string", "speaker": "string"}
)

assert len(data) == 2000
assert len(split) == 400
assert all(c in data.columns for c in FEATURES)

keys = ["pilot_index", "audio_id", "speaker", "fluency"]

df = data.merge(
    split[keys + ["validation_fold"]],
    on=keys,
    how="inner",
    validate="many_to_one"
)

assert len(df) == 2000
assert df["audio_id"].nunique() == 400
assert df["speaker"].nunique() == 20
assert df.groupby("audio_id")["hypothesis_id"].nunique().eq(5).all()
assert df.groupby("speaker")["validation_fold"].nunique().eq(1).all()

predictions = []
fold_metrics = []

ALPHA = 1.0

for fold in range(1, 6):

    train = df.loc[df["validation_fold"] != fold].copy()
    val = df.loc[df["validation_fold"] == fold].copy()

    assert len(train) == 1600
    assert len(val) == 400
    assert train["audio_id"].nunique() == 320
    assert val["audio_id"].nunique() == 80
    assert train["speaker"].nunique() == 16
    assert val["speaker"].nunique() == 4

    assert set(train["speaker"]).isdisjoint(set(val["speaker"]))
    assert set(train["audio_id"]).isdisjoint(set(val["audio_id"]))

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=ALPHA))
    ])

    model.fit(
        train[FEATURES],
        train["fluency"].to_numpy(dtype=float)
    )

    pred = model.predict(val[FEATURES])
    assert np.isfinite(pred).all()

    result = val[
        keys + ["hypothesis_id", "validation_fold"]
    ].copy()
    result["b1_prediction"] = pred
    predictions.append(result)

    # Each fold has 80 audios and exactly five
    # predictions per audio.
    fold_metrics.append({
        "fold": fold,
        "train_speakers": 16,
        "validation_speakers": 4,
        "train_unique_audios": 320,
        "validation_unique_audios": 80,
        "validation_hypotheses": 400,
        "MAE": float(mean_absolute_error(val["fluency"], pred)),
        "RMSE": float(np.sqrt(
            mean_squared_error(val["fluency"], pred)
        ))
    })

oof = pd.concat(predictions, ignore_index=True).sort_values(
    ["pilot_index", "hypothesis_id"]
).reset_index(drop=True)

fold_table = pd.DataFrame(fold_metrics)

assert len(oof) == 2000
assert not oof.duplicated(["audio_id", "hypothesis_id"]).any()
assert oof["audio_id"].nunique() == 400

# Absolute errors for every hypothesis.
oof["absolute_error"] = (
    oof["fluency"] - oof["b1_prediction"]
).abs()

# Per-audio instability of B1 prediction across H1-H5.
audio_stability = oof.groupby("audio_id")["b1_prediction"].agg(
    ["min", "max", "std"]
)

audio_stability["prediction_range"] = (
    audio_stability["max"] - audio_stability["min"]
)

# Per-hypothesis accuracy on the same 400 audios.
hypothesis_metrics = {}
for hid, group in oof.groupby("hypothesis_id"):
    hypothesis_metrics[hid] = {
        "MAE": float(mean_absolute_error(
            group["fluency"], group["b1_prediction"]
        )),
        "RMSE": float(np.sqrt(mean_squared_error(
            group["fluency"], group["b1_prediction"]
        )))
    }

# Speaker-level MAE: 20 independent speaker groups.
speaker_rows = []
for speaker, group in oof.groupby("speaker"):
    speaker_rows.append({
        "speaker": str(speaker),
        "fold": int(group["validation_fold"].iloc[0]),
        "unique_audios": group["audio_id"].nunique(),
        "hypothesis_rows": len(group),
        "MAE": float(group["absolute_error"].mean())
    })

speaker_table = pd.DataFrame(speaker_rows)
assert len(speaker_table) == 20

overall = {
    "MAE": float(mean_absolute_error(
        oof["fluency"], oof["b1_prediction"]
    )),
    "RMSE": float(np.sqrt(mean_squared_error(
        oof["fluency"], oof["b1_prediction"]
    ))),
    "mean_prediction_range": float(
        audio_stability["prediction_range"].mean()
    ),
    "median_prediction_range": float(
        audio_stability["prediction_range"].median()
    ),
    "mean_prediction_sd": float(
        audio_stability["std"].mean()
    ),
    "audios_with_nonzero_prediction_range": int(
        (audio_stability["prediction_range"] > 1e-10).sum()
    )
}

# Compare with previously saved acoustic-only OOF predictions.
a = pd.read_csv(
    ACOUSTIC_PRED,
    dtype={"audio_id": "string", "speaker": "string"}
)

assert len(a) == 400
assert a["audio_id"].is_unique

comparison = oof.merge(
    a[["audio_id", "acoustic_ridge_prediction"]],
    on="audio_id",
    how="inner",
    validate="many_to_one"
)

assert len(comparison) == 2000
assert comparison["fluency"].notna().all()

a_mae = float(mean_absolute_error(
    comparison["fluency"],
    comparison["acoustic_ridge_prediction"]
))

# The acoustic value is repeated five times for
# comparison, not five independent observations.
assert np.isclose(a_mae, 0.9399, atol=0.0001)

RESULT_DIR.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

def save_csv(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8")
    if path.exists():
        assert sha256(path) == sha256(tmp), (
            f"Existing result differs: {path}"
        )
        tmp.unlink()
    else:
        tmp.replace(path)

save_csv(oof, PRED_OUT)
save_csv(fold_table, FOLD_OUT)
save_csv(speaker_table, SPEAKER_OUT)

report = {
    "status": "PASS",
    "model": "B1_transcript_only_ridge",
    "alpha": ALPHA,
    "hyperparameter_tuning": False,
    "feature_names": FEATURES,
    "feature_count": 7,
    "speaker_disjoint_folds": 5,
    "train_preprocessing_only": True,
    "training_hypotheses_per_audio": 5,
    "unique_validation_audios": 400,
    "validation_hypothesis_rows": 2000,
    "overall_metrics": overall,
    "per_hypothesis_metrics": hypothesis_metrics,
    "reference_acoustic_MAE": a_mae,
    "source_modeling_sha256": sha256(DATA),
    "source_frozen_split_sha256": sha256(SPLIT),
    "predictions_sha256": sha256(PRED_OUT),
    "fold_metrics_sha256": sha256(FOLD_OUT),
    "speaker_metrics_sha256": sha256(SPEAKER_OUT),
    "note": (
        "All five hypotheses used as repeated training examples. "
        "Each audio has equal total weight because each has "
        "five hypotheses. Validation speakers remain unseen. "
        "Hypothesis rows are not independent observations."
    )
}

if REPORT.exists():
    previous = json.loads(REPORT.read_text(encoding="utf-8"))
    assert previous == report, "Existing report differs"
else:
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    tmp.replace(REPORT)

progress = json.loads(CHECK.read_text(encoding="utf-8"))
progress["step5_b1_transcript_ridge"] = "qc_pass"
progress["step5_b1_predictions_sha256"] = sha256(PRED_OUT)
progress["next_task"] = (
    "Review B1 accuracy and prediction instability; "
    "then evaluate prompt-aware B2 and fusion C1/C2"
)
progress["updated_at"] = (
    __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat()
)

CHECK.write_text(
    json.dumps(progress, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("=== STEP 5.4 B1 TRANSCRIPT RIDGE SUMMARY ===")
print("Status:", report["status"])
print("Unique audios:", overall and 400)
print("Speakers:", 20)
print("OOF hypothesis predictions:", len(oof))
print("Features:", len(FEATURES))
print("Alpha:", ALPHA)
print("Speaker-disjoint validation: PASS")

print("\n=== OVERALL COMPARISON ===")
print(f"A acoustic MAE: {a_mae:.4f}")
print(f"B1 transcript MAE: {overall['MAE']:.4f}")
print(f"B1 transcript RMSE: {overall['RMSE']:.4f}")

print("\n=== H1-H5 INDIVIDUAL METRICS ===")
for hid, metrics in sorted(hypothesis_metrics.items()):
    print(
        f"{hid}: MAE={metrics['MAE']:.4f}, "
        f"RMSE={metrics['RMSE']:.4f}"
    )

print("\n=== PREDICTION INSTABILITY ===")
for key in [
    "mean_prediction_range",
    "median_prediction_range",
    "mean_prediction_sd",
    "audios_with_nonzero_prediction_range"
]:
    print(f"{key}: {overall[key]}")

print("\n=== FOLD METRICS ===")
print(fold_table.round(4).to_string(index=False))

print("\n=== SPEAKER METRICS ===")
print(speaker_table.round(4).to_string(index=False))

print("\nPredictions:", PRED_OUT)
print("SHA256:", sha256(PRED_OUT))
print("Report:", REPORT)
print("Checkpoint updated.")



# ========== ORIGINAL KAGGLE CELL 109 ==========
# %%

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

P = Path("/kaggle/working/acoustic_linguistic_research")
DATA = P / "data/modeling/modeling_dataset_0001-0400.csv"
SPLIT = P / "data/splits/speaker_groupkfold5_frozen.csv"
DESIGN = P / "configs/fluency_feature_design_v1.json"
CV = P / "configs/speaker_disjoint_cv_v1.json"
A_PRED = P / "results/step05/baseline_acoustic_oof_predictions.csv"
B1_PRED = P / "results/step05/b1_transcript_ridge_oof_predictions.csv"
CHECK = P / "checkpoints/research_progress.json"

OUTDIR = P / "results/step05"
PRED = OUTDIR / "c1_fusion_ridge_oof_predictions.csv"
FOLDS = OUTDIR / "c1_fusion_ridge_fold_metrics.csv"
AUDIO = OUTDIR / "c1_fusion_prediction_stability_per_audio.csv"
REPORT = P / "reports/step05_c1_fusion_ridge_v1.json"

EXPECTED = {
    DATA: "7d20428a14d9e00b88f1a20b36b7147bf5b98234045a093172fef76b68276746",
    SPLIT: "5f87cdc1c2f69922bcb542ee9516f5f89168bf68402fbd2449b28fcf61a375e7",
    DESIGN: "626b0050f58f35fafa4f2efcd1888fda2a195454e544c4cb094cd50e796a1b83",
    CV: "302c7ea0086a54d644a4ebc0b8a5d96d63575a3896b5a29d76dba7e26dacd472",
    A_PRED: "48407b4eac0f213e4ebf39d5b88aa131178e1aca3597cd41011cef168eac1595",
    B1_PRED: "375ac49b0d5b565adf65d04407d6afc1816f64f5ba108afef8d22458383807dc"
}

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for path, expected in EXPECTED.items():
    assert path.is_file()
    assert sha256(path) == expected, f"SHA mismatch: {path}"

design = json.loads(DESIGN.read_text(encoding="utf-8"))
protocol = json.loads(CV.read_text(encoding="utf-8"))

assert protocol["status"] == "frozen"
assert design["models"]["C1"]["feature_groups"] == [
    "acoustic", "linguistic"
]

data = pd.read_csv(
    DATA, dtype={"audio_id": "string", "speaker": "string"}
)
split = pd.read_csv(
    SPLIT, dtype={"audio_id": "string", "speaker": "string"}
)

ID = ["pilot_index", "audio_id", "speaker"]
KEYS = ID + ["hypothesis_id"]
LING = design["linguistic_feature_names"]
PROMPT = design["prompt_feature_names"]

ACOUSTIC = [
    c for c in data.columns
    if c not in KEYS + ["fluency"] + LING + PROMPT
]

FEATURES = ACOUSTIC + LING

assert len(ACOUSTIC) == 14
assert len(LING) == 7
assert len(FEATURES) == 21
assert "fluency" not in FEATURES
assert not set(FEATURES).intersection(KEYS + PROMPT)

assert len(data) == 2000
assert len(split) == 400
assert not data.duplicated(KEYS).any()

df = data.merge(
    split[ID + ["fluency", "validation_fold"]],
    on=ID + ["fluency"],
    how="inner",
    validate="many_to_one"
)

assert len(df) == 2000
assert df.groupby("audio_id")["hypothesis_id"].nunique().eq(5).all()
assert df.groupby("speaker")["validation_fold"].nunique().eq(1).all()

ALPHA = 1.0
pred_parts = []
fold_rows = []

for fold in range(1, 6):
    train = df.loc[df["validation_fold"] != fold]
    val = df.loc[df["validation_fold"] == fold]

    assert len(train) == 1600
    assert len(val) == 400
    assert train["audio_id"].nunique() == 320
    assert val["audio_id"].nunique() == 80

    assert set(train["speaker"]).isdisjoint(set(val["speaker"]))
    assert set(train["audio_id"]).isdisjoint(set(val["audio_id"]))

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=ALPHA))
    ])

    model.fit(
        train[FEATURES],
        train["fluency"].to_numpy(dtype=float)
    )

    predictions = model.predict(val[FEATURES])
    assert np.isfinite(predictions).all()

    part = val[KEYS + ["fluency", "validation_fold"]].copy()
    part["c1_prediction"] = predictions
    pred_parts.append(part)

    fold_rows.append({
        "fold": fold,
        "validation_audios": 80,
        "validation_hypotheses": 400,
        "MAE": float(mean_absolute_error(
            val["fluency"], predictions
        )),
        "RMSE": float(np.sqrt(mean_squared_error(
            val["fluency"], predictions
        )))
    })

oof = pd.concat(pred_parts, ignore_index=True).sort_values(
    ["pilot_index", "hypothesis_id"]
).reset_index(drop=True)

assert len(oof) == 2000
assert oof["audio_id"].nunique() == 400
assert not oof.duplicated(["audio_id", "hypothesis_id"]).any()

# Verify C1 rows against earlier acoustic and transcript OOF.
a = pd.read_csv(
    A_PRED, dtype={"audio_id": "string", "speaker": "string"}
)
b = pd.read_csv(
    B1_PRED, dtype={"audio_id": "string", "speaker": "string"}
)

comp = oof.merge(
    a[ID + ["fluency", "validation_fold", "acoustic_ridge_prediction"]],
    on=ID + ["fluency", "validation_fold"],
    validate="many_to_one"
).merge(
    b[KEYS + ["fluency", "validation_fold", "b1_prediction"]],
    on=KEYS + ["fluency", "validation_fold"],
    validate="one_to_one"
)

assert len(comp) == 2000

def metrics(y, pred):
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred)))
    }

comparison = {
    "A_acoustic": metrics(
        comp["fluency"], comp["acoustic_ridge_prediction"]
    ),
    "B1_transcript": metrics(
        comp["fluency"], comp["b1_prediction"]
    ),
    "C1_fusion": metrics(
        comp["fluency"], comp["c1_prediction"]
    )
}

audio = (
    oof.groupby(ID + ["validation_fold"], as_index=False)
    .agg(
        fluency=("fluency", "first"),
        prediction_min=("c1_prediction", "min"),
        prediction_max=("c1_prediction", "max"),
        prediction_sd=("c1_prediction", "std"),
        hypothesis_count=("hypothesis_id", "count")
    )
)

audio["prediction_range"] = (
    audio["prediction_max"] - audio["prediction_min"]
)

assert len(audio) == 400
assert audio["hypothesis_count"].eq(5).all()

hyp_metrics = {
    h: metrics(
        g["fluency"], g["c1_prediction"]
    )
    for h, g in oof.groupby("hypothesis_id")
}

stability = {
    "mean_range": float(audio["prediction_range"].mean()),
    "median_range": float(audio["prediction_range"].median()),
    "p95_range": float(audio["prediction_range"].quantile(0.95)),
    "max_range": float(audio["prediction_range"].max()),
    "variable_audio_count": int(
        (audio["prediction_range"] > 1e-10).sum()
    )
}

critical = oof.loc[
    oof["audio_id"].eq("030210164") &
    oof["hypothesis_id"].eq("H2")
]
assert len(critical) == 1

OUTDIR.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

def save_csv(frame, path):
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8")
    if path.exists():
        assert sha256(path) == sha256(temp), (
            f"Existing file differs: {path}"
        )
        temp.unlink()
    else:
        temp.replace(path)

save_csv(oof, PRED)
save_csv(pd.DataFrame(fold_rows), FOLDS)
save_csv(audio, AUDIO)

report = {
    "status": "PASS",
    "model": "C1_acoustic_linguistic_fusion_ridge",
    "alpha": ALPHA,
    "hyperparameter_tuning": False,
    "feature_count": 21,
    "acoustic_feature_count": 14,
    "linguistic_feature_count": 7,
    "validation": "frozen_5fold_speaker_disjoint",
    "OOF_prediction_rows": 2000,
    "unique_audios": 400,
    "comparison": comparison,
    "per_hypothesis_metrics": hyp_metrics,
    "stability": stability,
    "critical_audio_H2_prediction": float(
        critical["c1_prediction"].iloc[0]
    ),
    "source_dataset_sha256": sha256(DATA),
    "source_split_sha256": sha256(SPLIT),
    "predictions_sha256": sha256(PRED),
    "fold_metrics_sha256": sha256(FOLDS),
    "audio_stability_sha256": sha256(AUDIO),
    "note": (
        "No clipping or hyperparameter tuning. "
        "Training-only imputation and scaling. "
        "All five ASR hypotheses remain in the same speaker fold."
    )
}

if REPORT.exists():
    previous = json.loads(REPORT.read_text(encoding="utf-8"))
    assert previous == report, (
        "Existing report differs; original preserved."
    )
else:
    temp = REPORT.with_suffix(".tmp")
    temp.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    temp.replace(REPORT)

progress = json.loads(CHECK.read_text(encoding="utf-8"))
progress["step5_c1_fusion_ridge"] = "qc_pass"
progress["step5_c1_predictions_sha256"] = sha256(PRED)
progress["next_task"] = (
    "Compare C1 fusion accuracy and ASR-induced instability "
    "against models A and B1"
)
progress["updated_at"] = datetime.now(timezone.utc).isoformat()

CHECK.write_text(
    json.dumps(progress, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("=== STEP 5.6 C1 FUSION RIDGE SUMMARY ===")
print("Status: PASS")
print("Features:", len(FEATURES))
print("Unique audios: 400")
print("OOF predictions:", len(oof))
print("Speaker-disjoint CV: PASS")
print("Ridge alpha:", ALPHA)

print("\n=== MODEL COMPARISON ===")
for name, m in comparison.items():
    print(f"{name}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}")

print("\n=== C1 H1-H5 METRICS ===")
for h, m in sorted(hyp_metrics.items()):
    print(f"{h}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}")

print("\n=== C1 PREDICTION STABILITY ===")
for key, value in stability.items():
    print(f"{key}: {value}")

print("\n=== C1 CRITICAL ASR CASE ===")
print("Audio: 030210164")
print("Hypothesis: H2")
print("Human fluency:", float(critical["fluency"].iloc[0]))
print("C1 prediction:", float(critical["c1_prediction"].iloc[0]))

print("\n=== C1 PER-FOLD METRICS ===")
print(pd.DataFrame(fold_rows).round(4).to_string(index=False))

print("\nPrediction CSV:", PRED)
print("SHA256:", sha256(PRED))
print("Checkpoint updated.")



# ========== ORIGINAL KAGGLE CELL 111 ==========
# %%

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

P = Path("/kaggle/working/acoustic_linguistic_research")
DATA = P / "data/modeling/modeling_dataset_0001-0400.csv"
SPLIT = P / "data/splits/speaker_groupkfold5_frozen.csv"
DESIGN = P / "configs/fluency_feature_design_v1.json"
CV = P / "configs/speaker_disjoint_cv_v1.json"
B1 = P / "results/step05/b1_transcript_ridge_oof_predictions.csv"
CHECK = P / "checkpoints/research_progress.json"

OUTDIR = P / "results/step05"
PRED = OUTDIR / "b2_prompt_aware_ridge_oof_predictions.csv"
FOLDS = OUTDIR / "b2_prompt_aware_ridge_fold_metrics.csv"
AUDIO = OUTDIR / "b2_prompt_aware_stability_per_audio.csv"
REPORT = P / "reports/step05_b2_prompt_aware_ridge_v1.json"

EXPECTED = {
    DATA: "7d20428a14d9e00b88f1a20b36b7147bf5b98234045a093172fef76b68276746",
    SPLIT: "5f87cdc1c2f69922bcb542ee9516f5f89168bf68402fbd2449b28fcf61a375e7",
    DESIGN: "626b0050f58f35fafa4f2efcd1888fda2a195454e544c4cb094cd50e796a1b83",
    CV: "302c7ea0086a54d644a4ebc0b8a5d96d63575a3896b5a29d76dba7e26dacd472",
    B1: "375ac49b0d5b565adf65d04407d6afc1816f64f5ba108afef8d22458383807dc"
}

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for path, expected in EXPECTED.items():
    assert path.is_file(), f"Missing: {path}"
    assert sha256(path) == expected, f"SHA mismatch: {path}"

design = json.loads(DESIGN.read_text(encoding="utf-8"))
cv = json.loads(CV.read_text(encoding="utf-8"))

assert cv["status"] == "frozen"
assert cv["speaker_disjoint"]
assert design["models"]["B2"]["feature_groups"] == [
    "linguistic", "prompt"
]

FEATURES = (
    design["linguistic_feature_names"]
    + design["prompt_feature_names"]
)
assert len(FEATURES) == 9
assert len(set(FEATURES)) == 9

KEYS = ["pilot_index", "audio_id", "speaker",
        "hypothesis_id", "fluency"]
AUDIO_KEYS = ["pilot_index", "audio_id", "speaker", "fluency"]

data = pd.read_csv(
    DATA, dtype={"audio_id": "string", "speaker": "string"}
)
split = pd.read_csv(
    SPLIT, dtype={"audio_id": "string", "speaker": "string"}
)
b1 = pd.read_csv(
    B1, dtype={"audio_id": "string", "speaker": "string"}
)

assert len(data) == len(b1) == 2000
assert len(split) == 400
assert not data.duplicated(KEYS).any()
assert not b1.duplicated(KEYS).any()
assert not split.duplicated(AUDIO_KEYS).any()

df = data.merge(
    split[AUDIO_KEYS + ["validation_fold"]],
    on=AUDIO_KEYS,
    how="inner",
    validate="many_to_one"
)

assert len(df) == 2000
assert df["audio_id"].nunique() == 400
assert df.groupby("speaker")["validation_fold"].nunique().eq(1).all()
assert df.groupby("audio_id")["hypothesis_id"].nunique().eq(5).all()

pred_parts = []
fold_rows = []
ALPHA = 1.0

for fold in range(1, 6):
    train = df.loc[df["validation_fold"] != fold]
    val = df.loc[df["validation_fold"] == fold]

    assert len(train) == 1600
    assert len(val) == 400
    assert train["speaker"].nunique() == 16
    assert val["speaker"].nunique() == 4
    assert set(train["speaker"]).isdisjoint(set(val["speaker"]))
    assert set(train["audio_id"]).isdisjoint(set(val["audio_id"]))

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=ALPHA))
    ])

    model.fit(
        train[FEATURES],
        train["fluency"].to_numpy(dtype=float)
    )

    prediction = model.predict(val[FEATURES])
    assert np.isfinite(prediction).all()

    part = val[KEYS + ["validation_fold"]].copy()
    part["b2_prediction"] = prediction
    pred_parts.append(part)

    err = prediction - val["fluency"].to_numpy(dtype=float)
    fold_rows.append({
        "fold": fold,
        "train_audios": 320,
        "validation_audios": 80,
        "validation_hypotheses": 400,
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2)))
    })

oof = pd.concat(pred_parts, ignore_index=True).sort_values(
    ["pilot_index", "hypothesis_id"]
).reset_index(drop=True)

assert len(oof) == 2000
assert not oof.duplicated(KEYS).any()

comp = oof.merge(
    b1[KEYS + ["validation_fold", "b1_prediction"]],
    on=KEYS + ["validation_fold"],
    how="inner",
    validate="one_to_one"
)
assert len(comp) == 2000

def metrics(frame, col):
    err = frame[col].to_numpy(dtype=float) - frame[
        "fluency"
    ].to_numpy(dtype=float)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2)))
    }

comparison = {
    "B1": metrics(comp, "b1_prediction"),
    "B2": metrics(comp, "b2_prediction")
}

hypothesis_metrics = {
    h: metrics(group, "b2_prediction")
    for h, group in oof.groupby("hypothesis_id")
}

audio = oof.groupby(
    AUDIO_KEYS + ["validation_fold"], as_index=False
).agg(
    prediction_min=("b2_prediction", "min"),
    prediction_max=("b2_prediction", "max"),
    prediction_sd=("b2_prediction", "std"),
    hypothesis_count=("hypothesis_id", "count")
)

audio["prediction_range"] = (
    audio["prediction_max"] - audio["prediction_min"]
)

assert len(audio) == 400
assert audio["hypothesis_count"].eq(5).all()

stability = {
    "mean_range": float(audio["prediction_range"].mean()),
    "median_range": float(audio["prediction_range"].median()),
    "p95_range": float(audio["prediction_range"].quantile(0.95)),
    "max_range": float(audio["prediction_range"].max()),
    "variable_audio_count": int(
        (audio["prediction_range"] > 1e-10).sum()
    )
}

critical = oof.loc[
    oof["audio_id"].eq("030210164") &
    oof["hypothesis_id"].eq("H2")
]
assert len(critical) == 1

without_critical = comp.loc[
    ~(comp["audio_id"].eq("030210164") &
      comp["hypothesis_id"].eq("H2"))
]

sensitivity = {
    "B1_without_one_H2": metrics(
        without_critical, "b1_prediction"
    ),
    "B2_without_one_H2": metrics(
        without_critical, "b2_prediction"
    )
}

OUTDIR.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

def save_csv(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8")
    if path.exists():
        assert sha256(path) == sha256(tmp), (
            f"Existing result differs: {path}"
        )
        tmp.unlink()
    else:
        tmp.replace(path)

save_csv(oof, PRED)
save_csv(pd.DataFrame(fold_rows), FOLDS)
save_csv(audio.sort_values("pilot_index"), AUDIO)

report = {
    "status": "PASS",
    "model": "B2_prompt_aware_ridge",
    "alpha": ALPHA,
    "hyperparameter_tuning": False,
    "features": FEATURES,
    "feature_count": 9,
    "speaker_disjoint_folds": 5,
    "unique_audios": 400,
    "OOF_predictions": 2000,
    "comparison": comparison,
    "per_hypothesis_metrics": hypothesis_metrics,
    "stability": stability,
    "critical_H2_prediction": float(
        critical["b2_prediction"].iloc[0]
    ),
    "sensitivity": sensitivity,
    "source_data_sha256": sha256(DATA),
    "frozen_split_sha256": sha256(SPLIT),
    "predictions_sha256": sha256(PRED),
    "fold_metrics_sha256": sha256(FOLDS),
    "audio_stability_sha256": sha256(AUDIO),
    "notes": (
        "Fixed-alpha Ridge, no clipping or tuning. "
        "Training-only imputation and scaling. "
        "Prompt disagreement is not ground-truth WER/CER. "
        "All 400 audios remain in primary evaluation."
    )
}

if REPORT.exists():
    old = json.loads(REPORT.read_text(encoding="utf-8"))
    assert old == report, "Existing report differs"
else:
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    tmp.replace(REPORT)

progress = json.loads(CHECK.read_text(encoding="utf-8"))
progress["step5_b2_prompt_aware_ridge"] = "qc_pass"
progress["step5_b2_predictions_sha256"] = sha256(PRED)
progress["next_task"] = (
    "Review B2 prompt-aware accuracy and instability; "
    "then evaluate C2 prompt-aware fusion"
)
progress["updated_at"] = datetime.now(timezone.utc).isoformat()

CHECK.write_text(
    json.dumps(progress, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("=== STEP 5.8 B2 PROMPT-AWARE RIDGE ===")
print("Status: PASS")
print("Features:", len(FEATURES))
print("OOF predictions:", len(oof))
print("Speaker-disjoint folds: 5")
print("Alpha:", ALPHA)

print("\n=== B1 VS B2 ACCURACY ===")
for name, values in comparison.items():
    print(
        f"{name}: MAE={values['MAE']:.4f}, "
        f"RMSE={values['RMSE']:.4f}"
    )

print("\n=== B2 H1-H5 METRICS ===")
for h, values in sorted(hypothesis_metrics.items()):
    print(
        f"{h}: MAE={values['MAE']:.4f}, "
        f"RMSE={values['RMSE']:.4f}"
    )

print("\n=== B2 PREDICTION STABILITY ===")
for key, value in stability.items():
    print(f"{key}: {value}")

print("\n=== B2 CRITICAL H2 CASE ===")
print("Audio: 030210164")
print("Human fluency:", float(critical["fluency"].iloc[0]))
print("B2 prediction:", float(critical["b2_prediction"].iloc[0]))

print("\n=== SENSITIVITY: WITHOUT ONE H2 PREDICTION ===")
for name, values in sensitivity.items():
    print(
        f"{name}: MAE={values['MAE']:.4f}, "
        f"RMSE={values['RMSE']:.4f}"
    )

print("\n=== B2 PER-FOLD METRICS ===")
print(pd.DataFrame(fold_rows).round(4).to_string(index=False))

print("\nPrediction CSV:", PRED)
print("SHA256:", sha256(PRED))
print("Report:", REPORT)
print("Checkpoint updated.")



# ========== ORIGINAL KAGGLE CELL 112 ==========
# %%

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

P = Path("/kaggle/working/acoustic_linguistic_research")
DATA = P / "data/modeling/modeling_dataset_0001-0400.csv"
SPLIT = P / "data/splits/speaker_groupkfold5_frozen.csv"
DESIGN = P / "configs/fluency_feature_design_v1.json"
CV = P / "configs/speaker_disjoint_cv_v1.json"
C1 = P / "results/step05/c1_fusion_ridge_oof_predictions.csv"
B2 = P / "results/step05/b2_prompt_aware_ridge_oof_predictions.csv"
CHECK = P / "checkpoints/research_progress.json"

OUTDIR = P / "results/step05"
PRED = OUTDIR / "c2_prompt_aware_fusion_ridge_oof_predictions.csv"
FOLDS = OUTDIR / "c2_prompt_aware_fusion_fold_metrics.csv"
AUDIO = OUTDIR / "c2_prompt_aware_fusion_stability_per_audio.csv"
REPORT = P / "reports/step05_c2_prompt_aware_fusion_v1.json"

EXPECTED = {
    DATA: "7d20428a14d9e00b88f1a20b36b7147bf5b98234045a093172fef76b68276746",
    SPLIT: "5f87cdc1c2f69922bcb542ee9516f5f89168bf68402fbd2449b28fcf61a375e7",
    DESIGN: "626b0050f58f35fafa4f2efcd1888fda2a195454e544c4cb094cd50e796a1b83",
    CV: "302c7ea0086a54d644a4ebc0b8a5d96d63575a3896b5a29d76dba7e26dacd472",
    C1: "43aa2b8d73c60e44ee4ee2f996f5b243ae2a09c5b9a78f909b2a56ba7306669e",
    B2: "ba212868934dbe4d92234f46ccb5f7d7e66a56c781d74aeb24b310b722cfb189"
}

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for path, expected in EXPECTED.items():
    assert path.exists(), f"Missing file: {path}"
    assert sha256(path) == expected, f"SHA mismatch: {path}"

design = json.loads(DESIGN.read_text(encoding="utf-8"))
cv = json.loads(CV.read_text(encoding="utf-8"))

assert cv["status"] == "frozen"
assert cv["speaker_disjoint"] is True
assert design["models"]["C2"]["feature_groups"] == [
    "acoustic", "linguistic", "prompt"
]

def read(path):
    return pd.read_csv(
        path,
        dtype={"audio_id": "string", "speaker": "string"}
    )

data = read(DATA)
split = read(SPLIT)
c1 = read(C1)
b2 = read(B2)

IDS = ["pilot_index", "audio_id", "speaker"]
KEYS = IDS + ["hypothesis_id"]
FULL_KEYS = KEYS + ["fluency", "validation_fold"]

LING = design["linguistic_feature_names"]
PROMPT = design["prompt_feature_names"]
ACOUSTIC = [
    x for x in data.columns
    if x not in KEYS + ["fluency"] + LING + PROMPT
]
FEATURES = ACOUSTIC + LING + PROMPT

assert len(ACOUSTIC) == 14
assert len(LING) == 7
assert len(PROMPT) == 2
assert len(FEATURES) == 23
assert len(set(FEATURES)) == 23
assert "fluency" not in FEATURES

assert len(data) == 2000
assert len(split) == 400
assert len(c1) == len(b2) == 2000

assert not data.duplicated(KEYS).any()
assert not split.duplicated(IDS).any()

df = data.merge(
    split[IDS + ["fluency", "validation_fold"]],
    on=IDS + ["fluency"],
    how="inner",
    validate="many_to_one"
)

assert len(df) == 2000
assert df["audio_id"].nunique() == 400
assert df["speaker"].nunique() == 20
assert df.groupby("speaker")["validation_fold"].nunique().eq(1).all()
assert df.groupby("audio_id")["hypothesis_id"].nunique().eq(5).all()

ALPHA = 1.0
parts = []
fold_rows = []

for fold in range(1, 6):
    train = df[df["validation_fold"] != fold]
    val = df[df["validation_fold"] == fold]

    assert len(train) == 1600
    assert len(val) == 400
    assert train["audio_id"].nunique() == 320
    assert val["audio_id"].nunique() == 80
    assert train["speaker"].nunique() == 16
    assert val["speaker"].nunique() == 4

    assert set(train["speaker"]).isdisjoint(set(val["speaker"]))
    assert set(train["audio_id"]).isdisjoint(set(val["audio_id"]))

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=ALPHA))
    ])

    model.fit(
        train[FEATURES],
        train["fluency"].to_numpy(dtype=float)
    )

    yhat = model.predict(val[FEATURES])
    assert np.isfinite(yhat).all()

    result = val[FULL_KEYS].copy()
    result["c2_prediction"] = yhat
    parts.append(result)

    err = yhat - val["fluency"].to_numpy(dtype=float)

    fold_rows.append({
        "fold": fold,
        "train_speakers": 16,
        "validation_speakers": 4,
        "train_unique_audios": 320,
        "validation_unique_audios": 80,
        "validation_hypotheses": 400,
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2)))
    })

oof = pd.concat(parts, ignore_index=True)
oof = oof.sort_values(
    ["pilot_index", "hypothesis_id"]
).reset_index(drop=True)

assert len(oof) == 2000
assert not oof.duplicated(KEYS).any()

# Strictly align with previously stored C1 and B2 predictions.
comparison = oof.merge(
    c1[FULL_KEYS + ["c1_prediction"]],
    on=FULL_KEYS,
    how="inner",
    validate="one_to_one"
).merge(
    b2[FULL_KEYS + ["b2_prediction"]],
    on=FULL_KEYS,
    how="inner",
    validate="one_to_one"
)

assert len(comparison) == 2000

def metrics(frame, prediction_column):
    error = (
        frame[prediction_column].to_numpy(dtype=float)
        - frame["fluency"].to_numpy(dtype=float)
    )
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2)))
    }

model_comparison = {
    "B2": metrics(comparison, "b2_prediction"),
    "C1": metrics(comparison, "c1_prediction"),
    "C2": metrics(comparison, "c2_prediction")
}

hypothesis_metrics = {
    hid: metrics(group, "c2_prediction")
    for hid, group in oof.groupby("hypothesis_id")
}

audio = oof.groupby(
    IDS + ["fluency", "validation_fold"],
    as_index=False
).agg(
    prediction_min=("c2_prediction", "min"),
    prediction_max=("c2_prediction", "max"),
    prediction_sd=("c2_prediction", "std"),
    prediction_mean=("c2_prediction", "mean"),
    hypothesis_count=("hypothesis_id", "count")
)

audio["prediction_range"] = (
    audio["prediction_max"] - audio["prediction_min"]
)

assert len(audio) == 400
assert audio["hypothesis_count"].eq(5).all()

stability = {
    "mean_range": float(audio["prediction_range"].mean()),
    "median_range": float(audio["prediction_range"].median()),
    "p95_range": float(audio["prediction_range"].quantile(0.95)),
    "max_range": float(audio["prediction_range"].max()),
    "mean_prediction_sd": float(audio["prediction_sd"].mean()),
    "variable_audio_count": int(
        (audio["prediction_range"] > 1e-10).sum()
    )
}

CRITICAL_ID = "030210164"

critical_mask = (
    comparison["audio_id"].eq(CRITICAL_ID)
    & comparison["hypothesis_id"].eq("H2")
)

assert critical_mask.sum() == 1
critical = comparison.loc[critical_mask].iloc[0]

without_one = comparison.loc[~critical_mask]
without_audio = comparison[
    comparison["audio_id"] != CRITICAL_ID
]

sensitivity = {
    "without_one_H2": {
        name: metrics(without_one, f"{name.lower()}_prediction")
        for name in ["B2", "C1", "C2"]
    },
    "without_entire_audio": {
        name: metrics(without_audio, f"{name.lower()}_prediction")
        for name in ["B2", "C1", "C2"]
    }
}

c2_error = (
    comparison["c2_prediction"] - comparison["fluency"]
)
c2_critical_sse_share = float(
    (critical["c2_prediction"] - critical["fluency"]) ** 2
    / np.square(c2_error).sum()
)

OUTDIR.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

def save_csv(frame, path):
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8")

    if path.exists():
        assert sha256(path) == sha256(temp), (
            f"Existing result differs: {path}"
        )
        temp.unlink()
    else:
        temp.replace(path)

save_csv(oof, PRED)
save_csv(pd.DataFrame(fold_rows), FOLDS)
save_csv(audio.sort_values("pilot_index"), AUDIO)

report = {
    "status": "PASS",
    "model": "C2_acoustic_linguistic_prompt_aware_ridge",
    "feature_count": 23,
    "acoustic_features": 14,
    "linguistic_features": 7,
    "prompt_features": 2,
    "alpha": ALPHA,
    "hyperparameter_tuning_performed": False,
    "prediction_clipping_performed": False,
    "speaker_disjoint_cv": "frozen_5fold",
    "unique_audio_count": 400,
    "OOF_predictions": 2000,
    "comparison": model_comparison,
    "per_hypothesis_metrics": hypothesis_metrics,
    "stability": stability,
    "critical_audio_H2": {
        "human_fluency": float(critical["fluency"]),
        "B2_prediction": float(critical["b2_prediction"]),
        "C1_prediction": float(critical["c1_prediction"]),
        "C2_prediction": float(critical["c2_prediction"]),
        "C2_squared_error_share": c2_critical_sse_share
    },
    "posthoc_sensitivity": sensitivity,
    "source_dataset_sha256": sha256(DATA),
    "frozen_split_sha256": sha256(SPLIT),
    "predictions_sha256": sha256(PRED),
    "fold_metrics_sha256": sha256(FOLDS),
    "audio_stability_sha256": sha256(AUDIO),
    "notes": (
        "Unclipped fixed-alpha Ridge predictions. "
        "Train-fold-only median imputation and scaling. "
        "Each audio contributes exactly five hypotheses. "
        "Prompt-ASR disagreement is not verified WER/CER. "
        "No primary-analysis observations excluded."
    )
}

if REPORT.exists():
    saved = json.loads(REPORT.read_text(encoding="utf-8"))
    assert saved == report, (
        "Existing C2 report differs; original preserved."
    )
else:
    temp = REPORT.with_suffix(".tmp")
    temp.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    temp.replace(REPORT)

progress = json.loads(CHECK.read_text(encoding="utf-8"))
progress["step5_c2_prompt_aware_fusion"] = "qc_pass"
progress["step5_c2_predictions_sha256"] = sha256(PRED)
progress["next_task"] = (
    "Compare five fixed-alpha models and estimate "
    "speaker-clustered uncertainty for paired differences"
)
progress["updated_at"] = datetime.now(
    timezone.utc
).isoformat()

CHECK.write_text(
    json.dumps(progress, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("=== STEP 5.9 C2 PROMPT-AWARE FUSION ===")
print("Status: PASS")
print("Features:", len(FEATURES))
print("Unique audios:", 400)
print("OOF predictions:", len(oof))
print("Speaker-disjoint CV: PASS")
print("Ridge alpha:", ALPHA)

print("\n=== B2 VS C1 VS C2 ACCURACY ===")
for name, m in model_comparison.items():
    print(
        f"{name}: MAE={m['MAE']:.4f}, "
        f"RMSE={m['RMSE']:.4f}"
    )

print("\n=== C2 H1-H5 METRICS ===")
for hid, m in sorted(hypothesis_metrics.items()):
    print(
        f"{hid}: MAE={m['MAE']:.4f}, "
        f"RMSE={m['RMSE']:.4f}"
    )

print("\n=== C2 PREDICTION STABILITY ===")
for key, value in stability.items():
    print(f"{key}: {value:.4f}")

print("\n=== C2 CRITICAL H2 CASE ===")
for key, value in report["critical_audio_H2"].items():
    print(f"{key}: {value}")

print("\n=== SENSITIVITY WITHOUT ONE H2 ===")
for name, m in sensitivity["without_one_H2"].items():
    print(
        f"{name}: MAE={m['MAE']:.4f}, "
        f"RMSE={m['RMSE']:.4f}"
    )

print("\n=== C2 PER-FOLD METRICS ===")
print(pd.DataFrame(fold_rows).round(4).to_string(index=False))

print("\nPrediction CSV:", PRED)
print("SHA256:", sha256(PRED))
print("Report:", REPORT)
print("Checkpoint updated.")


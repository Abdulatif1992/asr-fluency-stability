#!/usr/bin/env python3
"""Source-preserving extract of speaker bootstrap, anomaly and variability analyses.

Original Kaggle source cells: [113, 115, 117, 118].
Requires project workspace restored at /kaggle/working/acoustic_linguistic_research.
This is a reorganized source extract, not an independently revalidated execution.
"""


# ========== ORIGINAL KAGGLE CELL 113 ==========
# %%

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

P = Path("/kaggle/working/acoustic_linguistic_research")
DIR = P / "results/step05"
CHECK = P / "checkpoints/research_progress.json"

SOURCES = {
    "A": DIR / "baseline_acoustic_oof_predictions.csv",
    "B1": DIR / "b1_transcript_ridge_oof_predictions.csv",
    "B2": DIR / "b2_prompt_aware_ridge_oof_predictions.csv",
    "C1": DIR / "c1_fusion_ridge_oof_predictions.csv",
    "C2": DIR / "c2_prompt_aware_fusion_ridge_oof_predictions.csv"
}

EXPECTED = {
    "A": "48407b4eac0f213e4ebf39d5b88aa131178e1aca3597cd41011cef168eac1595",
    "B1": "375ac49b0d5b565adf65d04407d6afc1816f64f5ba108afef8d22458383807dc",
    "B2": "ba212868934dbe4d92234f46ccb5f7d7e66a56c781d74aeb24b310b722cfb189",
    "C1": "43aa2b8d73c60e44ee4ee2f996f5b243ae2a09c5b9a78f909b2a56ba7306669e",
    "C2": "2592e8d28c656db43280d6eea30d578cd488d75f0382e3ed083abf5e2ab86df9"
}

PRED_COLUMNS = {
    "A": "acoustic_ridge_prediction",
    "B1": "b1_prediction",
    "B2": "b2_prediction",
    "C1": "c1_prediction",
    "C2": "c2_prediction"
}

OUT = DIR / "speaker_cluster_bootstrap_10000_ci.csv"
REPORT = P / "reports/step06_speaker_cluster_bootstrap_v1.json"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for model, path in SOURCES.items():
    assert path.is_file(), f"Missing: {path}"
    assert sha256(path) == EXPECTED[model], (
        f"SHA mismatch for {model}"
    )

def read(path):
    return pd.read_csv(
        path,
        dtype={"audio_id": "string", "speaker": "string"}
    )

frames = {name: read(path) for name, path in SOURCES.items()}

KEYS = [
    "pilot_index", "audio_id", "speaker",
    "fluency", "validation_fold"
]
HKEYS = KEYS + ["hypothesis_id"]

assert len(frames["A"]) == 400

for name in ["B1", "B2", "C1", "C2"]:
    assert len(frames[name]) == 2000
    assert not frames[name].duplicated(HKEYS).any()

base = frames["B1"][
    HKEYS + [PRED_COLUMNS["B1"]]
].copy()

for name in ["B2", "C1", "C2"]:
    base = base.merge(
        frames[name][HKEYS + [PRED_COLUMNS[name]]],
        on=HKEYS,
        how="inner",
        validate="one_to_one"
    )

base = base.merge(
    frames["A"][KEYS + [PRED_COLUMNS["A"]]],
    on=KEYS,
    how="inner",
    validate="many_to_one"
)

assert len(base) == 2000
assert base["audio_id"].nunique() == 400
assert base["speaker"].nunique() == 20
assert base.groupby("audio_id")["hypothesis_id"].nunique().eq(5).all()
assert base.groupby("speaker")["validation_fold"].nunique().eq(1).all()
assert base.groupby("speaker")["audio_id"].nunique().eq(20).all()

# One row per audio: average the five hypothesis errors
# BEFORE speaker-level resampling.


# Explicit error columns avoid ambiguity in aggregation.
for name, col in PRED_COLUMNS.items():
    base[f"{name}_ae"] = (
        base[col] - base["fluency"]
    ).abs()
    base[f"{name}_se"] = (
        base[col] - base["fluency"]
    ) ** 2

agg = {}
for name in PRED_COLUMNS:
    agg[f"{name}_ae"] = (f"{name}_ae", "mean")
    agg[f"{name}_se"] = (f"{name}_se", "mean")
    agg[f"{name}_min"] = (PRED_COLUMNS[name], "min")
    agg[f"{name}_max"] = (PRED_COLUMNS[name], "max")

audio = base.groupby(
    KEYS, as_index=False
).agg(**agg)

assert len(audio) == 400
assert audio["audio_id"].is_unique

for name in PRED_COLUMNS:
    audio[f"{name}_range"] = (
        audio[f"{name}_max"] - audio[f"{name}_min"]
    )

# A must be invariant to ASR hypotheses.
assert np.allclose(audio["A_range"], 0, atol=1e-12)

MODELS = ["A", "B1", "B2", "C1", "C2"]
SPEAKERS = sorted(audio["speaker"].unique())
assert len(SPEAKERS) == 20

# Each speaker has 20 audios, so equal speaker weights
# are also equal audio weights in this pilot.
speaker = audio.groupby("speaker").agg(
    **{
        f"{name}_{metric}": (f"{name}_{metric}", "mean")
        for name in MODELS
        for metric in ["ae", "se", "range"]
    }
).reindex(SPEAKERS)

assert speaker.shape[0] == 20
assert speaker.notna().all().all()

N_BOOT = 10000
SEED = 20261026
rng = np.random.default_rng(SEED)

# Matrix: iterations x sampled speakers.
indices = rng.integers(
    low=0,
    high=20,
    size=(N_BOOT, 20)
)

bootstrap = {}
observed = {}

for name in MODELS:
    for metric in ["ae", "se", "range"]:
        values = speaker[f"{name}_{metric}"].to_numpy(
            dtype=float
        )

        avg = values[indices].mean(axis=1)

        if metric == "se":
            bootstrap[(name, "RMSE")] = np.sqrt(avg)
            observed[(name, "RMSE")] = float(
                np.sqrt(values.mean())
            )
        elif metric == "ae":
            bootstrap[(name, "MAE")] = avg
            observed[(name, "MAE")] = float(values.mean())
        else:
            bootstrap[(name, "mean_range")] = avg
            observed[(name, "mean_range")] = float(
                values.mean()
            )

# Verify observed metrics agree with previous reports.
expected_metrics = {
    "A": (0.9399, 1.2066),
    "B1": (1.2426, 1.9972),
    "B2": (1.1761, 1.9518),
    "C1": (0.8972, 1.3735),
    "C2": (0.7782, 1.2461)
}

for name, (mae, rmse) in expected_metrics.items():
    assert np.isclose(observed[(name, "MAE")], mae, atol=0.0001)
    assert np.isclose(observed[(name, "RMSE")], rmse, atol=0.0001)

def ci(values):
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975))
    ]

rows = []

for name in MODELS:
    for metric in ["MAE", "RMSE", "mean_range"]:
        low, high = ci(bootstrap[(name, metric)])
        rows.append({
            "comparison": name,
            "metric": metric,
            "observed": observed[(name, metric)],
            "ci_2_5": low,
            "ci_97_5": high
        })

# Paired differences are calculated within the SAME
# bootstrap speaker draws for both models.
PAIRINGS = [
    ("C1", "A"),
    ("C2", "A"),
    ("C2", "C1"),
    ("B2", "B1"),
    ("C1", "B1"),
    ("C2", "B2")
]

for model, reference in PAIRINGS:
    for metric in ["MAE", "RMSE", "mean_range"]:
        diff = (
            bootstrap[(model, metric)]
            - bootstrap[(reference, metric)]
        )
        low, high = ci(diff)

        rows.append({
            "comparison": f"{model}_minus_{reference}",
            "metric": metric,
            "observed": (
                observed[(model, metric)]
                - observed[(reference, metric)]
            ),
            "ci_2_5": low,
            "ci_97_5": high
        })

results = pd.DataFrame(rows)

assert len(results) == 33
assert results[["observed", "ci_2_5", "ci_97_5"]].notna().all().all()

OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

tmp = OUT.with_suffix(".tmp")
results.to_csv(tmp, index=False, encoding="utf-8")

if OUT.exists():
    assert sha256(OUT) == sha256(tmp), (
        "Existing bootstrap CSV differs; preserved."
    )
    tmp.unlink()
else:
    tmp.replace(OUT)

report = {
    "status": "PASS",
    "method": "paired_speaker_cluster_percentile_bootstrap",
    "bootstrap_iterations": N_BOOT,
    "random_seed": SEED,
    "confidence_level": 0.95,
    "resampling_unit": "speaker",
    "unique_speakers": 20,
    "unique_audios": 400,
    "hypotheses_per_audio": 5,
    "source_split": "frozen_5fold_speaker_disjoint",
    "all_oof_predictions_unchanged": True,
    "all_outliers_retained": True,
    "refitting_during_bootstrap": False,
    "models": MODELS,
    "comparison_pairs": [
        f"{m}_minus_{r}" for m, r in PAIRINGS
    ],
    "results_sha256": sha256(OUT),
    "source_sha256": {
        name: sha256(path)
        for name, path in SOURCES.items()
    },
    "interpretation": (
        "Paired speaker-cluster bootstrap CIs conditional "
        "on previously fitted OOF models; not nested-CV "
        "or refitting uncertainty. "
        "Negative error difference favors the first model. "
        "Bootstrap intervals are descriptive, especially "
        "with only 20 speaker clusters."
    )
}

if REPORT.exists():
    old = json.loads(REPORT.read_text(encoding="utf-8"))
    assert old == report, (
        "Existing bootstrap report differs; preserved."
    )
else:
    temp = REPORT.with_suffix(".tmp")
    temp.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    temp.replace(REPORT)

progress = json.loads(CHECK.read_text(encoding="utf-8"))
progress["step6_speaker_cluster_bootstrap"] = "qc_pass"
progress["step6_bootstrap_results_sha256"] = sha256(OUT)
progress["next_task"] = (
    "Review speaker-cluster bootstrap uncertainty; "
    "perform robustness and final model comparison"
)
progress["updated_at"] = datetime.now(
    timezone.utc
).isoformat()

CHECK.write_text(
    json.dumps(progress, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("=== STEP 6.1 SPEAKER-CLUSTER BOOTSTRAP ===")
print("Status: PASS")
print("Bootstrap iterations:", N_BOOT)
print("Speaker clusters:", len(SPEAKERS))
print("Confidence interval: 95%")
print("Refitting: NO")
print("All 400 audios retained: YES")

print("\n=== MODEL ACCURACY AND 95% CI ===")
print(
    results[
        results["comparison"].isin(MODELS) &
        results["metric"].isin(["MAE", "RMSE"])
    ].round(4).to_string(index=False)
)

print("\n=== PAIRED MODEL DIFFERENCES AND 95% CI ===")
print(
    results[
        results["comparison"].str.contains("_minus_") &
        results["metric"].isin(["MAE", "RMSE"])
    ].round(4).to_string(index=False)
)

print("\n=== PREDICTION STABILITY AND 95% CI ===")
print(
    results[results["metric"] == "mean_range"]
    .round(4).to_string(index=False)
)

print("\nCSV:", OUT)
print("SHA256:", sha256(OUT))
print("Report:", REPORT)
print("Checkpoint updated.")



# ========== ORIGINAL KAGGLE CELL 115 ==========
# %%

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

P = Path("/kaggle/working/acoustic_linguistic_research")
R = P / "results/step05"
CHECK = P / "checkpoints/research_progress.json"

SOURCES = {
    "A": ("baseline_acoustic_oof_predictions.csv",
          "48407b4eac0f213e4ebf39d5b88aa131178e1aca3597cd41011cef168eac1595",
          "acoustic_ridge_prediction"),
    "B1": ("b1_transcript_ridge_oof_predictions.csv",
           "375ac49b0d5b565adf65d04407d6afc1816f64f5ba108afef8d22458383807dc",
           "b1_prediction"),
    "B2": ("b2_prompt_aware_ridge_oof_predictions.csv",
           "ba212868934dbe4d92234f46ccb5f7d7e66a56c781d74aeb24b310b722cfb189",
           "b2_prediction"),
    "C1": ("c1_fusion_ridge_oof_predictions.csv",
           "43aa2b8d73c60e44ee4ee2f996f5b243ae2a09c5b9a78f909b2a56ba7306669e",
           "c1_prediction"),
    "C2": ("c2_prompt_aware_fusion_ridge_oof_predictions.csv",
           "2592e8d28c656db43280d6eea30d578cd488d75f0382e3ed083abf5e2ab86df9",
           "c2_prediction")
}

AUDIT = P / "reports/step04_anomaly_classification_v1.csv"
AUDIT_SHA = "80122fe23528b9a221836ac1f738876a1ae503baa8dd3354cce329952e8952b0"

OUT = R / "step06_anomaly_sensitivity_400_399_396.csv"
PAIRED = R / "step06_anomaly_sensitivity_paired_differences.csv"
REPORT = P / "reports/step06_anomaly_sensitivity_v1.json"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return pd.read_csv(
        path,
        dtype={"audio_id": "string", "speaker": "string"}
    )

assert sha256(AUDIT) == AUDIT_SHA

audit = read(AUDIT)

EXPECTED_CANDIDATES = {
    "030210164", "096340009",
    "096340012", "096340017"
}

# Verify the four previously declared anomaly candidates.
possible_flags = [
    c for c in audit.columns
    if "candidate" in c.lower()
    and ("sensitivity" in c.lower() or "anomaly" in c.lower())
]

if "sensitivity_analysis_candidate" in audit.columns:
    flag = "sensitivity_analysis_candidate"
elif len(possible_flags) == 1:
    flag = possible_flags[0]
else:
    raise ValueError(
        f"Review audit flag column: {audit.columns.tolist()}"
    )

values = audit[flag].astype(str).str.strip().str.lower()

candidate_ids = set(
    audit.loc[
        values.isin(["true", "1", "yes"]),
        "audio_id"
    ].astype(str)
)

assert candidate_ids == EXPECTED_CANDIDATES, (
    f"Audit candidates mismatch: {candidate_ids}"
)

KEYS = [
    "pilot_index", "audio_id", "speaker",
    "fluency", "validation_fold"
]
HKEYS = KEYS + ["hypothesis_id"]

frames = {}

for name, (filename, expected, col) in SOURCES.items():
    path = R / filename
    assert sha256(path) == expected, f"SHA mismatch: {name}"
    frames[name] = read(path)

base = frames["B1"][
    HKEYS + [SOURCES["B1"][2]]
].copy()

for name in ["B2", "C1", "C2"]:
    base = base.merge(
        frames[name][HKEYS + [SOURCES[name][2]]],
        on=HKEYS,
        validate="one_to_one"
    )

base = base.merge(
    frames["A"][KEYS + [SOURCES["A"][2]]],
    on=KEYS,
    validate="many_to_one"
)

assert len(base) == 2000
assert base["audio_id"].nunique() == 400
assert base.groupby("audio_id")["hypothesis_id"].nunique().eq(5).all()

MODELS = list(SOURCES)
aggregations = {}

for name in MODELS:
    pred_col = SOURCES[name][2]

    base[f"{name}_ae"] = (
        base[pred_col] - base["fluency"]
    ).abs()

    base[f"{name}_se"] = (
        base[pred_col] - base["fluency"]
    ) ** 2

    aggregations[f"{name}_ae"] = (f"{name}_ae", "mean")
    aggregations[f"{name}_se"] = (f"{name}_se", "mean")
    aggregations[f"{name}_min"] = (pred_col, "min")
    aggregations[f"{name}_max"] = (pred_col, "max")

audio = base.groupby(
    KEYS, as_index=False
).agg(**aggregations)

assert len(audio) == 400
assert audio["audio_id"].is_unique

for name in MODELS:
    audio[f"{name}_range"] = (
        audio[f"{name}_max"] - audio[f"{name}_min"]
    )

assert np.allclose(audio["A_range"], 0, atol=1e-12)

CONDITIONS = {
    "primary_all_400": set(),
    "diagnostic_without_critical_399": {"030210164"},
    "diagnostic_without_4_candidates_396": candidate_ids
}

rows = []
pairs = []

PAIRINGS = [
    ("C2", "A"),
    ("C2", "C1"),
    ("B2", "B1")
]

for condition, excluded in CONDITIONS.items():
    sub = audio.loc[
        ~audio["audio_id"].isin(excluded)
    ].copy()

    assert len(sub) == 400 - len(excluded)

    for name in MODELS:
        rows.append({
            "condition": condition,
            "model": name,
            "audio_count": len(sub),
            "hypothesis_count": len(sub) * 5,
            "speaker_count": sub["speaker"].nunique(),
            "MAE": float(sub[f"{name}_ae"].mean()),
            "RMSE": float(
                np.sqrt(sub[f"{name}_se"].mean())
            ),
            "mean_prediction_range": float(
                sub[f"{name}_range"].mean()
            ),
            "median_prediction_range": float(
                sub[f"{name}_range"].median()
            ),
            "p95_prediction_range": float(
                sub[f"{name}_range"].quantile(0.95)
            )
        })

    for left, right in PAIRINGS:
        pairs.append({
            "condition": condition,
            "comparison": f"{left}_minus_{right}",
            "MAE_difference": float(
                (sub[f"{left}_ae"] -
                 sub[f"{right}_ae"]).mean()
            ),
            "mean_range_difference": float(
                (sub[f"{left}_range"] -
                 sub[f"{right}_range"]).mean()
            )
        })

result = pd.DataFrame(rows)
paired = pd.DataFrame(pairs)

assert len(result) == 15
assert len(paired) == 9

EXPECTED_PRIMARY = {
    "A": (0.9399, 1.2066),
    "B1": (1.2426, 1.9972),
    "B2": (1.1761, 1.9518),
    "C1": (0.8972, 1.3735),
    "C2": (0.7782, 1.2461)
}

for name, (mae, rmse) in EXPECTED_PRIMARY.items():
    row = result.loc[
        result["condition"].eq("primary_all_400") &
        result["model"].eq(name)
    ].iloc[0]

    assert np.isclose(row["MAE"], mae, atol=1e-4)
    assert np.isclose(row["RMSE"], rmse, atol=1e-4)

OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

def safe_csv(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8")

    if path.exists():
        assert sha256(path) == sha256(tmp), (
            f"Existing file differs: {path}"
        )
        tmp.unlink()
    else:
        tmp.replace(path)

safe_csv(result, OUT)
safe_csv(paired, PAIRED)

report = {
    "status": "PASS",
    "analysis": "preflagged_asr_anomaly_sensitivity",
    "primary_audio_count": 400,
    "primary_observations_retained": True,
    "refitting_performed": False,
    "candidate_audio_ids": sorted(candidate_ids),
    "audit_sha256": sha256(AUDIT),
    "conditions": {
        name: {
            "excluded_audio_ids": sorted(ids),
            "remaining_audio_count": 400 - len(ids)
        }
        for name, ids in CONDITIONS.items()
    },
    "main_results_sha256": sha256(OUT),
    "paired_results_sha256": sha256(PAIRED),
    "source_prediction_sha256": {
        name: sha256(R / spec[0])
        for name, spec in SOURCES.items()
    },
    "interpretation": (
        "Post-hoc sensitivity analysis using previously "
        "flagged ASR anomalies. No retraining and no "
        "primary-analysis exclusions. Reading prompt "
        "disagreement is not verified WER or CER."
    )
}

if REPORT.exists():
    old = json.loads(
        REPORT.read_text(encoding="utf-8")
    )
    assert old == report, (
        "Existing sensitivity report differs."
    )
else:
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    tmp.replace(REPORT)

progress = json.loads(
    CHECK.read_text(encoding="utf-8")
)
progress["step6_anomaly_sensitivity"] = "qc_pass"
progress["step6_anomaly_sensitivity_sha256"] = sha256(OUT)
progress["next_task"] = (
    "Prepare integrated final model, uncertainty and "
    "robustness result tables"
)
progress["updated_at"] = datetime.now(
    timezone.utc
).isoformat()

CHECK.write_text(
    json.dumps(progress, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("=== STEP 6.2 ANOMALY SENSITIVITY AUDIT ===")
print("Status: PASS")
print("Pre-flagged candidates:", sorted(candidate_ids))
print("Primary observations removed: NONE")
print("Model refitting: NO")

print("\n=== PRIMARY VS SENSITIVITY METRICS ===")
print(
    result[
        [
            "condition", "model", "audio_count",
            "MAE", "RMSE", "mean_prediction_range"
        ]
    ].round(4).to_string(index=False)
)

print("\n=== PAIRED MODEL DIFFERENCES ===")
print(paired.round(4).to_string(index=False))

print("\nResults:", OUT)
print("SHA256:", sha256(OUT))
print("Paired results:", PAIRED)
print("Report:", REPORT)
print("Checkpoint updated.")



# ========== ORIGINAL KAGGLE CELL 117 ==========
# %%

import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

P = Path("/kaggle/working/acoustic_linguistic_research")
R = P / "results/step05"
CHECK = P / "checkpoints/research_progress.json"

ASR = P / "features/asr/full/asr_hypotheses_0001-0400.csv"

FILES = {
    "B1": R / "b1_transcript_ridge_oof_predictions.csv",
    "B2": R / "b2_prompt_aware_ridge_oof_predictions.csv",
    "C1": R / "c1_fusion_ridge_oof_predictions.csv",
    "C2": R / "c2_prompt_aware_fusion_ridge_oof_predictions.csv"
}

EXPECTED = {
    ASR: "b67dfe5cca2329407ab07658809434c2e3bad00d227485f3f8c8c9c1dbbf3e80",
    FILES["B1"]: "375ac49b0d5b565adf65d04407d6afc1816f64f5ba108afef8d22458383807dc",
    FILES["B2"]: "ba212868934dbe4d92234f46ccb5f7d7e66a56c781d74aeb24b310b722cfb189",
    FILES["C1"]: "43aa2b8d73c60e44ee4ee2f996f5b243ae2a09c5b9a78f909b2a56ba7306669e",
    FILES["C2"]: "2592e8d28c656db43280d6eea30d578cd488d75f0382e3ed083abf5e2ab86df9"
}

OUT = P / "results/step06_final_tables/step06_asr_variability_per_audio.csv"
SUMMARY = P / "results/step06_final_tables/step06_variability_association.csv"
REPORT = P / "reports/step06_asr_variability_association_v1.json"

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

for path, expected in EXPECTED.items():
    assert path.is_file(), f"Missing: {path}"
    assert sha(path) == expected, f"SHA mismatch: {path}"

def read(path):
    return pd.read_csv(
        path,
        dtype={"audio_id": "string", "speaker": "string"},
        keep_default_na=False
    )

asr = read(ASR)
assert len(asr) == 2000

TOKEN_PATTERN = re.compile(
    r"[^\W_]+(?:['’][^\W_]+)*",
    flags=re.UNICODE
)

def lexical_normalize(text):
    return " ".join(
        w.casefold() for w in TOKEN_PATTERN.findall(str(text))
    )

asr["lexical_text"] = asr["transcription"].map(lexical_normalize)
asr["raw_text"] = asr["transcription"].astype(str).str.strip()

ID = ["pilot_index", "audio_id", "speaker"]
KEYS = ID + ["hypothesis_id"]

assert not asr.duplicated(KEYS).any()
assert asr.groupby("audio_id")["hypothesis_id"].nunique().eq(5).all()

audio = asr.groupby(ID, as_index=False).agg(
    raw_unique=("raw_text", "nunique"),
    lexical_unique=("lexical_text", "nunique"),
    hypothesis_count=("hypothesis_id", "count")
)

assert len(audio) == 400
assert audio["hypothesis_count"].eq(5).all()

audio["raw_variable"] = audio["raw_unique"] > 1
audio["lexical_variable"] = audio["lexical_unique"] > 1

PRED_COLS = {
    "B1": "b1_prediction",
    "B2": "b2_prediction",
    "C1": "c1_prediction",
    "C2": "c2_prediction"
}

for model, path in FILES.items():
    df = read(path)
    col = PRED_COLS[model]
    df[col] = pd.to_numeric(df[col], errors="raise")

    assert len(df) == 2000
    assert not df.duplicated(KEYS).any()

    stat = df.groupby(ID, as_index=False).agg(
        pred_min=(col, "min"),
        pred_max=(col, "max"),
        pred_sd=(col, "std"),
        rows=(col, "count")
    )
    assert len(stat) == 400
    assert stat["rows"].eq(5).all()

    stat[f"{model}_range"] = stat["pred_max"] - stat["pred_min"]
    stat[f"{model}_sd"] = stat["pred_sd"]

    audio = audio.merge(
        stat[ID + [f"{model}_range", f"{model}_sd"]],
        on=ID,
        validate="one_to_one"
    )

assert len(audio) == 400
assert audio["speaker"].nunique() == 20

# Independently reproduce previously reported counts.
assert int(audio["raw_variable"].sum()) == 224
assert int(audio["lexical_variable"].sum()) == 172

observations = []

for model in PRED_COLS:
    col = f"{model}_range"

    rho, naive_p = spearmanr(
        audio["lexical_unique"].to_numpy(),
        audio[col].to_numpy()
    )

    group_means = audio.groupby(
        "lexical_variable"
    )[col].mean()

    observations.append({
        "model": model,
        "n_audios": 400,
        "spearman_rho_lexical_unique_vs_range": float(rho),
        "naive_p_not_for_inference": float(naive_p),
        "mean_range_lexically_invariant": float(
            group_means.loc[False]
        ),
        "mean_range_lexically_variable": float(
            group_means.loc[True]
        ),
        "mean_range_all": float(audio[col].mean()),
        "nonzero_range_audios": int(
            (audio[col] > 1e-10).sum()
        )
    })

summary = pd.DataFrame(observations)

# Invariant normalized transcripts should have exactly
# invariant linguistic features and model predictions.
# Verify this structural expectation; any violation
# requires investigation.
invariant = audio.loc[~audio["lexical_variable"]]

for model in PRED_COLS:
    assert np.allclose(
        invariant[f"{model}_range"],
        0,
        atol=1e-9
    ), f"Unexpected variability in {model} for invariant text"

OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

def safe_csv(frame, path):
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8")
    if path.exists():
        assert sha(path) == sha(temp), (
            f"Existing file differs: {path}"
        )
        temp.unlink()
    else:
        temp.replace(path)

safe_csv(audio.sort_values("pilot_index"), OUT)
safe_csv(summary, SUMMARY)

report = {
    "status": "PASS",
    "analysis": "asr_lexical_variability_vs_prediction_range",
    "audio_count": 400,
    "speaker_count": 20,
    "raw_variable_audios": int(audio["raw_variable"].sum()),
    "lexically_variable_audios": int(
        audio["lexical_variable"].sum()
    ),
    "correlation_type": "Spearman descriptive",
    "naive_p_values_for_inference": False,
    "speaker_clustered_uncertainty_estimated": False,
    "models_retrained": False,
    "all_original_observations_retained": True,
    "source_sha256": {
        str(path.relative_to(P)): sha(path)
        for path in EXPECTED
    },
    "audio_csv_sha256": sha(OUT),
    "summary_csv_sha256": sha(SUMMARY),
    "interpretation": (
        "Observed association, not causal evidence. "
        "Zero prediction range for lexically invariant "
        "hypotheses is expected from deterministic features. "
        "Speaker-clustered uncertainty is still required."
    )
}

if REPORT.exists():
    assert json.loads(
        REPORT.read_text(encoding="utf-8")
    ) == report
else:
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    tmp.replace(REPORT)

progress = json.loads(CHECK.read_text(encoding="utf-8"))
progress["step6_asr_variability_association"] = "qc_pass"
progress["step6_asr_variability_association_sha256"] = sha(SUMMARY)
progress["next_task"] = (
    "Review ASR variability associations and assess "
    "speaker-clustered uncertainty and interpretation"
)
progress["updated_at"] = datetime.now(timezone.utc).isoformat()

CHECK.write_text(
    json.dumps(progress, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("=== STEP 6.4 ASR VARIABILITY ASSOCIATION ===")
print("Status: PASS")
print("Audios:", len(audio))
print("Speakers:", audio["speaker"].nunique())
print("Raw-variable audios:", int(audio["raw_variable"].sum()))
print("Lexically-variable audios:", int(audio["lexical_variable"].sum()))
print("Model refitting: NO")

print("\n=== VARIABILITY VS PREDICTION RANGE ===")
print(summary.round(4).to_string(index=False))

print("\n=== LEXICAL UNIQUENESS DISTRIBUTION ===")
print(
    audio["lexical_unique"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n=== TOP 10 C2 PREDICTION RANGES ===")
print(
    audio.nlargest(10, "C2_range")[
        ["audio_id", "speaker", "lexical_unique",
         "B1_range", "C1_range", "C2_range"]
    ].round(4).to_string(index=False)
)

print("\nAudio CSV:", OUT)
print("Summary CSV:", SUMMARY)
print("Report:", REPORT)
print("Checkpoint updated.")




# ========== ORIGINAL KAGGLE CELL 118 ==========
# %%

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

P = Path("/kaggle/working/acoustic_linguistic_research")
SOURCE = P / "results/step06_final_tables/step06_asr_variability_per_audio.csv"
CHECK = P / "checkpoints/research_progress.json"

OUT = P / "results/step06_final_tables/step06_variable_subset_bootstrap.csv"
REPORT = P / "reports/step06_variable_subset_bootstrap_v1.json"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

assert SOURCE.is_file()
df = pd.read_csv(
    SOURCE,
    dtype={"audio_id": "string", "speaker": "string"}
)

assert len(df) == 400
assert df["audio_id"].nunique() == 400
assert df["speaker"].nunique() == 20

MODELS = ["B1", "B2", "C1", "C2"]
for model in MODELS:
    assert f"{model}_range" in df.columns

assert "lexical_unique" in df.columns

# The main analysis is restricted to genuinely
# variable normalized ASR hypotheses.
variable = df.loc[df["lexical_unique"] > 1].copy()

assert len(variable) == 172
assert variable["audio_id"].is_unique

speaker_ids = sorted(df["speaker"].astype(str).unique())
assert len(speaker_ids) == 20

variable_speakers = sorted(variable["speaker"].astype(str).unique())

# Speaker bootstrap of the complete observed variable
# subset, keeping all selected audios of a speaker together.
groups = {
    s: variable.loc[
        variable["speaker"] == s
    ].copy()
    for s in speaker_ids
}

N_BOOT = 10000
SEED = 20261026
rng = np.random.default_rng(SEED)

draws = rng.integers(0, len(speaker_ids), size=(N_BOOT, 20))

ANOMALIES = {
    "030210164",
    "096340009",
    "096340012",
    "096340017"
}

def rho_or_nan(frame, model):
    x = frame["lexical_unique"].to_numpy(dtype=float)
    y = frame[f"{model}_range"].to_numpy(dtype=float)

    if len(frame) < 5:
        return float("nan")

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")

    return float(spearmanr(x, y).statistic)

rows = []

for model in MODELS:

    observed = rho_or_nan(variable, model)
    assert np.isfinite(observed)

    samples = []

    for draw in draws:
        pieces = [
            groups[speaker_ids[int(i)]]
            for i in draw
            if len(groups[speaker_ids[int(i)]]) > 0
        ]

        if not pieces:
            continue

        sample = pd.concat(pieces, ignore_index=True)
        r = rho_or_nan(sample, model)

        if np.isfinite(r):
            samples.append(r)

    samples = np.asarray(samples, dtype=float)
    assert len(samples) >= 9500, (
        f"Too few valid bootstrap draws: {model}"
    )

    filtered = variable.loc[
        ~variable["audio_id"].isin(ANOMALIES)
    ]

    diagnostic_rho = rho_or_nan(filtered, model)

    rows.append({
        "model": model,
        "subset": "lexically_variable_172",
        "unique_audios": len(variable),
        "speaker_clusters_present": len(variable_speakers),
        "observed_spearman_rho": observed,
        "bootstrap_ci_low": float(
            np.quantile(samples, 0.025)
        ),
        "bootstrap_ci_high": float(
            np.quantile(samples, 0.975)
        ),
        "valid_bootstrap_draws": len(samples),
        "mean_prediction_range": float(
            variable[f"{model}_range"].mean()
        ),
        "diagnostic_n_without_4": len(filtered),
        "diagnostic_rho_without_4": diagnostic_rho
    })

results = pd.DataFrame(rows)

assert len(results) == 4
assert results["observed_spearman_rho"].notna().all()

OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

temp = OUT.with_suffix(".tmp")
results.to_csv(
    temp,
    index=False,
    encoding="utf-8",
    float_format="%.12g"
)

if OUT.exists():
    assert sha(OUT) == sha(temp), (
        "Existing results differ; original preserved."
    )
    temp.unlink()
else:
    temp.replace(OUT)

report = {
    "status": "PASS",
    "analysis": "lexically_variable_subset_speaker_bootstrap",
    "source_sha256": sha(SOURCE),
    "audio_count_full": 400,
    "audio_count_variable": 172,
    "speaker_clusters_total": 20,
    "speaker_clusters_with_variable_audio": len(variable_speakers),
    "bootstrap_iterations": N_BOOT,
    "random_seed": SEED,
    "confidence_interval_method": (
        "speaker_cluster_percentile_bootstrap"
    ),
    "bootstrap_model_refitting": False,
    "excluded_from_primary_model_evaluation": False,
    "diagnostic_anomaly_audio_ids": sorted(ANOMALIES),
    "output_sha256": sha(OUT),
    "limitations": [
        "Analysis is conditional on having lexically variable ASR outputs.",
        "Spearman association does not establish causality.",
        "The same ASR outputs construct both lexical features and prediction variation.",
        "Bootstrap resamples speakers, not independent hypothesis rows.",
        "Only 20 speakers are available in the full pilot.",
        "No fitted model is retrained during bootstrap.",
        "Four-anomaly exclusion is sensitivity analysis only."
    ]
}

if REPORT.exists():
    old = json.loads(
        REPORT.read_text(encoding="utf-8")
    )
    assert old == report, (
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
progress["step6_variable_subset_bootstrap"] = "qc_pass"
progress["step6_variable_subset_bootstrap_sha256"] = sha(OUT)
progress["next_task"] = (
    "Interpret within-variable-audio ASR association, "
    "then assess feature contributions and reporting limitations"
)
progress["updated_at"] = datetime.now(
    timezone.utc
).isoformat()

CHECK.write_text(
    json.dumps(progress, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("=== STEP 6.5 VARIABLE-SUBSET BOOTSTRAP ===")
print("Status: PASS")
print("Full audios: 400")
print("Lexically-variable audios:", len(variable))
print("Speakers represented:", len(variable_speakers))
print("Bootstrap iterations:", N_BOOT)
print("Model refitting: NO")

print("\n=== WITHIN-VARIABLE CORRELATION AND 95% CI ===")
print(
    results[
        [
            "model", "unique_audios",
            "observed_spearman_rho",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "valid_bootstrap_draws"
        ]
    ].round(4).to_string(index=False)
)

print("\n=== SENSITIVITY WITHOUT FOUR ANOMALY CANDIDATES ===")
print(
    results[
        [
            "model", "diagnostic_n_without_4",
            "observed_spearman_rho",
            "diagnostic_rho_without_4"
        ]
    ].round(4).to_string(index=False)
)

print("\n=== VARIABLE-SUBSET MEAN PREDICTION RANGE ===")
print(
    results[
        ["model", "mean_prediction_range"]
    ].round(4).to_string(index=False)
)

print("\nOutput:", OUT)
print("SHA256:", sha(OUT))
print("Report:", REPORT)
print("Checkpoint updated.")


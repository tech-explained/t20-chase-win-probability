"""Train the cricket v1 engines and run the graduation gate.

Pipeline:
  matches (Cricsheet JSON)
    -> chase snapshots      (2nd innings, per completed over)
    -> time split: train < 2024, validate 2024, test >= 2025
    -> logistic regression vs HistGradientBoosting, pick by validation Brier
    -> isotonic calibration of the winner on validation
    -> final Brier / log-loss / calibration table on TEST ONLY
    -> graduation gate (rater/graduation.py)

  first-innings total regressor (HGB) for rating 1st-innings states.

Artifacts -> artifacts/cricket/:
  chase_model.joblib  {model, scaler, features, version}
  total_model.joblib  {model, features, version}
  calibration_report.json
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from rater.engines.cricket.data import load_matches
from rater.engines.cricket.features import (
    CHASE_FEATURES,
    FIRST_INN_FEATURES,
    build_chase_dataset,
    build_first_innings_dataset,
)
from rater.graduation import evaluate_graduation

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts", "cricket")
VERSION = "v1"

# FORMAT selects the training universe: all | t20 | odi.
# T20 and ODI are different enough games that separate models are justified.
FORMAT = os.environ.get("FORMAT", "all")
if FORMAT != "all":
    ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "artifacts", f"cricket-{FORMAT}")

# Split points are overridable via env for experiments, e.g.:
#   TRAIN_END=2024-01-01 VALID_END=2025-07-01 python3 train_model.py
TRAIN_START = os.environ.get("TRAIN_START", "2000-01-01")
TRAIN_END = os.environ.get("TRAIN_END", "2024-01-01")
VALID_END = os.environ.get("VALID_END", "2025-01-01")


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, width: float = 0.05):
    rows = []
    edges = np.arange(0, 1 + width, width)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1 else y_prob <= hi)
        n = int(mask.sum())
        rows.append({
            "bucket": f"{lo:.0%}-{hi:.0%}",
            "n": n,
            "mean_predicted": float(y_prob[mask].mean()) if n else 0.0,
            "actual_win_rate": float(y_true[mask].mean()) if n else 0.0,
        })
    return rows


def brier_baseline_base_rate(y_true: np.ndarray) -> float:
    p = np.full_like(y_true, y_true.mean(), dtype=float)
    return float(brier_score_loss(y_true, p))


def main() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("Loading matches...", flush=True)
    matches = load_matches()
    print(f"  usable matches: {len(matches)}", flush=True)

    print("Building chase dataset...", flush=True)
    chase = build_chase_dataset(matches)
    if FORMAT == "t20":
        chase = chase[chase.is_t20 == 1].reset_index(drop=True)
    elif FORMAT == "odi":
        chase = chase[chase.is_t20 == 0].reset_index(drop=True)
    print(f"  chase snapshots: {len(chase)} (format={FORMAT})", flush=True)

    chase["split"] = "train"
    chase.loc[chase["date"] < TRAIN_START, "split"] = "drop"
    chase.loc[chase["date"] >= TRAIN_END, "split"] = "valid"
    chase.loc[chase["date"] >= VALID_END, "split"] = "test"
    chase = chase[chase.split != "drop"]
    tr = chase[chase.split == "train"]
    va = chase[chase.split == "valid"]
    te = chase[chase.split == "test"]
    print(f"  train={len(tr)} valid={len(va)} test={len(te)}", flush=True)
    if len(te) == 0:
        raise SystemExit("No test snapshots — check date coverage.")

    Xtr, ytr = tr[CHASE_FEATURES].to_numpy(), tr["label"].to_numpy()
    Xva, yva = va[CHASE_FEATURES].to_numpy(), va["label"].to_numpy()
    Xte, yte = te[CHASE_FEATURES].to_numpy(), te["label"].to_numpy()

    scaler = StandardScaler().fit(Xtr)
    candidates = {
        "logreg": LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(Xtr), ytr),
        "hgb": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                             max_leaf_nodes=31, random_state=7).fit(Xtr, ytr),
    }
    # validation Brier to pick the winner
    val_brier = {}
    for name, mdl in candidates.items():
        Xv = scaler.transform(Xva) if name == "logreg" else Xva
        val_brier[name] = brier_score_loss(yva, mdl.predict_proba(Xv)[:, 1])
        print(f"  valid Brier [{name}]: {val_brier[name]:.4f}", flush=True)
    winner = min(val_brier, key=val_brier.get)
    print(f"  winner: {winner}", flush=True)
    mdl = candidates[winner]

    # isotonic calibration on validation, then refit winner on train+valid.
    # Keep the calibrator only if it actually helps on validation (no-leakage
    # model selection): raw logistic probabilities are often well-calibrated
    # on their own, and isotonic step functions can overfit thin regions.
    # CALIB=off forces raw probabilities (useful when regimes shift).
    Xcal = scaler.transform(Xva) if winner == "logreg" else Xva
    p_cal_raw = mdl.predict_proba(Xcal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal_raw, yva)
    p_cal_iso = iso.predict(p_cal_raw)
    use_iso = (os.environ.get("CALIB", "on") != "off"
               and brier_score_loss(yva, p_cal_iso) < brier_score_loss(yva, p_cal_raw))
    print(f"  calibrator: {'isotonic (helps on valid)' if use_iso else 'none (raw better on valid)'}",
          flush=True)
    Xfull = np.vstack([Xtr, Xva])
    yfull = np.concatenate([ytr, yva])
    if winner == "logreg":
        scaler = StandardScaler().fit(Xfull)
        mdl = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(Xfull), yfull)
        p_raw = mdl.predict_proba(scaler.transform(Xte))[:, 1]
    else:
        mdl = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                             max_leaf_nodes=31, random_state=7).fit(Xfull, yfull)
        p_raw = mdl.predict_proba(Xte)[:, 1]
    p_test = iso.predict(p_raw) if use_iso else p_raw

    brier = float(brier_score_loss(yte, p_test))
    ll = float(log_loss(yte, p_test))
    cal = calibration_table(yte, p_test)
    print(f"  TEST Brier: {brier:.4f}  log-loss: {ll:.4f}", flush=True)

    # ---- first-innings total regressor ---------------------------------
    print("Building first-innings dataset...", flush=True)
    fi = build_first_innings_dataset(matches)
    if FORMAT == "t20":
        fi = fi[fi.is_t20 == 1].reset_index(drop=True)
    elif FORMAT == "odi":
        fi = fi[fi.is_t20 == 0].reset_index(drop=True)
    fi["split"] = "train"
    fi.loc[fi["date"] >= TRAIN_END, "split"] = "valid"
    fi.loc[fi["date"] >= VALID_END, "split"] = "test"
    ftr = fi[fi.split.isin(["train", "valid"])]
    fte = fi[fi.split == "test"]
    treg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                        max_leaf_nodes=31, random_state=7)
    treg.fit(ftr[FIRST_INN_FEATURES].to_numpy(), ftr["final_total"].to_numpy())
    fi_pred = treg.predict(fte[FIRST_INN_FEATURES].to_numpy())
    fi_mae = float(mean_absolute_error(fte["final_total"].to_numpy(), fi_pred))
    print(f"  first-innings total MAE on test: {fi_mae:.1f} runs (n={len(fte)})", flush=True)

    # ---- graduation ------------------------------------------------------
    report = {
        "engine": f"cricket-chase-{VERSION}-{FORMAT}",
        "brier": brier,
        "log_loss": ll,
        "brier_baseline_base_rate": brier_baseline_base_rate(yte),
        # No historical bookmaker odds in Cricsheet: book baseline unavailable.
        "brier_baseline_book": None,
        "n_test": len(te),
        "calibration": cal,
        "train_window": f"<= {TRAIN_END}",
        "valid_window": f"{TRAIN_END}..{VALID_END}",
        "test_window": f">= {VALID_END}",
        "first_innings_total_mae": round(fi_mae, 1),
        "n_first_innings_test": len(fte),
        "test_chase_win_rate": round(float(yte.mean()), 4),
    }
    verdict = evaluate_graduation(report)
    report["graduated"] = verdict.go
    report["graduation_gaps"] = verdict.gaps
    print("\n" + str(verdict), flush=True)

    joblib.dump({"model": mdl, "scaler": scaler if winner == "logreg" else None,
                 "calibrator": iso if use_iso else None, "features": CHASE_FEATURES,
                 "kind": winner, "version": VERSION,
                 "train_start": TRAIN_START, "train_end": TRAIN_END, "valid_end": VALID_END},
                os.path.join(ARTIFACT_DIR, "chase_model.joblib"))
    joblib.dump({"model": treg, "features": FIRST_INN_FEATURES, "version": VERSION},
                os.path.join(ARTIFACT_DIR, "total_model.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "calibration_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nArtifacts written to {ARTIFACT_DIR}", flush=True)


if __name__ == "__main__":
    main()

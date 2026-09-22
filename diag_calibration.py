"""Diagnose the mid-range miscalibration: label shift or model bias?"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

from rater.engines.cricket.data import load_matches
from rater.engines.cricket.features import CHASE_FEATURES, build_chase_dataset

ART = "artifacts/cricket"
bundle = joblib.load(f"{ART}/chase_model.joblib")
model, scaler, iso, kind = bundle["model"], bundle["scaler"], bundle["calibrator"], bundle["kind"]

matches = load_matches()
df = build_chase_dataset(matches)
df["split"] = "train"
df.loc[df["date"] >= "2024-01-01", "split"] = "valid"
df.loc[df["date"] >= "2025-01-01", "split"] = "test"
df["year"] = df["date"].dt.year

print("== label mean (chasing team win rate) by split ==")
print(df.groupby("split")["label"].agg(["mean", "count"]).round(4))
print()
print("== by year (2022+) ==")
print(df[df.year >= 2022].groupby("year")["label"].agg(["mean", "count"]).round(4))
print()

for split in ["valid", "test"]:
    d = df[df.split == split]
    X = d[CHASE_FEATURES].to_numpy()
    if kind == "logreg":
        X = scaler.transform(X)
    p_raw = model.predict_proba(X)[:, 1]
    p = iso.predict(p_raw)
    d = d.assign(p=p)
    mid = d[(d.p >= 0.25) & (d.p <= 0.80)]
    print(f"== {split}: mid-range (pred 25-80%) ==")
    print(f"   n={len(mid)}  mean_pred={mid.p.mean():.3f}  actual={mid.label.mean():.3f}  "
          f"gap={mid.label.mean() - mid.p.mean():+.3f}")
    print("   by year:")
    print(mid.groupby("year").agg(mean_pred=("p", "mean"), actual=("label", "mean"),
                                  n=("label", "count")).round(3))
    print()

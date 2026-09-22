#!/usr/bin/env python3
"""Print dataset stats (matches, snapshots, splits) without training."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rater.engines.cricket.data import load_matches
from rater.engines.cricket.features import build_chase_dataset, build_first_innings_dataset

matches = load_matches()
print(f"matches: {len(matches)}")
chase = build_chase_dataset(matches)
fi = build_first_innings_dataset(matches)
print(f"chase snapshots: {len(chase)}  (label mean={chase['label'].mean():.3f})")
print(f"1st-innings snapshots: {len(fi)}")
for name, df in (("chase", chase), ("1st-inn", fi)):
    q = df["date"].dt.date
    print(f"  {name}: {q.min()} .. {q.max()}")
    n_tr = (df["date"] < "2024-01-01").sum()
    n_va = ((df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01")).sum()
    n_te = (df["date"] >= "2025-01-01").sum()
    print(f"    train={n_tr} valid={n_va} test={n_te}")

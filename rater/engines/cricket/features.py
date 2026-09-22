"""Build training snapshots from parsed matches. No leakage: every feature
is knowable at the snapshot moment; the label is the eventual result.

Two datasets:
1. chase snapshots — one row per completed over of a 2nd innings, while the
   result is still undecided. Label: 1 if the chasing side eventually won.
2. first-innings snapshots — one row per completed over of a 1st innings.
   Target: final 1st-innings total (regression). Used only to project an
   expected target so 1st-innings states can be rated through the chase model.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .data import delivery_runs, innings_progress, innings_total, is_legal_ball, wickets_in_delivery

CHASE_FEATURES = [
    "is_t20",
    "target",
    "balls_bowled",
    "balls_remaining",
    "runs_scored",
    "wickets_lost",
    "runs_needed",
    "current_rr",
    "required_rr",
    "rr_diff",          # required_rr - current_rr (negative = ahead)
    "needed_per_ball",
    "wickets_in_hand",
    "pct_balls_done",
    "runs_last_3_overs",
    "wkts_last_3_overs",
]
FIRST_INN_FEATURES = [
    "is_t20",
    "balls_bowled",
    "runs_scored",
    "wickets_lost",
    "current_rr",
    "pct_balls_done",
    "runs_last_3_overs",
    "wkts_last_3_overs",
]


def _scheduled_balls(match: Dict) -> int | None:
    overs = match.get("scheduled_overs")
    if overs:
        return int(overs) * 6
    # fall back to format default
    return 120 if match["match_type"] == "T20" else 300


def _over_snapshots(inning: Dict, total_balls: int) -> List[Dict]:
    """Running state after each completed over (6 legal balls)."""
    balls = runs = wickets = 0
    # ring buffer of last 18 legal balls for momentum features
    recent: List[Dict] = []
    snaps = []
    legal_in_over = 0
    for over in inning.get("overs", []):
        for d in over.get("deliveries", []):
            r = delivery_runs(d)
            w = wickets_in_delivery(d)
            if is_legal_ball(d):
                balls += 1
                legal_in_over += 1
                recent.append({"runs": r, "wkts": w})
                recent = recent[-18:]
            else:
                # extras still count toward runs/wickets and momentum
                if recent:
                    recent[-1]["runs"] += r
                    recent[-1]["wkts"] += w
                else:
                    recent.append({"runs": r, "wkts": w})
            runs += r
            wickets += w
            if legal_in_over == 6:
                legal_in_over = 0
                snaps.append({
                    "balls_bowled": balls,
                    "runs_scored": runs,
                    "wickets_lost": wickets,
                    "runs_last_3_overs": sum(x["runs"] for x in recent),
                    "wkts_last_3_overs": sum(x["wkts"] for x in recent),
                })
    return snaps


def build_chase_dataset(matches: List[Dict]) -> pd.DataFrame:
    rows = []
    for m in matches:
        total_balls = _scheduled_balls(m)
        if not total_balls:
            continue
        first, second = m["innings"][0], m["innings"][1]
        first_tot = innings_total(first)
        # 1st innings must be complete (all overs or all out); else target is suspect
        if first_tot["balls"] < total_balls and first_tot["wickets"] < 10:
            continue
        target = first_tot["runs"] + 1
        chasing_team = second.get("team")
        label = 1 if m["winner"] == chasing_team else 0
        for s in _over_snapshots(second, total_balls):
            balls, runs, wkts = s["balls_bowled"], s["runs_scored"], s["wickets_lost"]
            if balls >= total_balls or wkts >= 10:
                continue  # innings over
            runs_needed = target - runs
            if runs_needed <= 0:
                continue  # already chased; nothing to rate
            balls_remaining = total_balls - balls
            current_rr = runs / balls * 6 if balls else 0.0
            required_rr = runs_needed / balls_remaining * 6
            rows.append({
                "match": m["file"],
                "date": m["date"],
                "is_t20": 1 if m["match_type"] == "T20" else 0,
                "target": target,
                "balls_bowled": balls,
                "balls_remaining": balls_remaining,
                "runs_scored": runs,
                "wickets_lost": wkts,
                "runs_needed": runs_needed,
                "current_rr": current_rr,
                "required_rr": required_rr,
                "rr_diff": required_rr - current_rr,
                "needed_per_ball": runs_needed / balls_remaining,
                "wickets_in_hand": 10 - wkts,
                "pct_balls_done": balls / total_balls,
                "runs_last_3_overs": s["runs_last_3_overs"],
                "wkts_last_3_overs": s["wkts_last_3_overs"],
                "label": label,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "match", "balls_bowled"]).reset_index(drop=True)
    return df


def build_first_innings_dataset(matches: List[Dict]) -> pd.DataFrame:
    rows = []
    for m in matches:
        total_balls = _scheduled_balls(m)
        if not total_balls:
            continue
        first = m["innings"][0]
        tot = innings_total(first)
        if tot["balls"] < total_balls and tot["wickets"] < 10:
            continue  # incomplete 1st innings
        final_total = tot["runs"]
        for s in _over_snapshots(first, total_balls):
            balls, runs, wkts = s["balls_bowled"], s["runs_scored"], s["wickets_lost"]
            if balls >= total_balls or wkts >= 10:
                continue
            rows.append({
                "match": m["file"],
                "date": m["date"],
                "is_t20": 1 if m["match_type"] == "T20" else 0,
                "balls_bowled": balls,
                "runs_scored": runs,
                "wickets_lost": wkts,
                "current_rr": runs / balls * 6 if balls else 0.0,
                "pct_balls_done": balls / total_balls,
                "runs_last_3_overs": s["runs_last_3_overs"],
                "wkts_last_3_overs": s["wkts_last_3_overs"],
                "final_total": final_total,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "match", "balls_bowled"]).reset_index(drop=True)
    return df

"""Parse Cricsheet ball-by-ball JSON into normalized match dicts.

Cricsheet schema (v1 JSON): top-level keys "meta", "info", "innings".
info: match_type ("T20"/"ODI"/...), dates, teams, outcome {winner, by},
      overs (scheduled), method (e.g. "DLS" when rain-affected), city/venue.
innings: [{"team": str, "overs": [{"over": int, "deliveries": [...]}]}]
delivery: {"batter","bowler","non_striker",
           "runs": {"batter": int, "extras": int, "total": int},
           "extras": {"wides": n, "noballs": n, ...} (optional),
           "wickets": [{"player_out": str, "kind": str, ...}] (optional)}
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Iterator, List

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_DIR = os.path.join(MODEL_DIR, "data", "raw")

WHITE_BALL_TYPES = {"T20", "ODI"}


def iter_match_files(raw_dir: str = RAW_DIR) -> Iterator[str]:
    for path in sorted(glob.glob(os.path.join(raw_dir, "*", "*.json"))):
        yield path


def load_match(path: str) -> Dict[str, Any] | None:
    """Load one file; return None for anything we can't use for training."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    info = data.get("info", {})
    if info.get("match_type") not in WHITE_BALL_TYPES:
        return None
    innings = data.get("innings", [])
    if len(innings) != 2:
        return None
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    # need a decisive result and no rain-rule revision of the target
    if not winner or "result" in outcome and outcome.get("result") in {"tie", "no result"}:
        return None
    if info.get("method") or info.get("revisions"):
        return None  # DLS / revised target: target math gets murky; drop in v1
    dates = info.get("dates", [])
    return {
        "file": os.path.basename(path),
        "match_type": info["match_type"],
        "date": dates[0] if dates else None,
        "teams": info.get("teams", []),
        "winner": winner,
        "scheduled_overs": info.get("overs"),
        "venue": info.get("venue", ""),
        "innings": innings,
    }


def delivery_runs(delivery: Dict) -> int:
    return int(delivery.get("runs", {}).get("total", 0))


def is_legal_ball(delivery: Dict) -> bool:
    extras = delivery.get("extras", {}) or {}
    return "wides" not in extras and "noballs" not in extras


def wickets_in_delivery(delivery: Dict) -> int:
    return len(delivery.get("wickets", []) or [])


def innings_progress(inning: Dict) -> Iterator[Dict[str, Any]]:
    """Yield running totals after each delivery: balls, runs, wickets.

    balls counts legal deliveries only; wides/no-balls add runs, not balls.
    """
    balls = runs = wickets = 0
    for over in inning.get("overs", []):
        for d in over.get("deliveries", []):
            if is_legal_ball(d):
                balls += 1
            runs += delivery_runs(d)
            wickets += wickets_in_delivery(d)
            yield {"balls": balls, "runs": runs, "wickets": wickets}


def innings_total(inning: Dict) -> Dict[str, int]:
    """Final totals for a completed innings."""
    last = {"balls": 0, "runs": 0, "wickets": 0}
    for s in innings_progress(inning):
        last = s
    return last


def load_matches(raw_dir: str = RAW_DIR, limit: int | None = None) -> List[Dict[str, Any]]:
    matches = []
    for path in iter_match_files(raw_dir):
        m = load_match(path)
        if m is not None:
            matches.append(m)
            if limit and len(matches) >= limit:
                break
    return matches

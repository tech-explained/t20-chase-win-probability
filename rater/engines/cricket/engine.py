"""Cricket v1 engine: live win probability for white-ball (T20/ODI) matches.

Implements rater.base.Engine. Loads the trained artifacts from
artifacts/cricket/ (chase_model.joblib, total_model.joblib,
calibration_report.json).

Supported event states (event.state):
  phase="chase":         {"format": "T20"|"ODI", "target": int,
                          "balls_bowled": int, "runs": int, "wickets": int,
                          "chasing_team": str,
                          optional: "runs_last_3_overs", "wkts_last_3_overs",
                          "scheduled_overs"}
  phase="first_innings": {"format": ..., "balls_bowled": int, "runs": int,
                          "wickets": int,  (+ same optionals)}
                         Rated via projected final total -> chase model at
                         chase start. Marked experimental in notes.
  phase="pre_match":     not modeled in v1 -> 50/50, low confidence,
                         graduated=False.

event.participants = (team_a, team_b); fair_probabilities keys are the
team names.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np

from rater.base import Engine, Event, Rating
from rater.engines.cricket.features import CHASE_FEATURES, FIRST_INN_FEATURES

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts", "cricket")


class CricketChaseEngine(Engine):
    name = "cricket-chase"
    sport = "cricket"
    version = "v1"

    def __init__(self, artifact_dir: str = ARTIFACT_DIR, flavor: str = "") -> None:
        """flavor: "" (combined), "t20", or "odi" — selects the artifact subdir."""
        if flavor:
            artifact_dir = os.path.join(os.path.dirname(artifact_dir), f"cricket-{flavor}")
        self.flavor = flavor
        try:
            bundle = joblib.load(os.path.join(artifact_dir, "chase_model.joblib"))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"cricket artifacts not found in {artifact_dir}. "
                f"Train first: FORMAT={flavor or 'all'} python3 train_model.py "
                f"(from the model/ directory)")
        self._model = bundle["model"]
        self._scaler = bundle["scaler"]          # None for tree model
        self._calibrator = bundle["calibrator"]  # isotonic, fitted on validation
        self._kind = bundle["kind"]
        tbundle = joblib.load(os.path.join(artifact_dir, "total_model.joblib"))
        self._total_model = tbundle["model"]
        with open(os.path.join(artifact_dir, "calibration_report.json")) as f:
            self._report = json.load(f)
        self._graduated = bool(self._report.get("graduated", False))

    @property
    def engine_id(self) -> str:
        base = f"{self.name}-{self.version}"
        return f"{base}-{self.flavor}" if self.flavor else base

    # -- Engine interface -------------------------------------------------
    def supports(self, event: Event) -> bool:
        if event.sport != "cricket":
            return False
        if event.state.get("phase") not in {"chase", "first_innings", "pre_match"}:
            return False
        if self.flavor and event.state.get("format", "T20") != self.flavor.upper():
            return False  # a T20-trained engine must not rate ODI states
        return True

    def calibration_report(self):
        return self._report

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _total_balls(state) -> int:
        if state.get("scheduled_overs"):
            return int(state["scheduled_overs"]) * 6
        return 120 if state.get("format", "T20") == "T20" else 300

    def _chase_features(self, state) -> np.ndarray:
        total_balls = self._total_balls(state)
        balls = int(state["balls_bowled"])
        runs = int(state["runs"])
        wkts = int(state["wickets"])
        target = int(state["target"])
        balls_remaining = total_balls - balls
        runs_needed = target - runs
        current_rr = runs / balls * 6 if balls else 0.0
        required_rr = runs_needed / balls_remaining * 6 if balls_remaining else 0.0
        recent_balls = min(balls, 18)
        r3 = state.get("runs_last_3_overs")
        w3 = state.get("wkts_last_3_overs")
        if r3 is None:  # prorate when recent history isn't supplied
            r3 = runs * (recent_balls / balls) if balls else 0.0
        if w3 is None:
            w3 = wkts * (recent_balls / balls) if balls else 0.0
        row = {
            "is_t20": 1 if state.get("format", "T20") == "T20" else 0,
            "target": target,
            "balls_bowled": balls,
            "balls_remaining": balls_remaining,
            "runs_scored": runs,
            "wickets_lost": wkts,
            "runs_needed": runs_needed,
            "current_rr": current_rr,
            "required_rr": required_rr,
            "rr_diff": required_rr - current_rr,
            "needed_per_ball": runs_needed / balls_remaining if balls_remaining else 0.0,
            "wickets_in_hand": 10 - wkts,
            "pct_balls_done": balls / total_balls,
            "runs_last_3_overs": r3,
            "wkts_last_3_overs": w3,
        }
        return np.array([[row[c] for c in CHASE_FEATURES]], dtype=float)

    def _predict_chase(self, state) -> float:
        X = self._chase_features(state)
        if self._kind == "logreg":
            X = self._scaler.transform(X)
        p_raw = self._model.predict_proba(X)[:, 1]
        if self._calibrator is not None:
            return float(self._calibrator.predict(p_raw)[0])
        return float(p_raw[0])

    def _project_total(self, state) -> float:
        balls = int(state["balls_bowled"])
        runs = int(state["runs"])
        wkts = int(state["wickets"])
        recent_balls = min(balls, 18)
        r3 = state.get("runs_last_3_overs")
        w3 = state.get("wkts_last_3_overs")
        if r3 is None:
            r3 = runs * (recent_balls / balls) if balls else 0.0
        if w3 is None:
            w3 = wkts * (recent_balls / balls) if balls else 0.0
        total_balls = self._total_balls(state)
        row = {
            "is_t20": 1 if state.get("format", "T20") == "T20" else 0,
            "balls_bowled": balls,
            "runs_scored": runs,
            "wickets_lost": wkts,
            "current_rr": runs / balls * 6 if balls else 0.0,
            "pct_balls_done": balls / total_balls,
            "runs_last_3_overs": r3,
            "wkts_last_3_overs": w3,
        }
        X = np.array([[row[c] for c in FIRST_INN_FEATURES]], dtype=float)
        return float(self._total_model.predict(X)[0])

    # -- rate ---------------------------------------------------------------
    def rate(self, event: Event) -> Rating:
        team_a, team_b = event.participants
        state = dict(event.state)
        phase = state.get("phase")
        notes = [f"model={self.engine_id}"]

        if phase == "pre_match":
            return Rating(event, self.engine_id,
                          {team_a: 0.5, team_b: 0.5},
                          confidence="low", graduated=False,
                          notes=notes + ["pre-match not modeled in v1; 50/50 placeholder"])

        if phase == "first_innings":
            proj_total = self._project_total(state)
            chase_state = {"format": state.get("format", "T20"), "target": int(round(proj_total)) + 1,
                           "balls_bowled": 0, "runs": 0, "wickets": 0,
                           "scheduled_overs": state.get("scheduled_overs")}
            p_chase_wins = self._predict_chase(chase_state)
            p_a = 1.0 - p_chase_wins  # team_a batted first by convention
            mae = self._report.get("first_innings_total_mae")
            return Rating(event, self.engine_id,
                          {team_a: p_a, team_b: 1.0 - p_a},
                          confidence="low", graduated=self._graduated,
                          notes=notes + [f"via projected 1st-innings total {proj_total:.0f} "
                                         f"(test MAE {mae} runs); experimental"])

        # phase == "chase"
        chasing = state.get("chasing_team", team_b)
        other = team_a if chasing == team_b else team_b
        balls = int(state["balls_bowled"])
        runs = int(state["runs"])
        wkts = int(state["wickets"])
        target = int(state["target"])
        total_balls = self._total_balls(state)

        if runs >= target:
            p = 1.0
            notes.append("target already reached")
        elif wkts >= 10 or balls >= total_balls:
            p = 1.0 if runs >= target else 0.0
            notes.append("innings complete")
        else:
            p = self._predict_chase(state)
        p = float(min(max(p, 0.0), 1.0))
        return Rating(event, self.engine_id,
                      {chasing: p, other: 1.0 - p},
                      confidence="medium", graduated=self._graduated,
                      notes=notes)

"""Core types shared by every sport engine.

Design rules:
- ``Event`` carries everything the engine may legally use at rating time.
  Nothing in ``event.state`` may depend on the eventual result (no leakage).
- ``Rating.fair_probabilities`` maps each possible outcome label
  (e.g. "home", "away", or team names) to a fair win probability.
  They must sum to ~1.0.
- ``graduated`` is False until the engine passes the graduation gate
  (see rater/graduation.py). Non-graduated ratings are research-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class Event:
    """A single rateable event.

    sport: canonical sport key, e.g. "cricket", "tennis", "soccer",
           "mlb", "nfl", "table_tennis", "volleyball", "hockey", "esports".
    competition: league/tournament label, free text.
    participants: (side_a, side_b) — teams or players, in a stable order.
    state: per-sport live/pre-match state dict. Keys are defined by each
           engine's documentation; only information known at rating time.
    market: optional observed bookmaker prices {"side_a": 1.95, ...}.
            Engines must NEVER need this; it is carried so the caller can
            compute edge = fair_prob * decimal_odds - 1 in one place.
    """
    sport: str
    competition: str = ""
    participants: Tuple[str, str] = ("", "")
    state: Dict[str, Any] = field(default_factory=dict)
    market: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraduationVerdict:
    """Outcome of the calibration graduation gate for one engine version."""
    go: bool
    gaps: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - convenience
        status = "GO" if self.go else "NO-GO"
        lines = [f"Graduation: {status}"]
        for k, v in self.metrics.items():
            lines.append(f"  {k}: {v}")
        for g in self.gaps:
            lines.append(f"  gap: {g}")
        return "\n".join(lines)


@dataclass
class Rating:
    """A fair-probability rating for one event."""
    event: Event
    engine: str                      # e.g. "cricket-chase-v1"
    fair_probabilities: Dict[str, float]
    confidence: str = "medium"       # high | medium | low
    graduated: bool = False
    notes: List[str] = field(default_factory=list)

    def edge_vs(self, side: str) -> float | None:
        """Expected value of a 1-unit stake at the observed decimal odds.

        Returns None when no market price is attached to the event.
        edge = fair_p * decimal_odds - 1  (fractional, e.g. 0.05 = +5% EV)
        """
        odds = self.event.market.get(side)
        p = self.fair_probabilities.get(side)
        if odds is None or p is None or odds <= 1.0:
            return None
        return p * odds - 1.0


class Engine:
    """Interface every per-sport engine implements."""

    name: str = "base"        # human label, e.g. "cricket-chase"
    sport: str = "base"       # canonical sport key this engine serves
    version: str = "v0"       # bump on any retrain / feature change

    # ---- required -----------------------------------------------------
    def supports(self, event: Event) -> bool:
        """True if this engine can rate this event (sport + state shape)."""
        raise NotImplementedError

    def rate(self, event: Event) -> Rating:
        """Return fair win probabilities for the event."""
        raise NotImplementedError

    # ---- strongly recommended -----------------------------------------
    def calibration_report(self) -> Dict[str, Any]:
        """Backtest metrics from the engine's own training run.

        Expected keys: brier, log_loss, n_test, calibration (bucket table),
        train_window, test_window. Used by the graduation gate.
        """
        return {}

    @property
    def engine_id(self) -> str:
        return f"{self.name}-{self.version}"

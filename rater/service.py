"""RatingService: registry + routing + graduation enforcement.

Usage:
    service = RatingService()
    service.register(CricketChaseEngine())
    rating = service.rate(event)              # always returns a Rating
    if rating.graduated and (edge := rating.edge_vs("away")) and edge > 0.05:
        ...  # actionable

``rate_actionable`` returns None unless the rating is graduated AND the
requested edge threshold is met — this is the only entry point the
research applications should use.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import Engine, Event, Rating


class RatingService:
    def __init__(self) -> None:
        self._engines: List[Engine] = []

    def register(self, engine: Engine) -> None:
        self._engines.append(engine)

    def engines(self) -> List[Engine]:
        return list(self._engines)

    def find_engine(self, event: Event) -> Optional[Engine]:
        for engine in self._engines:
            if engine.sport == event.sport and engine.supports(event):
                return engine
        return None

    def rate(self, event: Event) -> Rating:
        engine = self.find_engine(event)
        if engine is None:
            if any(e.sport == event.sport for e in self._engines):
                note = (f"engines registered for sport={event.sport!r} but none "
                        f"supports this event state")
            else:
                note = f"no engine registered for sport={event.sport!r}"
            return Rating(
                event=event,
                engine="none",
                fair_probabilities={},
                confidence="low",
                graduated=False,
                notes=[note],
            )
        return engine.rate(event)

    def rate_actionable(self, event: Event, side: str, min_edge: float = 0.05) -> Optional[Rating]:
        """Return the rating only if graduated and edge >= min_edge."""
        rating = self.rate(event)
        if not rating.graduated:
            return None
        edge = rating.edge_vs(side)
        if edge is None or edge < min_edge:
            return None
        return rating

    def status(self) -> Dict[str, Dict]:
        """One-line health per registered engine."""
        out = {}
        for e in self._engines:
            rep = e.calibration_report()
            out[e.engine_id] = {
                "sport": e.sport,
                "graduated": bool(rep.get("graduated", False)),
                "brier": rep.get("brier"),
                "n_test": rep.get("n_test"),
            }
        return out

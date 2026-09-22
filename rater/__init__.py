"""One rating service, pluggable per-sport engines.

Every sport implements the same interface:

    engine = SomeSportEngine(...)
    rating = engine.rate(event)          # -> Rating with fair win probabilities

The RatingService routes an Event to the right engine and enforces the
graduation gate: an engine that has not passed calibration is never
presented as actionable. Downstream applications must check
``rating.graduated`` (or use ``service.rate_actionable``) before staking.
"""

from .base import Event, Rating, Engine, GraduationVerdict
from .service import RatingService
from .graduation import evaluate_graduation

__all__ = ["Event", "Rating", "Engine", "GraduationVerdict", "RatingService", "evaluate_graduation"]

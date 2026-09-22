#!/usr/bin/env python3
"""Query the rating service for a live cricket match state.

Examples:
  # 2nd innings chase: India need 42 off 30 balls, 180/4, chasing 222
  python3 predict.py --sport cricket --team-a "India" --team-b "Sri Lanka" \\
      --phase chase --format T20 --target 222 --balls 90 --runs 180 --wickets 4 \\
      --chasing "India" --market-odds '{"India": 1.61, "Sri Lanka": 2.30}'

  # 1st innings: 96/2 after 10 overs
  python3 predict.py --sport cricket --team-a "India" --team-b "Sri Lanka" \\
      --phase first_innings --format T20 --balls 60 --runs 96 --wickets 2

  # pre-match
  python3 predict.py --sport cricket --team-a "India" --team-b "Sri Lanka" \\
      --phase pre_match
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rater import Event, RatingService
from rater.engines.cricket.engine import CricketChaseEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default="cricket")
    ap.add_argument("--competition", default="")
    ap.add_argument("--team-a", required=True)
    ap.add_argument("--team-b", required=True)
    ap.add_argument("--phase", required=True, choices=["chase", "first_innings", "pre_match"])
    ap.add_argument("--format", default="T20", choices=["T20", "ODI"])
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--balls", type=int, default=0)
    ap.add_argument("--runs", type=int, default=0)
    ap.add_argument("--wickets", type=int, default=0)
    ap.add_argument("--chasing", default="")
    ap.add_argument("--flavor", default="t20", choices=["t20", "odi", "all"],
                    help="model flavor: t20 (graduated), odi/all (research-only)")
    ap.add_argument("--market-odds", default="{}",
                    help='JSON map team->decimal odds, e.g. \'{"India": 1.61}\'')
    args = ap.parse_args()

    state = {"phase": args.phase, "format": args.format,
             "balls_bowled": args.balls, "runs": args.runs, "wickets": args.wickets}
    if args.phase == "chase":
        state.update({"target": args.target, "chasing_team": args.chasing or args.team_b})
    event = Event(sport=args.sport, competition=args.competition,
                  participants=(args.team_a, args.team_b),
                  state=state, market=json.loads(args.market_odds))

    service = RatingService()
    service.register(CricketChaseEngine(flavor="" if args.flavor == "all" else args.flavor))
    rating = service.rate(event)

    print(f"engine: {rating.engine}  graduated={rating.graduated}  "
          f"confidence={rating.confidence}")
    for team, p in rating.fair_probabilities.items():
        edge = rating.edge_vs(team)
        edge_s = f"  edge vs {event.market.get(team, 'n/a')}: {edge:+.1%}" if edge is not None else ""
        print(f"  {team}: fair {p:.1%}{edge_s}")
    for n in rating.notes:
        print(f"  note: {n}")
    if not rating.graduated:
        print("RESEARCH ONLY — engine has not graduated; treat as research output only.")


if __name__ == "__main__":
    main()

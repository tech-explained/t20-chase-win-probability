# Calibrated Live Win Probabilities for Men's T20 Run Chases

A deliberately simple, well-calibrated logistic-regression model for live win
probability during men's T20 run chases — plus a reusable **graduation gate**
that only lets a model ship when it proves calibration on strictly
time-ordered, out-of-sample data.

Companion to the paper *"Less Is More: Calibrated Live Win Probabilities for
Men's T20 Run Chases from a Simple Logistic Model"* (submitted to the MIT
Sloan Sports Analytics Conference 2027 Research Paper Competition).

## Results (held-out test: 19,070 chase snapshots, Jul 2025 – Sep 2026)

| Metric | Model | Baseline |
|---|---|---|
| Brier score | **0.1114** | 0.2500 (constant base rate) |
| Brier skill | **55.4%** | — |
| Log-loss | 0.344 | — |
| Calibration | every 5pp bucket with ≥200 samples within **5pp** of observed win rate (worst: 4.6pp) | — |

Notable negative results: gradient boosting did not beat logistic regression
out-of-sample, and isotonic recalibration fitted on validation *hurt* test
calibration — evidence of regime shift that post-hoc recalibration cannot fix.

## Method in brief

- **Data:** free Cricsheet ball-by-ball JSON — men's T20Is plus IPL, BBL, PSL,
  CPL, BPL, MLC, LPL, ILT20, MSL, NPL, SMA (~10,200 usable matches after
  excluding DLS-affected, tied, and no-result games).
- **Snapshots:** one row per completed over of the second innings while the
  result is still undecided; label = 1 if the chasing side eventually won.
- **Features (15, zero leakage — all knowable at snapshot time):** target,
  balls bowled/remaining, runs scored, wickets lost/in hand, runs needed,
  current vs required run rate and their difference, runs needed per ball,
  proportion of innings completed, runs/wickets in the last 3 overs.
- **Splits:** strictly chronological — train 2019–2023, validate 2024–2025H1,
  test 2025-07→2026-09. No random shuffling, no peeking.

## Reproduce

```bash
pip install -r requirements.txt
./download_data.sh            # fetches free Cricsheet JSON (~10k matches)
python3 train_model.py        # trains, backtests, runs the graduation gate
python3 predict.py --team-a "India" --team-b "Sri Lanka" \
    --phase chase --target 222 --balls 90 --runs 180 --wickets 4 \
    --chasing "India"
```

Pre-trained artifacts (`artifacts/cricket-t20/`) are included so you can query
the model without retraining. `diag_calibration.py` regenerates the
calibration plot from `calibration_report.json`.

## The graduation gate

`rater/graduation.py` encodes the bar every model must clear on the held-out
time-ordered test set before it is considered usable:

1. Brier score beats the constant base-rate predictor,
2. every 5pp probability bucket with ≥200 samples is within 5pp of the actual
   win rate,
3. at least 2,000 test samples.

The gate is sport-agnostic — the `rater/` package is a template for adding
tennis, soccer, MLB, NFL, or other sports (see `rater/engines/cricket/` as the
reference implementation).

## License

MIT — see `LICENSE`.

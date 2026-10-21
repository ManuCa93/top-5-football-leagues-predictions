"""Backtest del sistema nuovo: ogni venerdì alle 12:00 schedine sulle partite fino a martedì,
con probabilità dal mercato pre-partita e quote Snai stimate; si chiude con i risultati veri.

  python -m predictor.backtest            # DEV (2020/21-2024/25)
  python -m predictor.backtest --holdout  # verdetto su 2025/26-2026/27
"""
import argparse
from datetime import timedelta

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .data import load_matches
from .odds import add_market, outcome_wins
from .slips import SLIP_TYPES, build_slips, candidate_legs
from .walkforward import DEV_SEASONS, HOLDOUT_SEASONS, fridays

MARKET_CACHE = DATA_DIR / "market_rows.pkl"
WINDOW_DAYS = 4


def matches_with_market(matches=None):
    matches = load_matches(refresh=False) if matches is None else matches
    cached = pd.read_pickle(MARKET_CACHE) if MARKET_CACHE.exists() else pd.DataFrame(columns=["match_id"])
    todo = matches[~matches.match_id.isin(cached.match_id)]
    if len(todo):
        new = add_market(todo)
        cols = ["match_id"] + [c for c in new.columns if c.startswith(("p_", "q_", "lambda_"))]
        cached = pd.concat([cached, new[cols]], ignore_index=True)
        cached.to_pickle(MARKET_CACHE)
    return matches.merge(cached, on="match_id", how="left")


def run(df, weekly_budget=5.0, slip_types=SLIP_TYPES):
    rows = []
    start, end = df.date.min().to_pydatetime(), df.date.max().to_pydatetime()
    for friday in fridays(start - timedelta(days=7), end):
        week = df[(df.date >= friday) & (df.date < friday + timedelta(days=WINDOW_DAYS))].dropna(subset=["p_1"])
        if week.empty:
            continue
        recs = [{**r, "label": f"{r['home_team']} - {r['away_team']}"} for r in week.to_dict("records")]
        results = {r["match_id"]: (r["home_goals"], r["away_goals"]) for r in recs}
        for s in build_slips(candidate_legs(recs), weekly_budget, slip_types=slip_types):
            won = all(outcome_wins(l["market"], *results[l["match_id"]]) for l in s["legs"])
            rows.append({"friday": friday, "type": s["type"], "prob": s["prob"], "odds": s["odds"], "ev": s["ev"],
                         "stake": s["stake"], "won": won, "pnl": s["stake"] * (s["odds"] - 1) if won else -s["stake"],
                         "markets": " ".join(l["market"] for l in s["legs"])})
    return pd.DataFrame(rows)


def summary(ledger, label):
    by = ledger.groupby("type").agg(schedine=("pnl", "size"), prob_media=("prob", "mean"), vinte=("won", "mean"),
                                    quota_mediana=("odds", "median"), ev_atteso=("ev", "mean"),
                                    puntato=("stake", "sum"), pnl=("pnl", "sum"))
    by["ROI"] = by.pnl / by.puntato
    tot_roi = ledger.pnl.sum() / ledger.stake.sum()
    # intervallo di confidenza del ROI con bootstrap per settimana
    weeks = ledger.groupby("friday").agg(stake=("stake", "sum"), pnl=("pnl", "sum"))
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        s = weeks.sample(len(weeks), replace=True, random_state=rng.integers(1 << 31))
        boot.append(s.pnl.sum() / s.stake.sum())
    lo, hi = np.percentile(boot, [5, 95])
    print(f"\n=== {label}: {weeks.shape[0]} settimane, puntato {ledger.stake.sum():.0f}EUR, "
          f"P&L {ledger.pnl.sum():+.0f}EUR, ROI {tot_roi:+.1%} (90%: {lo:+.1%} / {hi:+.1%}), "
          f"EV atteso medio {np.average(ledger.ev, weights=ledger.stake):+.1%}")
    print(by.round(3).to_string())
    print("mercati più usati:", pd.Series(" ".join(ledger.markets).split()).value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    df = matches_with_market()
    seasons = HOLDOUT_SEASONS if args.holdout else DEV_SEASONS
    part = df[df.season.isin(seasons)]
    ledger = run(part)
    ledger.to_csv(DATA_DIR / f"backtest_{'holdout' if args.holdout else 'dev'}.csv", index=False)
    summary(ledger, "HOLDOUT" if args.holdout else "DEV")


if __name__ == "__main__":
    main()

"""Simulazione walk-forward del sistema di marzo 2026 (predictor/legacy.py).

Ogni venerdì alle 12:00 il sistema vede solo le partite già giocate, riaddestra il modello
e gioca le partite fino al martedì successivo, con le quote medie pre-partita (proxy Snai).

  python -m predictor.backtest_legacy predict   # lento: previsioni settimana per settimana
  python -m predictor.backtest_legacy simulate  # veloce: schedine, puntate, bankroll
"""
import argparse
import pickle
import random
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import poisson

from . import legacy
from .config import DATA_DIR, season_of
from .data import load_matches
from .odds import add_market, outcome_wins

OUT_DIR = DATA_DIR / "backtest_legacy"
PRED_FILE = OUT_DIR / "predictions.csv"
START = datetime(2025, 8, 15)
END = datetime(2026, 10, 16)
WINDOW_DAYS = 4  # venerdì 12:00 -> martedì 12:00


def fridays(start, end):
    d = start + timedelta(days=(4 - start.weekday()) % 7)
    while d <= end:
        yield d.replace(hour=12)
        d += timedelta(days=7)


def window_features(matches, season):
    """Feature (senza h2h) sulla finestra usata dal sistema: 2 stagioni passate + quella corrente."""
    path = OUT_DIR / f"features_{season}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    t = time.time()
    win = matches[matches.season.between(season - 2, season)].copy()
    df, X, y = legacy.build_window_features(win)
    path.write_bytes(pickle.dumps((df, X, y)))
    print(f"[features] stagione {season}: {len(df)} partite in {time.time() - t:.0f}s", flush=True)
    return df, X, y


def predict_all(start=START, end=END):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches(refresh=False)
    done = set(pd.read_csv(PRED_FILE).friday) if PRED_FILE.exists() else set()
    for friday in fridays(start, end):
        key = friday.strftime("%Y-%m-%d")
        if key in done:
            continue
        targets = matches[(matches.date >= friday) & (matches.date < friday + timedelta(days=WINDOW_DAYS))]
        if targets.empty:
            continue
        t = time.time()
        season = season_of(friday)
        df, X, y = window_features(matches, season)
        cut = (df.date < friday).values
        df_cut = df[cut].reset_index(drop=True)
        X_cut = X[cut].copy()
        X_cut[:, legacy.H2H_COLS] = legacy.h2h_for_cutoff(df_cut)
        model, scaler, selector, test_acc = legacy.train(
            X_cut, y[cut], df_cut.matchday.values, df_cut.league.values)
        fx = targets
        probs = legacy.predict(df_cut, fx, model, scaler, selector, now=friday)
        n = len(df_cut) + 1
        p_ng = []
        for _, row in fx.iterrows():
            hs = legacy.compute_advanced_stats(df_cut, row.home_team, n)
            as_ = legacy.compute_advanced_stats(df_cut, row.away_team, n)
            lh, la = legacy.calc_poisson_v23(row.home_team, row.away_team, hs['scored_overall'],
                                             hs['conceded_overall'], as_['scored_overall'], as_['conceded_overall'])
            p_ng.append(poisson.pmf(0, lh) + poisson.pmf(0, la) - poisson.pmf(0, lh) * poisson.pmf(0, la))
        out = pd.DataFrame({
            "friday": key, "match_id": fx.match_id.values, "p_away": probs[:, 0], "p_draw": probs[:, 1],
            "p_home": probs[:, 2], "p_ng": p_ng, "test_acc": test_acc, "train_rows": int(cut.sum()),
        })
        out.to_csv(PRED_FILE, mode="a", header=not PRED_FILE.exists(), index=False)
        print(f"[{key}] {len(fx)} partite, acc test dichiarata {test_acc:.3f}, {time.time() - t:.0f}s", flush=True)


def load_predictions():
    matches = load_matches(refresh=False)
    pred = pd.read_csv(PRED_FILE)
    df = pred.merge(matches, on="match_id", how="left", validate="many_to_one")
    return add_market(df)


def weekly_bets(week, staking, rng):
    fx = week.copy()
    probs = fx[["p_away", "p_draw", "p_home"]].values
    options = []
    for (_, row), pr in zip(fx.iterrows(), probs):
        pa, px, ph = pr
        lbl = f"[{row.league}] {row.home_team} vs {row.away_team}"
        for typ, prob in (('1', ph), ('X', px), ('2', pa), ('1X', ph + px), ('X2', pa + px),
                          ('GG', 1 - row.p_ng), ('NG', row.p_ng)):
            quota = row[f"q_{typ}"]
            if not np.isfinite(quota) or quota <= 1.05:
                continue
            if prob * quota > 1.05:
                options.append({'match': lbl, 'match_id': row.match_id, 'type': typ,
                                'prob': prob, 'quota': quota, 'ev': prob * quota})
    portfolio = legacy.generate_tiered_portfolio(options, rng)
    return legacy.choose_stakes(portfolio, legacy.STAKING[staking])


def settle(picks, results):
    rows = []
    for p in picks:
        won = all(outcome_wins(leg['type'], *results[leg['match_id']]) for leg in p['legs'])
        rows.append({"tier": p['tier'], "legs": len(p['legs']), "quota": p['quota'], "stake": p['stake'],
                     "won": won, "pnl": p['stake'] * (p['quota'] - 1) if won else -p['stake'],
                     "desc": " + ".join(f"{leg['match']} {leg['type']}@{leg['quota']:.2f}" for leg in p['legs'])})
    return rows


def simulate(staking="march", seeds=range(20)):
    df = load_predictions()
    results = {r.match_id: (r.home_goals, r.away_goals) for r in df.itertuples()}
    ledgers = []
    for seed in seeds:
        rng = random.Random(seed)
        for friday, week in df.groupby("friday", sort=True):
            for r in settle(weekly_bets(week, staking, rng), results):
                ledgers.append({"seed": seed, "friday": friday, **r})
    ledger = pd.DataFrame(ledgers)
    ledger.to_csv(OUT_DIR / f"ledger_{staking}.csv", index=False)
    return df, ledger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["predict", "simulate"])
    ap.add_argument("--start", default=START.strftime("%Y-%m-%d"))
    ap.add_argument("--end", default=END.strftime("%Y-%m-%d"))
    args = ap.parse_args()
    if args.step == "predict":
        predict_all(datetime.fromisoformat(args.start), datetime.fromisoformat(args.end))
    else:
        for staking in legacy.STAKING:
            simulate(staking)


if __name__ == "__main__":
    main()

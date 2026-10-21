"""Previsioni walk-forward: ogni venerdì alle 12:00 addestra sul passato e prevede i 7 giorni seguenti.

DEV (taratura di parametri, pesi e soglie): stagioni 2020/21 - 2024/25
HOLDOUT (verdetto finale, da guardare una volta sola): 2025/26 e 2026/27
"""
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .data import load_matches
from .model import GoalModel, market_implied, market_probs_from_rates
from .odds import implied_goal_rates, market_summary, score_matrix

OUT_DIR = DATA_DIR / "walkforward"
START = datetime(2020, 8, 14)
DEV_SEASONS = range(2020, 2025)
HOLDOUT_SEASONS = (2025, 2026)


def fridays(start, end):
    d = start + timedelta(days=(4 - start.weekday()) % 7)
    while d <= end:
        yield d.replace(hour=12)
        d += timedelta(days=7)


def market_table(matches):
    """Probabilità di mercato per partita, inclusa P(GG) dai gol attesi impliciti (in cache)."""
    path = OUT_DIR / "market.csv"
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["match_id"])
    todo = matches[~matches.match_id.isin(cached.match_id)].dropna(subset=["odds_1", "odds_x", "odds_2"])
    rows = []
    for r in todo.itertuples():
        m = market_implied(r)
        lh, la = implied_goal_rates(r.odds_1, r.odds_x, r.odds_2, r.odds_over25, r.odds_under25)
        m["GG"] = market_summary(score_matrix(lh, la))["GG"]
        m["NG"] = 1 - m["GG"]
        rows.append({"match_id": r.match_id, **{f"mkt_{k}": v for k, v in m.items()}})
    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        cached.to_csv(path, index=False)
    return cached


def run(configs, start=START, end=None, matches=None):
    """configs: {nome: kwargs di GoalModel}. Restituisce una riga per partita con mod_<nome>_<mercato>."""
    matches = load_matches(refresh=False) if matches is None else matches
    end = end or matches.date.max().to_pydatetime()
    out = []
    t = time.time()
    for friday in fridays(start, end):
        week = matches[(matches.date >= friday) & (matches.date < friday + timedelta(days=7))]
        if week.empty:
            continue
        past = matches[matches.date < friday]
        models = {name: GoalModel(**kw).fit(past, friday) for name, kw in configs.items()}
        for r in week.itertuples():
            row = {"match_id": r.match_id, "friday": friday}
            for name, m in models.items():
                if r.league not in m.params:
                    continue
                lh, la = m.rates(r.league, r.home_team, r.away_team)
                row.update({f"mod_{name}_{k}": v for k, v in market_probs_from_rates(lh, la).items()})
                row[f"mod_{name}_lh"], row[f"mod_{name}_la"] = lh, la
            out.append(row)
    print(f"walk-forward: {len(out)} partite in {time.time() - t:.0f}s", flush=True)
    preds = pd.DataFrame(out).merge(matches, on="match_id")
    return preds.merge(market_table(matches), on="match_id", how="left")


def outcomes(df):
    hg, ag = df.home_goals.values, df.away_goals.values
    return {"1": hg > ag, "X": hg == ag, "2": hg < ag, "1X": hg >= ag, "X2": hg <= ag,
            "O25": hg + ag > 2.5, "U25": hg + ag < 2.5, "GG": (hg > 0) & (ag > 0), "NG": (hg == 0) | (ag == 0)}


def logloss(df, prefix, groups):
    """Log loss medio per gruppo di esiti mutuamente esclusivi, es. ("1","X","2")."""
    y = outcomes(df)
    p = np.zeros(len(df))
    for k in groups:
        col = df[f"{prefix}_{k}"].values
        p = np.where(y[k], col, p)
    p = np.clip(p, 1e-6, 1)
    return float(-np.mean(np.log(p)))

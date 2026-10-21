"""Storia delle squadre, calcolata per ogni partita usando SOLO le partite precedenti.

Ogni statistica è "prima del calcio d'inizio": shift(1) sulle partite della squadra,
così il risultato della partita stessa non entra mai nelle sue feature (vedi tests/test_leakage.py).
"""
import numpy as np
import pandas as pd

ELO_K = 20
ELO_HOME = 60  # vantaggio campo in punti Elo
ELO_CARRY = 0.8  # a inizio stagione l'Elo regredisce verso 1500


def team_long(matches):
    """Una riga per (partita, squadra) con gol fatti/subiti e punti."""
    base = matches[["match_id", "league", "season", "date", "home_team", "away_team",
                    "home_goals", "away_goals", "home_sot", "away_sot", "home_xg", "away_xg"]]
    h = base.rename(columns={"home_team": "team", "away_team": "opp", "home_goals": "gf", "away_goals": "ga",
                             "home_sot": "sotf", "away_sot": "sota", "home_xg": "xgf", "away_xg": "xga"})
    a = base.rename(columns={"away_team": "team", "home_team": "opp", "away_goals": "gf", "home_goals": "ga",
                             "away_sot": "sotf", "home_sot": "sota", "away_xg": "xgf", "home_xg": "xga"})
    h["is_home"], a["is_home"] = 1, 0
    long = pd.concat([h, a], ignore_index=True).sort_values(["team", "date"]).reset_index(drop=True)
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga], [3, 1], 0).astype(float)
    long.loc[long.gf.isna(), "pts"] = np.nan  # partite non ancora giocate
    return long


def team_stats_before(long):
    """Statistiche di ogni squadra prima di ogni partita (shift(1): niente partita corrente)."""
    g = long.groupby("team", group_keys=False)
    out = long[["match_id", "team", "is_home", "date", "season"]].copy()
    for col in ("pts", "gf", "ga", "sotf", "sota"):
        out[f"{col}_last5"] = g[col].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    gs = long.groupby(["team", "season"], group_keys=False)
    out["ppg_season"] = gs["pts"].transform(lambda s: s.shift(1).expanding().mean())
    out["played_season"] = gs.cumcount()
    out["xgf_season"] = gs["xgf"].transform(lambda s: s.shift(1).expanding().mean())
    out["xga_season"] = gs["xga"].transform(lambda s: s.shift(1).expanding().mean())
    out["rest_days"] = g["date"].transform(lambda s: s.diff().dt.days).clip(upper=14)
    return out


def elo_before(matches, k=ELO_K, home=ELO_HOME, carry=ELO_CARRY):
    """Elo di casa e trasferta prima di ogni partita, aggiornato in ordine di data."""
    elo, last_season = {}, {}
    eh, ea = np.empty(len(matches)), np.empty(len(matches))
    for i, r in enumerate(matches.itertuples()):
        for t in (r.home_team, r.away_team):
            if t not in elo:
                elo[t] = 1450.0  # le neopromosse partono sotto la media
            elif last_season[t] != r.season:
                elo[t] = 1500 + carry * (elo[t] - 1500)
            last_season[t] = r.season
        eh[i], ea[i] = elo[r.home_team], elo[r.away_team]
        if pd.isna(r.home_goals):
            continue
        exp_h = 1 / (1 + 10 ** ((ea[i] - eh[i] - home) / 400))
        res = 1.0 if r.home_goals > r.away_goals else 0.5 if r.home_goals == r.away_goals else 0.0
        margin = np.log1p(abs(r.home_goals - r.away_goals)) + 1
        elo[r.home_team] += k * margin * (res - exp_h)
        elo[r.away_team] -= k * margin * (res - exp_h)
    return pd.DataFrame({"match_id": matches.match_id.values, "elo_home": eh, "elo_away": ea}), elo


def current_elo(matches):
    return elo_before(matches.sort_values("date"))[1]


def match_features(matches):
    """Una riga per partita con le statistiche pre-partita di casa (h_) e trasferta (a_)."""
    matches = matches.sort_values("date").reset_index(drop=True)
    stats = team_stats_before(team_long(matches))
    cols = [c for c in stats.columns if c not in ("match_id", "team", "is_home", "date", "season")]
    home = stats[stats.is_home == 1][["match_id"] + cols].add_prefix("h_").rename(columns={"h_match_id": "match_id"})
    away = stats[stats.is_home == 0][["match_id"] + cols].add_prefix("a_").rename(columns={"a_match_id": "match_id"})
    elo, _ = elo_before(matches)
    return matches[["match_id"]].merge(home, on="match_id").merge(away, on="match_id").merge(elo, on="match_id")

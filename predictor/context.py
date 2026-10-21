"""Storia delle squadre da mostrare nel report: classifica, forma, gol, xG, scontri diretti,
più il parere del modello dei gol. Solo contesto: le decisioni usano le probabilità di mercato,
perché nei test la storia non le migliora (vedi README)."""
import numpy as np
import pandas as pd

from .features import match_features
from .model import GoalModel, market_probs_from_rates


def standings(played, league, season):
    g = played[(played.league == league) & (played.season == season)]
    rows = {}
    for r in g.itertuples():
        for team, gf, ga in ((r.home_team, r.home_goals, r.away_goals), (r.away_team, r.away_goals, r.home_goals)):
            s = rows.setdefault(team, {"team": team, "pg": 0, "pt": 0, "gf": 0, "gs": 0})
            s["pg"] += 1
            s["gf"] += gf
            s["gs"] += ga
            s["pt"] += 3 if gf > ga else 1 if gf == ga else 0
    table = pd.DataFrame(rows.values())
    if table.empty:
        return table
    table["dr"] = table.gf - table.gs
    table = table.sort_values(["pt", "dr", "gf"], ascending=False).reset_index(drop=True)
    table["pos"] = table.index + 1
    return table.set_index("team")


def form_string(played, team, n=5):
    g = played[(played.home_team == team) | (played.away_team == team)].tail(n)
    out = []
    for r in g.itertuples():
        gf, ga = (r.home_goals, r.away_goals) if r.home_team == team else (r.away_goals, r.home_goals)
        out.append("V" if gf > ga else "N" if gf == ga else "P")
    return "".join(out)


def head_to_head(played, home, away, n=3):
    g = played[((played.home_team == home) & (played.away_team == away)) |
               ((played.home_team == away) & (played.away_team == home))].tail(n)
    return ", ".join(f"{r.home_team} {r.home_goals}-{r.away_goals} {r.away_team}" for r in g.itertuples())


def fixture_context(played, fixtures, cutoff):
    """Una riga di contesto per ogni partita in programma (usa solo partite giocate prima di cutoff)."""
    played = played[played.date < cutoff]
    feats = match_features(pd.concat([played, fixtures], ignore_index=True))
    feats = feats[feats.match_id.isin(fixtures.match_id)].set_index("match_id")
    model = GoalModel(half_life_days=365, target="sot").fit(played, cutoff)
    tables = {}
    rows = []
    for r in fixtures.itertuples():
        key = (r.league, r.season)
        if key not in tables:
            tables[key] = standings(played, *key)
        t = tables[key]
        f = feats.loc[r.match_id]
        row = {"match_id": r.match_id}
        for side, team in (("h", r.home_team), ("a", r.away_team)):
            row[f"{side}_pos"] = int(t.loc[team, "pos"]) if team in t.index else None
            row[f"{side}_pt"] = int(t.loc[team, "pt"]) if team in t.index else None
            row[f"{side}_form"] = form_string(played, team)
            row[f"{side}_gf5"], row[f"{side}_ga5"] = f[f"{side}_gf_last5"], f[f"{side}_ga_last5"]
            row[f"{side}_xgf"], row[f"{side}_xga"] = f[f"{side}_xgf_season"], f[f"{side}_xga_season"]
            row[f"{side}_rest"] = f[f"{side}_rest_days"]
        row["elo_diff"] = f.elo_home - f.elo_away
        row["h2h"] = head_to_head(played, r.home_team, r.away_team)
        if r.league in model.params:
            probs = market_probs_from_rates(*model.rates(r.league, r.home_team, r.away_team))
            row.update({f"mod_{k}": probs[k] for k in ("1", "X", "2")})
        else:
            row.update({f"mod_{k}": np.nan for k in ("1", "X", "2")})
        rows.append(row)
    return pd.DataFrame(rows)

"""Statistiche avanzate da Understat (dal 2014/15): xG fatti e subiti, xG senza rigori, PPDA (passaggi concessi
per azione difensiva: più è basso, più la squadra pressa), passaggi profondi e punti attesi. Una richiesta per
campionato e stagione, in cache in data/understat/.
"""
import json
import time
from datetime import datetime

import pandas as pd
import requests

from .config import DATA_DIR, season_of

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

URL = "https://understat.com/getLeagueData/{league}/{season}"
HEADERS = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}
LEAGUES = {"SA": "Serie_A", "PL": "EPL", "PD": "La_liga", "BL1": "Bundesliga", "FL1": "Ligue_1"}
FIRST_SEASON = 2014
UNDERSTAT_DIR = DATA_DIR / "understat"
MAX_AGE_HOURS = 6


def _season_teams(league, season):
    path = UNDERSTAT_DIR / f"{league}_{season}.json"
    current = season == season_of(datetime.now())
    fresh = path.exists() and (not current or time.time() - path.stat().st_mtime < MAX_AGE_HOURS * 3600)
    if not fresh:
        try:
            time.sleep(2)
            r = requests.get(URL.format(league=LEAGUES[league], season=season), headers=HEADERS, timeout=60)
            r.raise_for_status()
            UNDERSTAT_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(r.json()["teams"]))
        except (requests.RequestException, ValueError, KeyError) as e:
            if not path.exists():
                print(f"[WARN] Understat {league} {season}: {e}")
                return {}
    return json.loads(path.read_text())


def load_team_matches(seasons=None):
    """Una riga per (squadra Understat, partita) con le statistiche avanzate."""
    seasons = seasons or range(FIRST_SEASON, season_of(datetime.now()) + 1)
    rows = []
    for league in LEAGUES:
        for season in seasons:
            for team in _season_teams(league, season).values():
                for h in team["history"]:
                    rows.append({"league": league, "season": season, "us_team": team["title"],
                                 "date": pd.Timestamp(h["date"]), "is_home": int(h["h_a"] == "h"),
                                 "xgf": h["xG"], "xga": h["xGA"], "npxgf": h["npxG"], "npxga": h["npxGA"],
                                 "ppda": h["ppda"]["att"] / max(h["ppda"]["def"], 1),
                                 "ppda_allowed": h["ppda_allowed"]["att"] / max(h["ppda_allowed"]["def"], 1),
                                 "deepf": h["deep"], "deepa": h["deep_allowed"], "xpts_us": h["xpts"]})
    return pd.DataFrame(rows)


def team_names(us, base):
    """Nome Understat -> nome football-data: le squadre che giocano in casa lo stesso giorno nella stessa lega."""
    h = us[us.is_home == 1].assign(day=lambda d: d.date.dt.normalize())
    b = base[["league", "date", "home_team"]].assign(day=base.date.dt.normalize())
    j = h.merge(b, on=["league", "day"])
    counts = j.groupby(["us_team", "home_team"]).size().reset_index(name="n")
    best = counts.sort_values("n", ascending=False).drop_duplicates("us_team")
    return dict(zip(best.us_team, best.home_team))


def advanced_stats(base):
    """Statistiche Understat per (match_id, is_home) delle partite di football-data: stessa squadra, stesso
    campo, data più vicina entro un giorno (gli orari sono in fusi diversi)."""
    us = load_team_matches()
    us["team"] = us.us_team.map(team_names(us, base))
    us = us.dropna(subset=["team"]).sort_values("date").drop(columns=["season", "us_team"])
    side = pd.concat([
        base[["match_id", "league", "date"]].assign(team=base.home_team, is_home=1),
        base[["match_id", "league", "date"]].assign(team=base.away_team, is_home=0)]).sort_values("date")
    j = pd.merge_asof(side, us.rename(columns={"date": "us_date"}), left_on="date", right_on="us_date",
                      by=["league", "team", "is_home"], tolerance=pd.Timedelta(days=1), direction="nearest")
    cols = ["xgf", "xga", "npxgf", "npxga", "ppda", "ppda_allowed", "deepf", "deepa", "xpts_us"]
    return j[["match_id", "is_home"] + cols]

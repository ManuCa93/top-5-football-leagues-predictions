"""Dati da Sofascore: partite di campionato, assenti e minuti giocati, calendario di ogni squadra in tutte
le competizioni (coppe ed Europa). Tutto in cache in data/sofascore/ e scaricato con calma (una richiesta
alla volta), così un aggiornamento settimanale costa poche decine di richieste.

  python -m predictor.sofascore   # scarica o aggiorna tutto
"""
import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from .config import DATA_DIR, FIRST_SEASON, LEAGUES, season_of

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

API = "https://www.sofascore.com/api/v1/"
HEADERS = {"User-Agent": "Mozilla/5.0"}
SLEEP = 1.5  # Sofascore risponde 403 se si va più veloci
SOFA_DIR = DATA_DIR / "sofascore"
LINEUPS_DIR = SOFA_DIR / "lineups"
TEAMS_DIR = SOFA_DIR / "teams"
UNIQUE_TOURNAMENT = {"SA": 23, "PL": 17, "PD": 8, "BL1": 35, "FL1": 34}
INJURY_REASON = 1  # reason 1 = infortunio; 3 e 13 = squalifiche e altro


def _get(path, tries=6):
    for i in range(tries):
        time.sleep(SLEEP)
        try:
            r = requests.get(API + path, headers=HEADERS, timeout=30)
        except requests.RequestException:
            time.sleep(10 * (i + 1))
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 200:
            return r.json()
        wait = min(60 * 2 ** i, 600)  # 403/429: troppo veloci, aspetto sempre di più
        print(f"[sofascore] {r.status_code}, attendo {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"Sofascore non risponde: {path}")


def _season_label(season):
    return f"{season % 100:02d}/{(season + 1) % 100:02d}"


def _slim_event(e):
    return {"id": e["id"], "ts": e["startTimestamp"], "status": e["status"]["type"],
            "home_id": e["homeTeam"]["id"], "home": e["homeTeam"]["name"],
            "away_id": e["awayTeam"]["id"], "away": e["awayTeam"]["name"],
            "home_score": e.get("homeScore", {}).get("current"), "away_score": e.get("awayScore", {}).get("current"),
            "tournament": e["tournament"].get("uniqueTournament", {}).get("id"),
            "tournament_name": e["tournament"].get("uniqueTournament", {}).get("name", e["tournament"]["name"])}


def league_events(league, season, refresh=False):
    """Tutte le partite di una stagione di campionato (giornata per giornata)."""
    path = SOFA_DIR / f"events_{league}_{season}.json"
    current = season == season_of(datetime.now())
    if path.exists() and not (refresh and current):
        return json.loads(path.read_text())
    ut = UNIQUE_TOURNAMENT[league]
    seasons = {s["year"]: s["id"] for s in _get(f"unique-tournament/{ut}/seasons")["seasons"]}
    sid = seasons.get(_season_label(season))
    events = []
    for rnd in range(1, 39):
        data = _get(f"unique-tournament/{ut}/season/{sid}/events/round/{rnd}") if sid else None
        if not data or not data.get("events"):
            break
        events += [_slim_event(e) for e in data["events"]]
    SOFA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(events))
    return events


def _slim_lineups(data):
    out = {}
    for side in ("home", "away"):
        s = data.get(side, {})
        out[side] = {
            "players": [{"id": p["player"]["id"], "pos": p["player"].get("position"), "sub": p.get("substitute"),
                         "min": (p.get("statistics") or {}).get("minutesPlayed", 0),
                         "rating": (p.get("statistics") or {}).get("rating")} for p in s.get("players", [])],
            "missing": [{"id": m["player"]["id"], "pos": m["player"].get("position"), "type": m.get("type"),
                         "reason": m.get("reason")} for m in s.get("missingPlayers", [])],
        }
    return out


def event_lineups(event_id):
    path = LINEUPS_DIR / f"{event_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = _get(f"event/{event_id}/lineups")
    slim = _slim_lineups(data) if data else {"home": {"players": [], "missing": []},
                                             "away": {"players": [], "missing": []}}
    LINEUPS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(slim))
    return slim


def team_events(team_id, since, refresh=False):
    """Partite di una squadra in tutte le competizioni dalla data `since` (amichevoli escluse)."""
    path = TEAMS_DIR / f"{team_id}.json"
    cached = json.loads(path.read_text()) if path.exists() else []
    if cached and not refresh:
        return cached
    known = {e["id"] for e in cached if e["status"] == "finished"}
    new, page = [], 0
    while True:
        data = _get(f"team/{team_id}/events/last/{page}")
        if not data:
            break
        evs = [_slim_event(e) for e in data["events"]]
        new += evs
        oldest = min(e["ts"] for e in evs)
        if not data.get("hasNextPage") or oldest < since.timestamp() or (cached and all(e["id"] in known for e in evs)):
            break
        page += 1
    nxt = _get(f"team/{team_id}/events/next/0")
    new += [_slim_event(e) for e in (nxt or {}).get("events", [])]
    merged = {e["id"]: e for e in cached}
    merged.update({e["id"]: e for e in new if e["ts"] >= since.timestamp()})
    events = sorted((e for e in merged.values() if "Friendl" not in (e["tournament_name"] or "")),
                    key=lambda e: e["ts"])
    TEAMS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(events))
    return events


def download(refresh=False, lineups=True, verbose=True):
    """Scarica tutto quello che manca: liste delle partite, calendari delle squadre, poi le formazioni
    dalla stagione più recente. refresh=True aggiorna la stagione in corso e i calendari."""
    seasons = range(FIRST_SEASON, season_of(datetime.now()) + 1)
    t0 = time.time()
    events = {(lg, s): league_events(lg, s, refresh=refresh) for lg in LEAGUES for s in seasons}
    teams = {e["home_id"] for evs in events.values() for e in evs}
    if verbose:
        print(f"liste partite pronte, {len(teams)} squadre ({time.time() - t0:.0f}s)", flush=True)
    since = datetime(FIRST_SEASON, 7, 1)
    for i, team in enumerate(sorted(teams)):
        team_events(team, since, refresh=refresh)
        if verbose and i % 20 == 0:
            print(f"calendari squadre: {i}/{len(teams)} ({time.time() - t0:.0f}s)", flush=True)
    for (league, season), evs in sorted(events.items(), key=lambda kv: -kv[0][1]) if lineups else []:
        todo = [e for e in evs if e["status"] == "finished" and not (LINEUPS_DIR / f"{e['id']}.json").exists()]
        if verbose and todo:
            print(f"{league} {season}: {len(todo)} formazioni da scaricare ({time.time() - t0:.0f}s)", flush=True)
        for e in todo:
            event_lineups(e["id"])
    if verbose:
        print(f"fatto in {time.time() - t0:.0f}s", flush=True)


def load_league_events():
    rows = []
    for path in SOFA_DIR.glob("events_*.json"):
        _, league, season = path.stem.split("_")
        for e in json.loads(path.read_text()):
            rows.append({**e, "league": league, "season": int(season)})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert("Europe/London").dt.tz_localize(None)
    return df


def load_team_events():
    rows = []
    for path in TEAMS_DIR.glob("*.json"):
        for e in json.loads(path.read_text()):
            rows.append(e)
    df = pd.DataFrame(rows).drop_duplicates("id")
    df["date"] = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert("Europe/London").dt.tz_localize(None)
    return df


if __name__ == "__main__":
    download(refresh=True)

"""Download e normalizzazione dei dati di football-data.co.uk (risultati + quote)."""
import io
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from .config import (CURRENT_SEASON_MAX_AGE_HOURS, FIRST_SEASON, LEAGUES,
                     RAW_DIR, season_of)

try:
    # su Windows il certificato di football-data.co.uk si verifica solo con lo store di sistema
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

BASE_URL = "https://football-data.co.uk/mmz4281/{code}/{div}.csv"
FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"

# colonne del CSV -> nomi interni (quelle mancanti diventano NaN)
COLUMNS = {
    "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "home_goals", "FTAG": "away_goals",
    "HS": "home_shots", "AS": "away_shots", "HST": "home_sot", "AST": "away_sot",
    "HxG": "home_xg", "AxG": "away_xg",
    # quote medie pre-partita (rilevate ~venerdì per il weekend, ~martedì per il turno infrasettimanale)
    "AvgH": "odds_1", "AvgD": "odds_x", "AvgA": "odds_2",
    "Avg>2.5": "odds_over25", "Avg<2.5": "odds_under25",
    "MaxH": "max_1", "MaxD": "max_x", "MaxA": "max_2",
    # quote di chiusura: solo per analisi, MAI per decidere le giocate
    "AvgCH": "close_1", "AvgCD": "close_x", "AvgCA": "close_2",
}


# fino al 2018/19 le medie erano di Betbrain, con altri nomi di colonna
FALLBACK_COLUMNS = {"AvgH": "BbAvH", "AvgD": "BbAvD", "AvgA": "BbAvA", "Avg>2.5": "BbAv>2.5", "Avg<2.5": "BbAv<2.5",
                    "MaxH": "BbMxH", "MaxD": "BbMxD", "MaxA": "BbMxA"}


def season_code(season):
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def _read_csv_bytes(content):
    for enc in ("utf-8-sig", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("encoding CSV non riconosciuto")


def _download(url, dest):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)


def _is_stale(path, is_current):
    if not path.exists():
        return True
    if not is_current:
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600
    return age_h > CURRENT_SEASON_MAX_AGE_HOURS


def load_league_season(league, season, current_season, refresh=True):
    div = LEAGUES[league]["div"]
    path = RAW_DIR / f"{div}_{season_code(season)}.csv"
    if not path.exists() or (refresh and _is_stale(path, season == current_season)):
        try:
            _download(BASE_URL.format(code=season_code(season), div=div), path)
        except requests.RequestException as e:
            if not path.exists():
                raise
            print(f"[WARN] download fallito per {div} {season} ({e}), uso la cache")
    raw = _read_csv_bytes(path.read_bytes())
    return _standardize(raw, league, season)


def _standardize(raw, league, season):
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"])
    raw = raw.rename(columns={old: new for new, old in FALLBACK_COLUMNS.items() if new not in raw})
    df = pd.DataFrame({new: raw[old] if old in raw else np.nan for old, new in COLUMNS.items()})
    time_str = raw["Time"].fillna("15:00") if "Time" in raw else "15:00"
    stamp = raw["Date"] + " " + time_str
    four_digits = stamp.str.match(r"\d{2}/\d{2}/\d{4}")  # i CSV fino al 2017/18 hanno l'anno a due cifre
    df["date"] = pd.to_datetime(stamp.where(four_digits), format="%d/%m/%Y %H:%M").fillna(
        pd.to_datetime(stamp.where(~four_digits), format="%d/%m/%y %H:%M"))
    df.insert(0, "season", season)
    df.insert(0, "league", league)
    return df


def load_matches(seasons=None, leagues=None, refresh=True, today=None):
    """Tutte le partite giocate, ordinate per data, con matchday approssimato."""
    today = today or datetime.now()
    current = season_of(today)
    seasons = seasons or range(FIRST_SEASON, current + 1)
    leagues = leagues or list(LEAGUES)
    frames = []
    for league in leagues:
        for season in seasons:
            try:
                frames.append(load_league_season(league, season, current, refresh))
            except requests.HTTPError as e:
                print(f"[WARN] {league} {season}: {e}")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["home_goals", "away_goals"])
    df[["home_goals", "away_goals"]] = df[["home_goals", "away_goals"]].astype(int)
    df = df.sort_values(["date", "league", "home_team"]).reset_index(drop=True)
    df["matchday"] = approx_matchday(df)
    df["match_id"] = df.league + "_" + df.date.dt.strftime("%Y%m%d") + "_" + df.home_team + "_" + df.away_team
    return df


def approx_matchday(df):
    """I CSV non hanno la giornata: la stimo come n-esima partita stagionale della squadra
    che ne ha giocate di più (i recuperi spostano di poco la stima)."""
    md = pd.Series(0, index=df.index)
    for _, g in df.groupby(["league", "season"]):
        played = {}
        for idx, row in g.sort_values("date").iterrows():
            h = played.get(row.home_team, 0) + 1
            a = played.get(row.away_team, 0) + 1
            played[row.home_team], played[row.away_team] = h, a
            md[idx] = max(h, a)
    return md


def load_fixtures():
    """Prossime partite dei 5 campionati con le quote attuali."""
    resp = requests.get(FIXTURES_URL, timeout=30)
    resp.raise_for_status()
    raw = _read_csv_bytes(resp.content)
    div_to_league = {v["div"]: k for k, v in LEAGUES.items()}
    raw = raw[raw["Div"].isin(div_to_league)]
    frames = []
    for div, g in raw.groupby("Div"):
        league = div_to_league[div]
        f = _standardize(g, league, None)
        f["season"] = f["date"].map(season_of)
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)

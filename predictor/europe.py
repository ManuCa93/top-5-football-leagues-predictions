"""Partite europee (Champions, Europa League, Conference, preliminari) da openfootball, dataset aperto su GitHub:
servono per riposo e carico "campionato e non". Le coppe nazionali non ci sono.

I file arrivano fino all'ultima stagione pubblicata; quelli passati sono in cache in data/europe/.
"""
import difflib
import re
import unicodedata
from datetime import datetime

import pandas as pd
import requests

from .config import DATA_DIR, season_of

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

URL = "https://raw.githubusercontent.com/openfootball/champions-league/master/{season}/{file}"
FILES = ("clq.txt", "cl.txt", "elq.txt", "el.txt", "confq.txt", "conf.txt")
EUROPE_DIR = DATA_DIR / "europe"
FIRST_SEASON = 2011
COUNTRY_LEAGUE = {"ITA": "SA", "ENG": "PL", "ESP": "PD", "GER": "BL1", "FRA": "FL1"}

DATE_RE = re.compile(r"^\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
MATCH_RE = re.compile(r"^\s+(?:(\d{1,2}[:.]\d{2})\s+)?(.+?)\s+\(([A-Z]{3})\)\s+v\s+(.+?)\s+\(([A-Z]{3})\)")
MONTHS = {m: i for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                                      "Nov", "Dec"), 1)}

# nomi openfootball -> football-data.co.uk che il confronto automatico sbaglia
ALIASES = {
    "atletico madrid": "Ath Madrid", "club atletico de madrid": "Ath Madrid", "athletic club": "Ath Bilbao",
    "athletic bilbao": "Ath Bilbao", "paris saint germain": "Paris SG", "manchester city": "Man City",
    "manchester united": "Man United", "internazionale": "Inter", "internazionale milano": "Inter",
    "inter": "Inter", "milan": "Milan", "bayer 04 leverkusen": "Leverkusen", "bayer leverkusen": "Leverkusen",
    "borussia monchengladbach": "M'gladbach", "eintracht frankfurt": "Ein Frankfurt",
    "real sociedad de futbol": "Sociedad", "real sociedad": "Sociedad", "real betis balompie": "Betis",
    "real betis": "Betis", "rc celta de vigo": "Celta", "celta vigo": "Celta",
    "wolverhampton wanderers": "Wolves", "tottenham hotspur": "Tottenham", "newcastle united": "Newcastle",
    "west ham united": "West Ham", "leicester city": "Leicester", "brighton hove albion": "Brighton",
    "olympique de marseille": "Marseille", "olympique lyonnais": "Lyon", "as saint etienne": "St Etienne",
    "saint etienne": "St Etienne", "rb leipzig": "RB Leipzig", "bayern munchen": "Bayern Munich",
    "1 fc union berlin": "Union Berlin", "union berlin": "Union Berlin", "tsg hoffenheim": "Hoffenheim",
    "tsg 1899 hoffenheim": "Hoffenheim", "vfl wolfsburg": "Wolfsburg", "1 fsv mainz 05": "Mainz",
    "1 fc koln": "FC Koln", "fc koln": "FC Koln", "sport club freiburg": "Freiburg", "sc freiburg": "Freiburg",
    "ogc nice": "Nice", "stade rennais": "Rennes", "losc lille": "Lille", "lille osc": "Lille",
    "rc lens": "Lens", "racing club lens": "Lens", "stade brestois 29": "Brest", "villarreal": "Villarreal", "sevilla": "Sevilla",
    "valencia": "Valencia", "girona": "Girona", "deportivo alaves": "Alaves", "rcd espanyol": "Espanol",
    "ss lazio": "Lazio", "ssc napoli": "Napoli", "atalanta bc": "Atalanta", "acf fiorentina": "Fiorentina",
    "bologna 1909": "Bologna", "aston villa": "Aston Villa", "nottingham forest": "Nott'm Forest",
    "crystal palace": "Crystal Palace", "vfb stuttgart": "Stuttgart", "werder bremen": "Werder Bremen",
    "schalke 04": "Schalke 04", "borussia dortmund": "Dortmund", "fc girondins de bordeaux": "Bordeaux",
    "montpellier hsc": "Montpellier", "fc nantes": "Nantes", "as monaco": "Monaco",
}
DROP = re.compile(r"\b(fc|cf|ac|as|ss|ssc|sc|afc|club|de|futbol|calcio|1909|1907|1913|1899|1900|04|05|"
                  r"vfl|vfb|tsg|rcd|rc|cd|ud|osc|ogc|hsc|bc|acf|sv|1)\b")


def _norm(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def _fetch(season, file):
    code = f"{season}-{(season + 1) % 100:02d}"
    path = EUROPE_DIR / code / file
    current = season >= season_of(datetime.now()) - 1
    if path.exists() and not current:
        return path.read_text(encoding="utf-8")
    try:
        r = requests.get(URL.format(season=code, file=file), timeout=30)
    except requests.RequestException:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    text = r.text if r.status_code == 200 else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def parse(text, season, competition):
    rows, day = [], None
    for line in text.splitlines():
        m = DATE_RE.match(line)
        if m:  # l'anno spesso manca e i gironi ripartono da settembre: lo ricavo dalla stagione
            mon, d = MONTHS[m.group(1)], int(m.group(2))
            day = datetime(season if mon >= 7 else season + 1, mon, d)
            continue
        m = MATCH_RE.match(line)
        if m and day is not None:
            for name, country in ((m.group(2), m.group(3)), (m.group(4), m.group(5))):
                rows.append({"date": day, "team_of": name.strip(), "country": country, "competition": competition})
    return rows


def load_european_matches(seasons=None):
    seasons = seasons or range(FIRST_SEASON, season_of(datetime.now()) + 1)
    rows = []
    for season in seasons:
        for file in FILES:
            rows += parse(_fetch(season, file), season, file.split(".")[0])
    return pd.DataFrame(rows)


def map_teams(euro, base):
    """Nome openfootball -> nome football-data, solo per le squadre dei 5 campionati."""
    euro = euro[euro.country.isin(COUNTRY_LEAGUE)]
    mapping = {}
    for (name, country), g in euro.groupby(["team_of", "country"]):
        league = COUNTRY_LEAGUE[country]
        season = season_of(g.date.min())
        pool = base[(base.league == league) & base.season.between(season - 1, season)]
        teams = sorted(set(pool.home_team))
        if not teams:
            continue
        n = _norm(name)
        target = ALIASES.get(n) or ALIASES.get(DROP.sub(" ", n).strip().replace("  ", " "))
        if target is None:
            short = " ".join(DROP.sub(" ", n).split())
            scores = {t: difflib.SequenceMatcher(None, short, _norm(t)).ratio() for t in teams}
            best = max(scores, key=scores.get)
            target = best if scores[best] >= 0.6 else None
        if target in teams:
            mapping[(name, country)] = target
    return mapping


def european_dates(base):
    """Una riga per (squadra football-data, data) di ogni partita europea."""
    euro = load_european_matches()
    mapping = map_teams(euro, base)
    euro["team"] = [mapping.get((n, c)) for n, c in zip(euro.team_of, euro.country)]
    return euro.dropna(subset=["team"])[["team", "date", "competition"]].drop_duplicates()

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = ROOT / "reports"

# codice interno -> divisione football-data.co.uk
LEAGUES = {
    "SA": {"div": "I1", "name": "Serie A"},
    "PL": {"div": "E0", "name": "Premier League"},
    "PD": {"div": "SP1", "name": "La Liga"},
    "BL1": {"div": "D1", "name": "Bundesliga"},
    "FL1": {"div": "F1", "name": "Ligue 1"},
}

# prima stagione scaricata (anno di inizio): da 2019/20 i CSV hanno le colonne Avg*
FIRST_SEASON = 2019

# Una stagione inizia a luglio: luglio 2026 -> stagione 2026 (2026/27)
def season_of(date):
    return date.year if date.month >= 7 else date.year - 1

# Le stagioni passate non cambiano più; quella in corso va riscaricata se più vecchia di così
CURRENT_SEASON_MAX_AGE_HOURS = 6

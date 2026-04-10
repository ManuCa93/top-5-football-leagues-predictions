import sys
import io
import os
import time
import traceback
import warnings
import random
import itertools
import json
import threading
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)

import flet as ft  # LIBRERIA PER L'INTERFACCIA GRAFICA

warnings.filterwarnings("ignore")

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except:
    pass

DEBUG_MODE = True

# VARIABILI GLOBALI MODIFICABILI DALLA GUI
DEBUG_MATCHDAYS = {'SA': 23, 'PL': 24, 'PD': 22, 'BL1': 20, 'FL1': 20}
DYNAMIC_ODDS = {}

def get_odds_mapping():
    # Se abbiamo inserito quote dall'interfaccia, diamo priorità a quelle
    if DYNAMIC_ODDS:
        return DYNAMIC_ODDS
        
    return {
        'SA': [
            {'home': 'Verona', 'away': 'Pisa', '1': 2.40, 'X': 3.00, '2': 3.25, '1X': 1.33, '2X': 1.55, 'GG': 1.90, 'NG': 1.80},
            {'home': 'Genoa', 'away': 'Napoli', '1': 4.40, 'X': 3.15, '2': 1.95, '1X': 1.80, '2X': 1.19, 'GG': 2.10, 'NG': 1.65},
            {'home': 'Fiorentina', 'away': 'Torino', '1': 1.70, 'X': 3.70, '2': 4.90, '1X': 1.16, '2X': 2.10, 'GG': 1.77, 'NG': 1.95},
            {'home': 'Bologna', 'away': 'Parma', '1': 1.60, 'X': 4.00, '2': 5.50, '1X': 1.13, '2X': 2.25, 'GG': 1.90, 'NG': 1.80},
            {'home': 'Lecce', 'away': 'Udinese', '1': 3.00, 'X': 2.85, '2': 2.70, '1X': 1.45, '2X': 1.35, 'GG': 2.15, 'NG': 1.63},
            {'home': 'Sassuolo', 'away': 'Inter', '1': 6.50, 'X': 4.60, '2': 1.45, '1X': 2.65, '2X': 1.10, 'GG': 1.90, 'NG': 1.80},
            {'home': 'Juventus', 'away': 'Lazio', '1': 1.45, 'X': 4.25, '2': 7.25, '1X': 1.08, '2X': 2.65, 'GG': 2.25, 'NG': 1.57},
            {'home': 'Atalanta', 'away': 'Cremonese', '1': 1.35, 'X': 4.90, '2': 8.00, '1X': 1.06, '2X': 3.05, 'GG': 1.95, 'NG': 1.77},
            {'home': 'Roma', 'away': 'Cagliari', '1': 1.45, 'X': 4.25, '2': 7.25, '1X': 1.08, '2X': 2.65, 'GG': 2.15, 'NG': 1.60},
        ],
        'PL': [
            {'home': 'Leeds', 'away': 'Nottingham Forest', '1': 2.15, 'X': 3.25, '2': 3.40, '1X': 1.30, '2X': 1.67, 'GG': 1.77, 'NG': 1.93},
            {'home': 'Man United', 'away': 'Tottenham', '1': 1.55, 'X': 4.50, '2': 4.75, '1X': 1.16, '2X': 2.35, 'GG': 1.50, 'NG': 2.40},
            {'home': 'Fulham', 'away': 'Everton', '1': 2.05, 'X': 3.20, '2': 3.65, '1X': 1.25, '2X': 1.70, 'GG': 1.87, 'NG': 1.85},
            {'home': 'Burnley', 'away': 'West Ham', '1': 3.25, 'X': 3.45, '2': 2.10, '1X': 1.67, '2X': 1.30, 'GG': 1.67, 'NG': 2.10},
            {'home': 'Wolves', 'away': 'Chelsea', '1': 4.75, 'X': 4.00, '2': 1.65, '1X': 2.15, '2X': 1.16, 'GG': 1.63, 'NG': 2.15},
            {'home': 'Bournemouth', 'away': 'Aston Villa', '1': 2.65, 'X': 3.55, '2': 2.40, '1X': 1.50, '2X': 1.43, 'GG': 1.50, 'NG': 2.40},
            {'home': 'Arsenal', 'away': 'Sunderland', '1': 1.20, 'X': 6.25, '2': 13.0, '1X': 1.09, '2X': 4.10, 'GG': 2.50, 'NG': 1.47},
            {'home': 'Newcastle', 'away': 'Brentford', '1': 2.00, 'X': 3.45, '2': 3.50, '1X': 1.27, '2X': 1.75, 'GG': 1.53, 'NG': 2.35},
            {'home': 'Brighton', 'away': 'Crystal Palace', '1': 1.92, 'X': 3.55, '2': 3.70, '1X': 1.24, '2X': 1.80, 'GG': 1.60, 'NG': 2.15},
            {'home': 'Liverpool', 'away': 'Man City', '1': 2.30, 'X': 3.65, '2': 2.80, '1X': 1.40, '2X': 1.57, 'GG': 1.45, 'NG': 2.55},
        ],
        'BL1': [
            {'home': 'Union Berlin', 'away': 'Eintracht Frankfurt', '1': 2.05, 'X': 3.45, '2': 3.55, '1X': 1.28, '2X': 1.73, 'GG': 1.67, 'NG': 2.10},
            {'home': 'Freiburg', 'away': 'Werder Bremen', '1': 1.80, 'X': 3.70, '2': 4.10, '1X': 1.20, '2X': 1.93, 'GG': 1.63, 'NG': 2.15},
            {'home': 'Wolfsburg', 'away': 'Dortmund', '1': 4.25, 'X': 4.00, '2': 1.70, '1X': 2.05, '2X': 1.19, 'GG': 1.50, 'NG': 2.40},
            {'home': 'Heidenheim', 'away': 'Hamburg', '1': 2.85, 'X': 3.40, '2': 2.35, '1X': 1.55, '2X': 1.40, 'GG': 1.60, 'NG': 2.20},
            {'home': 'Mainz', 'away': 'Augsburg', '1': 2.00, 'X': 3.50, '2': 3.45, '1X': 1.27, '2X': 1.75, 'GG': 1.65, 'NG': 2.15},
            {'home': 'St. Pauli', 'away': 'Stoccarda', '1': 4.25, 'X': 3.65, '2': 1.77, '1X': 1.95, '2X': 1.19, 'GG': 1.70, 'NG': 2.00},
            {'home': 'Monchengladbach', 'away': 'Leverkusen', '1': 3.45, 'X': 3.65, '2': 1.95, '1X': 1.77, '2X': 1.27, 'GG': 1.50, 'NG': 2.40},
            {'home': 'Colonia', 'away': 'Lipsia', '1': 3.20, 'X': 3.75, '2': 2.05, '1X': 1.70, '2X': 1.30, 'GG': 1.40, 'NG': 2.70},
            {'home': 'Bayern', 'away': 'Hoffenheim', '1': 1.27, 'X': 6.75, '2': 8.00, '1X': 1.05, '2X': 3.55, 'GG': 1.50, 'NG': 2.40},
        ],
        'FL1': [
            {'home': 'Metz', 'away': 'Lille', '1': 5.50, 'X': 4.25, '2': 1.55, '1X': 2.40, '2X': 1.13, 'GG': 1.77, 'NG': 1.95},
            {'home': 'Lens', 'away': 'Rennes', '1': 1.75, 'X': 3.75, '2': 4.25, '1X': 1.19, '2X': 2.00, 'GG': 1.50, 'NG': 2.40},
            {'home': 'Brest', 'away': 'Lorient', '1': 2.20, 'X': 3.05, '2': 3.45, '1X': 1.27, '2X': 1.60, 'GG': 1.75, 'NG': 1.97},
            {'home': 'Nantes', 'away': 'Lione', '1': 4.90, 'X': 3.75, '2': 1.65, '1X': 2.15, '2X': 1.15, 'GG': 1.80, 'NG': 1.90},
            {'home': 'Nizza', 'away': 'Monaco', '1': 3.05, 'X': 3.70, '2': 2.10, '1X': 1.65, '2X': 1.32, 'GG': 1.40, 'NG': 2.65},
            {'home': 'Le Havre', 'away': 'Strasburgo', '1': 3.90, 'X': 3.45, '2': 1.87, '1X': 1.80, '2X': 1.21, 'GG': 1.73, 'NG': 1.97},
            {'home': 'Auxerre', 'away': 'Paris FC', '1': 2.40, 'X': 3.10, '2': 2.95, '1X': 1.35, '2X': 1.50, 'GG': 1.77, 'NG': 1.90},
            {'home': 'Angers', 'away': 'Tolosa', '1': 3.40, 'X': 3.05, '2': 2.20, '1X': 1.60, '2X': 1.27, 'GG': 1.92, 'NG': 1.77},
            {'home': 'PSG', 'away': 'Marsiglia', '1': 1.40, 'X': 5.00, '2': 6.25, '1X': 1.08, '2X': 2.75, 'GG': 1.57, 'NG': 2.25},
        ],
        'PD': [
            {'home': 'Celta', 'away': 'Osasuna', '1': 1.97, 'X': 3.30, '2': 4.10, '1X': 1.22, '2X': 1.80, 'GG': 1.90, 'NG': 1.80},
            {'home': 'Vallecano', 'away': 'Real Oviedo', '1': 1.75, 'X': 3.40, '2': 4.75, '1X': 1.16, '2X': 2.00, 'GG': 2.00, 'NG': 1.73},
            {'home': 'Barcellona', 'away': 'Maiorca', '1': 1.14, 'X': 8.50, '2': 14.0, '1X': 1.04, '2X': 5.00, 'GG': 1.87, 'NG': 1.85},
            {'home': 'Siviglia', 'away': 'Girona', '1': 2.05, 'X': 3.30, '2': 3.55, '1X': 1.27, '2X': 1.70, 'GG': 1.73, 'NG': 2.00},
            {'home': 'Real Sociedad', 'away': 'Elche', '1': 1.63, 'X': 3.90, '2': 5.00, '1X': 1.14, '2X': 2.15, 'GG': 1.75, 'NG': 1.97},
            {'home': 'Alaves', 'away': 'Getafe', '1': 2.25, 'X': 2.80, '2': 3.70, '1X': 1.24, '2X': 1.60, 'GG': 2.50, 'NG': 1.47},
            {'home': 'Athletic Bilbao', 'away': 'Levante', '1': 1.60, 'X': 3.85, '2': 5.50, '1X': 1.12, '2X': 2.20, 'GG': 1.87, 'NG': 1.83},
            {'home': 'Atletico Madrid', 'away': 'Betis', '1': 1.45, 'X': 4.75, '2': 6.00, '1X': 1.10, '2X': 2.60, 'GG': 1.75, 'NG': 1.97},
            {'home': 'Valencia', 'away': 'Real Madrid', '1': 5.50, 'X': 4.50, '2': 1.50, '1X': 2.45, '2X': 1.12, 'GG': 1.60, 'NG': 2.20},
            {'home': 'Villarreal', 'away': 'Espanyol', '1': 1.70, 'X': 4.10, '2': 4.25, '1X': 1.19, '2X': 2.05, 'GG': 1.67, 'NG': 2.10},
        ],
    }

# =======================
# LOGGING SYSTEM
# =======================
def get_next_sunday_log_filename():
    today = datetime.now()
    days_until_sunday = (6 - today.weekday()) % 7
    if days_until_sunday == 0 and today.weekday() != 6:
        days_until_sunday = 7
    next_sunday = today + timedelta(days=days_until_sunday)
    month_name = next_sunday.strftime("%B").lower()
    day = next_sunday.strftime("%d").lstrip('0')
    return f"predictor_{month_name}_{day}.log"

LOG_FILE = get_next_sunday_log_filename()

def log_msg(msg, level="INFO"):
    log_entry = f"{msg}"
    print(log_entry, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    except:
        pass

# =======================
# CONFIGURAZIONE
# =======================
API_KEY = "f65cdbbd6d67477883d3f468626a19cf"
SEASONS_TRAIN = [2024, 2023]
SEASONS_CURRENT = [2025]
PREDICT_SEASON = 2025
SEED = 42
np.random.seed(SEED)
BUDGET_TOTALE = 100.0

LEAGUES_CONFIG = [
    {'code': 'SA', 'id': 2019, 'name': 'Serie A'},
    {'code': 'PL', 'id': 2021, 'name': 'Premier League'},
    {'code': 'PD', 'id': 2014, 'name': 'La Liga'},
    {'code': 'BL1', 'id': 2002, 'name': 'Bundesliga'},
    {'code': 'FL1', 'id': 2015, 'name': 'Ligue 1'}      
]

# =======================
# TEAM NAME NORMALIZATION
# =======================
TEAM_ALIASES = {
    'FC Internazionale Milano': 'Inter', 'Internazionale': 'Inter', 'Inter Milan': 'Inter',
    'AC Milan': 'Milan', 'Milan': 'Milan', 'Associazione Calcio Milan': 'Milan',
    'SSC Napoli': 'Napoli', 'Napoli': 'Napoli', 'Juventus FC': 'Juventus', 'Juventus': 'Juventus',
    'AS Roma': 'Roma', 'Roma': 'Roma', 'SS Lazio': 'Lazio', 'Lazio': 'Lazio',
    'Atalanta Bergamasca Calcio': 'Atalanta', 'Atalanta': 'Atalanta', 'Atalanta BC': 'Atalanta',
    'Hellas Verona FC': 'Verona', 'Verona': 'Verona', 'US Lecce': 'Lecce', 'Lecce': 'Lecce',
    'Cagliari Calcio': 'Cagliari', 'Cagliari': 'Cagliari', 'Udinese Calcio': 'Udinese', 'Udinese': 'Udinese',
    'Genoa CFC': 'Genoa', 'Genoa': 'Genoa', 'Como 1907': 'Como', 'Como': 'Como',
    'US Cremonese': 'Cremonese', 'Cremonese': 'Cremonese', 'AC Pisa 1909': 'Pisa', 'Pisa': 'Pisa',
    'Parma Calcio 1913': 'Parma', 'Parma': 'Parma', 'Torino FC': 'Torino', 'Torino': 'Torino',
    'US Sassuolo Calcio': 'Sassuolo', 'Sassuolo': 'Sassuolo', 'ACF Fiorentina': 'Fiorentina',
    'Fiorentina': 'Fiorentina', 'Bologna FC 1909': 'Bologna', 'Bologna': 'Bologna',
    'Empoli FC': 'Empoli', 'Empoli': 'Empoli', 'Monza': 'Monza', 'AC Monza': 'Monza',
    'Venezia FC': 'Venezia', 'Venezia': 'Venezia',
    'Manchester City FC': 'Man City', 'Man City': 'Man City', 'Manchester United FC': 'Man United',
    'Man United': 'Man United', 'Arsenal FC': 'Arsenal', 'Arsenal': 'Arsenal',
    'Liverpool FC': 'Liverpool', 'Liverpool': 'Liverpool', 'Chelsea FC': 'Chelsea', 'Chelsea': 'Chelsea',
    'Tottenham Hotspur': 'Tottenham', 'Tottenham': 'Tottenham', 'Newcastle United': 'Newcastle',
    'Newcastle': 'Newcastle', 'Aston Villa FC': 'Aston Villa', 'Aston Villa': 'Aston Villa',
    'Brighton and Hove Albion': 'Brighton', 'Brighton': 'Brighton', 'West Ham United': 'West Ham',
    'West Ham': 'West Ham', 'Fulham FC': 'Fulham', 'Fulham': 'Fulham', 'Crystal Palace': 'Crystal Palace',
    'Bournemouth AFC': 'Bournemouth', 'Bournemouth': 'Bournemouth', 'Brentford FC': 'Brentford',
    'Brentford': 'Brentford', 'Nottingham Forest': 'Nottingham', 'Nottingham Forest FC': 'Nottingham',
    'Everton FC': 'Everton', 'Everton': 'Everton', 'Wolverhampton Wanderers': 'Wolves',
    'Wolves': 'Wolves', 'Wolverhampton Wanderers FC': 'Wolves', 'Leicester City': 'Leicester',
    'Leicester': 'Leicester', 'Southampton FC': 'Southampton', 'Southampton': 'Southampton',
    'Ipswich Town': 'Ipswich', 'Ipswich': 'Ipswich', 'Leeds United': 'Leeds', 'Leeds': 'Leeds',
    'Sunderland AFC': 'Sunderland', 'Sunderland': 'Sunderland', 'Burnley FC': 'Burnley', 'Burnley': 'Burnley',
    'Real Madrid CF': 'Real Madrid', 'Real Madrid': 'Real Madrid', 'FC Barcelona': 'Barcelona',
    'Barcelona': 'Barcelona', 'Atlético Madrid': 'Atletico', 'Atletico Madrid': 'Atletico',
    'Atleti': 'Atletico', 'Real Betis Balompié': 'Real Betis', 'Real Betis': 'Real Betis',
    'Betis': 'Real Betis', 'Villarreal CF': 'Villarreal', 'Villarreal': 'Villarreal',
    'Athletic Club': 'Athletic Club', 'Athletic Bilbao': 'Athletic Club', 'Athletic': 'Athletic Club',
    'Real Sociedad': 'Real Sociedad', 'Real Sociedad de Fútbol': 'Real Sociedad',
    'Girona FC': 'Girona', 'Girona': 'Girona', 'Getafe CF': 'Getafe', 'Getafe': 'Getafe',
    'Sevilla FC': 'Sevilla', 'Sevilla': 'Sevilla', 'Valencia CF': 'Valencia', 'Valencia': 'Valencia',
    'Rayo Vallecano': 'Rayo', 'Vallecano': 'Rayo', 
    'RCD Espanyol de Barcelona': 'Espanyol', 'RCD Espanyol': 'Espanyol', 'Espanyol': 'Espanyol',
    'Real Oviedo': 'Oviedo', 'Oviedo': 'Oviedo', 'RCD Mallorca': 'Mallorca', 'Mallorca': 'Mallorca',
    'Elche CF': 'Elche', 'Elche': 'Elche', 'Celta Vigo': 'Celta', 'Celta de Vigo': 'Celta',
    'Celta': 'Celta', 'Levante UD': 'Levante', 'Levante': 'Levante', 'CA Osasuna': 'Osasuna',
    'Osasuna': 'Osasuna', 'Las Palmas': 'Las Palmas', 'CD Leganés': 'Leganes',
    'Deportivo Alavés': 'Alaves', 'Alavés': 'Alaves', 'Alaves': 'Alaves',
    'Leganes': 'Leganes', 'Real Valladolid': 'Valladolid', 'Valladolid': 'Valladolid',
    'FC Bayern München': 'Bayern', 'Bayern Munich': 'Bayern', 'Bayern': 'Bayern',
    'Borussia Dortmund': 'Dortmund', 'BVB': 'Dortmund',
    'Bayer 04 Leverkusen': 'Leverkusen', 'Bayer Leverkusen': 'Leverkusen',
    'RB Leipzig': 'Leipzig', 'RasenBallsport Leipzig': 'Leipzig',
    'VfB Stuttgart': 'Stuttgart', 'Stuttgart': 'Stuttgart',
    'Eintracht Frankfurt': 'Frankfurt', 'Eintracht': 'Frankfurt',
    'VfL Wolfsburg': 'Wolfsburg', 'Borussia Mönchengladbach': 'Gladbach', 'Mönchengladbach': 'Gladbach',
    'SC Freiburg': 'Freiburg', 'TSG 1899 Hoffenheim': 'Hoffenheim',
    '1. FC Union Berlin': 'Union Berlin', 'Union Berlin': 'Union Berlin',
    '1. FSV Mainz 05': 'Mainz', 'Mainz 05': 'Mainz', 'Mainz': 'Mainz',
    'FC Augsburg': 'Augsburg', 'SV Werder Bremen': 'Werder Bremen', 'Werder Bremen': 'Werder Bremen',
    '1. FC Heidenheim 1846': 'Heidenheim', 'Heidenheim': 'Heidenheim',
    'VfL Bochum 1848': 'Bochum', 'Bochum': 'Bochum',
    'FC St. Pauli': 'St. Pauli', 'St. Pauli': 'St. Pauli',
    '1. FC Köln': 'Koln', '1. FC Koeln': 'Koln', 'Köln': 'Koln',
    'Hamburger SV': 'Hamburger SV',
    'Holstein Kiel': 'Holstein Kiel', 'Kiel': 'Holstein Kiel',
    'Paris Saint-Germain FC': 'PSG', 'Paris Saint-Germain': 'PSG', 'Paris SG': 'PSG',
    'Paris FC': 'Paris FC', 'PFC': 'Paris FC',
    'Olympique de Marseille': 'Marseille', 'Marseille': 'Marseille',
    'AS Monaco FC': 'Monaco', 'AS Monaco': 'Monaco', 'Monaco': 'Monaco',
    'Olympique Lyonnais': 'Lyon', 'Lyon': 'Lyon',
    'LOSC Lille': 'Lille', 'Lille': 'Lille',
    'RC Lens': 'Lens', 'Racing Club de Lens': 'Lens',
    'Stade Rennais FC': 'Rennes', 'Rennes': 'Rennes',
    'OGC Nice': 'Nice', 'Nice': 'Nice',
    'Stade de Reims': 'Reims', 'Reims': 'Reims',
    'Stade Brestois 29': 'Brest', 'Brest': 'Brest',
    'Montpellier HSC': 'Montpellier', 'Montpellier': 'Montpellier',
    'RC Strasbourg Alsace': 'Strasbourg', 'Strasbourg': 'Strasbourg',
    'FC Nantes': 'Nantes', 'Toulouse FC': 'Toulouse', 'Toulouse': 'Toulouse',
    'Le Havre AC': 'Le Havre', 'Le Havre': 'Le Havre',
    'AJ Auxerre': 'Auxerre', 'Angers SCO': 'Angers', 'Angers': 'Angers',
    'AS Saint-Étienne': 'Saint-Etienne', 'Saint-Etienne': 'Saint-Etienne',
    'FC Lorient': 'Lorient', 'Lorient': 'Lorient',
    'FC Metz': 'Metz', 'Metz': 'Metz'
}

def normalize_team_name(name):
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    best_match = None
    best_ratio = 0
    for key, value in TEAM_ALIASES.items():
        ratio = SequenceMatcher(None, name.lower(), key.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = value
    if best_ratio > 0.7:
        return best_match
    return name

TOP_TEAMS = [
    'Inter', 'Juventus', 'Milan', 'Napoli', 'Atalanta', 'Roma', 'Lazio',
    'Man City', 'Arsenal', 'Liverpool', 'Chelsea', 'Tottenham', 'Man United', 'Newcastle',
    'Real Madrid', 'Barcelona', 'Atletico', 'Girona', 'Athletic Club',
    'Bayern', 'Leverkusen', 'Dortmund', 'Leipzig', 
    'PSG', 'Monaco', 'Marseille', 'Lille'
]

WEAK_ATTACKS = [
    'Lecce', 'Cagliari', 'Empoli', 'Monza', 'Venezia', 'Genoa', 'Verona', 'Udinese', 'Como',
    'Southampton', 'Ipswich', 'Leicester', 'Everton', 'Wolves', 'Crystal Palace',
    'Leganes', 'Valladolid', 'Espanyol', 'Getafe', 'Las Palmas', 'Valencia',
    'Bochum', 'Holstein Kiel', 'St. Pauli', 'Union Berlin',
    'Angers', 'Le Havre', 'Saint-Etienne', 'Montpellier', 'Auxerre'
]

RATE_LIMIT_DELAY = 2.0
API_CALL_COUNT = 0
LAST_API_CALL_TIME = None

def respect_rate_limit():
    global API_CALL_COUNT, LAST_API_CALL_TIME
    current_time = time.time()
    if LAST_API_CALL_TIME is not None and (current_time - LAST_API_CALL_TIME) > 60:
        API_CALL_COUNT = 0
    if API_CALL_COUNT >= 8:
        wait_time = 61 - (current_time - LAST_API_CALL_TIME)
        if wait_time > 0:
            log_msg(f"[WAIT] Rate Limit Protection... attendo {wait_time:.1f}s")
            time.sleep(wait_time)
            API_CALL_COUNT = 0
            LAST_API_CALL_TIME = time.time()
            return
    if LAST_API_CALL_TIME is not None:
        elapsed = current_time - LAST_API_CALL_TIME
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
    LAST_API_CALL_TIME = time.time()
    API_CALL_COUNT += 1

def fetch_matches(comp_id, season, league_name):
    respect_rate_limit()
    url = f"https://api.football-data.org/v4/competitions/{comp_id}/matches"
    headers = {"X-Auth-Token": API_KEY}
    params = {"season": season}
    log_msg(f"[API] Scaricamento {league_name} ({season})...")
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        matches = resp.json().get("matches", [])
        return matches
    except Exception as e:
        log_msg(f"[ERROR] Fallito scaricamento {league_name}: {e}", level="ERROR")
        return []

def fetch_odds_global(df_matches):
    log_msg("[QUOTE] Assegnazione quote ai match (MATCHING INTELLIGENTE V2)...")
    odds_list = []
    odds_mapping = get_odds_mapping()
    default_odds = {'1': 1.01, 'X': 1.01, '2': 1.01, '1X': 1.01, '2X': 1.01, 'GG': 1.01, 'NG': 1.01}

    try:
        matches_matched = 0
        for idx, row in df_matches.iterrows():
            league = row['league']
            home_api = row['home_team']
            away_api = row['away_team']
            league_odds_data = odds_mapping.get(league, [])
            found_odds = None
            
            h_api_clean = home_api.lower().replace("fc", "").replace("1.", "").strip()
            a_api_clean = away_api.lower().replace("fc", "").replace("1.", "").strip()

            for stored_match in league_odds_data:
                if 'home' not in stored_match or 'away' not in stored_match:
                    continue
                h_store = stored_match['home'].lower().replace("fc", "").strip()
                a_store = stored_match['away'].lower().replace("fc", "").strip()
                h_ratio = SequenceMatcher(None, h_api_clean, h_store).ratio()
                a_ratio = SequenceMatcher(None, a_api_clean, a_store).ratio()
                h_sub = h_store in h_api_clean or h_api_clean in h_store
                a_sub = a_store in a_api_clean or a_api_clean in a_store

                if (h_ratio > 0.7 or h_sub) and (a_ratio > 0.7 or a_sub):
                    found_odds = stored_match
                    break 
            
            if found_odds:
                odds_list.append(found_odds)
                matches_matched += 1
            else:
                log_msg(f"[WARN] Quote NON trovate per: {home_api} - {away_api} ({league}). Uso default.", level="WARNING")
                odds_list.append(default_odds)

        log_msg(f"[OK] Quote assegnate correttamente a {matches_matched}/{len(df_matches)} match.")
        return odds_list

    except Exception as e:
        log_msg(f"[ERROR] Errore assegnazione quote: {e}", level="ERROR")
        traceback.print_exc()
        return []

def parse_match(m, s, league_code):
    try:
        home = normalize_team_name(m["homeTeam"]["name"])
        away = normalize_team_name(m["awayTeam"]["name"])
        return {
            "league": league_code,
            "season": s,
            "matchday": m.get("matchday", 0),
            "home_team": home,
            "away_team": away,
            "date": m["utcDate"],
            "home_goals": m["score"]["fullTime"]["home"],
            "away_goals": m["score"]["fullTime"]["away"],
        }
    except Exception as e:
        log_msg(f"[ERROR] Errore parsing match: {e}", level="ERROR")
        return None

def build_global_dataset(leagues, seasons_train, seasons_curr, debug_mds):
    log_msg("\n[1] COSTRUZIONE DATASET GLOBALE (CACHE SYSTEM)...")
    log_msg("-" * 80)
    
    CACHE_FILE = "history_cache.csv"
    all_train_rows = []
    
    if os.path.exists(CACHE_FILE):
        log_msg(f"[CACHE] Trovato file '{CACHE_FILE}'. Caricamento dati storici da locale...")
        try:
            df_train = pd.read_csv(CACHE_FILE)
            df_train["date"] = pd.to_datetime(df_train["date"])
            log_msg(f"[CACHE] Caricati {len(df_train)} match storici dal file.")
        except Exception as e:
            log_msg(f"[ERROR] Errore lettura cache ({e}). Riscarico tutto.", level="ERROR")
            df_train = pd.DataFrame()
    else:
        df_train = pd.DataFrame()

    if df_train.empty:
        log_msg("[API] Scaricamento stagioni passate (TRAIN) da API...")
        for league in leagues:
            l_code = league['code']
            l_id = league['id']
            for s in seasons_train:
                matches = fetch_matches(l_id, s, league['name'])
                for m in matches:
                    if m["status"] in ["FINISHED", "LIVE"]:
                        parsed = parse_match(m, s, l_code)
                        if parsed:
                            all_train_rows.append(parsed)
        
        df_train = pd.DataFrame(all_train_rows)
        if not df_train.empty:
            try:
                df_train.to_csv(CACHE_FILE, index=False)
                log_msg(f"[CACHE] Salvato file '{CACHE_FILE}' con {len(df_train)} match.")
            except Exception as e:
                log_msg(f"[WARN] Impossibile salvare cache: {e}", level="WARNING")

    log_msg("[API] Scaricamento stagione CORRENTE (2025) per dati aggiornati...")
    all_curr_rows = []
    try:
        for league in leagues:
            l_code = league['code']
            l_id = league['id']
            current_md = debug_mds.get(l_code, 10)
            
            for s in seasons_curr:
                matches = fetch_matches(l_id, s, league['name'])
                count_curr = 0
                for m in matches:
                    if m["status"] in ["FINISHED", "LIVE"]:
                        if m.get("matchday", 0) <= current_md:
                            parsed = parse_match(m, s, l_code)
                            if parsed:
                                all_curr_rows.append(parsed)
                                count_curr += 1
                log_msg(f" -> {league['name']}: {count_curr} partite correnti scaricate.")
    except Exception as e:
        log_msg(f"[ERROR] Errore scaricamento current: {e}", level="ERROR")

    df_curr = pd.DataFrame(all_curr_rows)

    if not df_train.empty and not df_curr.empty:
        df_final = pd.concat([df_train, df_curr], ignore_index=True)
    elif not df_train.empty:
        df_final = df_train
    else:
        df_final = df_curr

    if not df_final.empty:
        df_final["date"] = pd.to_datetime(df_final["date"])
        df_final = df_final.sort_values('date').reset_index(drop=True)

    log_msg(f"[OK] TOTALE GLOBALE: {len(df_final)} partite pronte (Storico + Corrente).\n")
    return df_final

def compute_elo(df, k=20):
    try:
        teams = list(set(df["home_team"]).union(set(df["away_team"])))
        elo = {t: 1500 for t in teams}
        home_elo_list, away_elo_list = [], []
        df = df.sort_values('date').reset_index(drop=True)
        for _, row in df.iterrows():
            home, away = row["home_team"], row["away_team"]
            hg, ag = row["home_goals"], row["away_goals"]
            home_elo_list.append(elo[home])
            away_elo_list.append(elo[away])
            if hg > ag: sh, sa = 1, 0
            elif hg < ag: sh, sa = 0, 1
            else: sh, sa = 0.5, 0.5
            eh = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
            ea = 1 - eh
            elo[home] += k * (sh - eh)
            elo[away] += k * (sa - ea)
        df["elo_home"] = home_elo_list
        df["elo_away"] = away_elo_list
        return df
    except Exception as e:
        log_msg(f"[ERROR] Errore calcolo ELO: {e}", level="ERROR")
        return df

def compute_advanced_stats(df, team, idx, last_n_recent=5, last_n_all=10):
    try:
        df_prev = df[df.index < idx]
        last_h = df_prev[df_prev["home_team"] == team]
        last_a = df_prev[df_prev["away_team"] == team]
        last_h_recent = last_h.tail(last_n_recent)
        last_a_recent = last_a.tail(last_n_recent)
        last_h_all = last_h.tail(last_n_all)
        last_a_all = last_a.tail(last_n_all)
        all_recent = pd.concat([last_h_recent, last_a_recent]).sort_values('date').tail(last_n_recent)
        all_matches = pd.concat([last_h_all, last_a_all]).sort_values('date').tail(last_n_all)

        if len(all_matches) < 2:
            return {'scored_overall': 1.4, 'conceded_overall': 1.3, 'form_overall': 1.0, 'points_overall': 0.5,
                    'home_advantage': 0.0, 'trend_recent': 0.0, 'efficiency': 0.5, 'defense_rating': 1.3,
                    'consistency': 0.5, 'win_ratio': 0.33, 'streak': 0, 'h2h_record': 0.5}

        scored_all = sum([m['home_goals'] if m['home_team'] == team else m['away_goals'] for _, m in all_matches.iterrows()])
        conceded_all = sum([m['away_goals'] if m['home_team'] == team else m['home_goals'] for _, m in all_matches.iterrows()])
        points_all = 0
        for _, m in all_matches.iterrows():
            if m['home_team'] == team:
                if m['home_goals'] > m['away_goals']: points_all += 3
                elif m['home_goals'] == m['away_goals']: points_all += 1
            else:
                if m['away_goals'] > m['home_goals']: points_all += 3
                elif m['away_goals'] == m['home_goals']: points_all += 1

        goals_per_match_all = scored_all / len(all_matches)
        conceded_per_match_all = conceded_all / len(all_matches)
        form_all = points_all / (len(all_matches) * 3)

        scored_recent = sum([m['home_goals'] if m['home_team'] == team else m['away_goals'] for _, m in all_recent.iterrows()])
        conceded_recent = sum([m['away_goals'] if m['home_team'] == team else m['home_goals'] for _, m in all_recent.iterrows()])
        points_recent = 0
        for _, m in all_recent.iterrows():
            if m['home_team'] == team:
                if m['home_goals'] > m['away_goals']: points_recent += 3
                elif m['home_goals'] == m['away_goals']: points_recent += 1
            else:
                if m['away_goals'] > m['home_goals']: points_recent += 3
                elif m['away_goals'] == m['home_goals']: points_recent += 1

        home_advantage = sum([m['home_goals'] for _, m in last_h_all.iterrows()]) / len(last_h_all) if len(last_h_all) > 0 else 0.0
        trend_recent = (points_recent / len(all_recent)) - (points_all / len(all_matches)) if len(all_recent) > 0 else 0.0
        efficiency = scored_all / max(1, scored_all + conceded_all)
        defense_rating = conceded_per_match_all
        all_gf = [m['home_goals'] if m['home_team']==team else m['away_goals'] for _, m in all_matches.iterrows()]
        consistency = np.std(all_gf) if len(all_gf) > 1 else 0.5
        wins = sum(1 for _, m in all_matches.iterrows() 
                   if (m['home_team']==team and m['home_goals']>m['away_goals']) or 
                      (m['away_team']==team and m['away_goals']>m['home_goals']))
        win_ratio = wins / len(all_matches) if len(all_matches) > 0 else 0.33

        return {'scored_overall': goals_per_match_all, 'conceded_overall': conceded_per_match_all,
                'form_overall': form_all, 'points_overall': points_all, 'home_advantage': home_advantage,
                'trend_recent': trend_recent, 'efficiency': efficiency, 'defense_rating': defense_rating,
                'consistency': consistency, 'win_ratio': win_ratio, 'streak': 0, 'h2h_record': 0.5}
    except Exception as e:
        log_msg(f"[WARN] Errore advanced stats per {team}: {e}", level="WARNING")
        return {'scored_overall': 1.4, 'conceded_overall': 1.3, 'form_overall': 1.0, 'points_overall': 0.5,
                'home_advantage': 0.0, 'trend_recent': 0.0, 'efficiency': 0.5, 'defense_rating': 1.3,
                'consistency': 0.5, 'win_ratio': 0.33, 'streak': 0, 'h2h_record': 0.5}

def calculate_xg(team, df_hist, idx, is_home=True):
    try:
        df_prev = df_hist[df_hist.index < idx]
        if is_home:
            team_matches = df_prev[df_prev['home_team'] == team]
            goals = team_matches['home_goals'].values
        else:
            team_matches = df_prev[df_prev['away_team'] == team]
            goals = team_matches['away_goals'].values
        
        if len(team_matches) < 2:
            return 1.4 if is_home else 1.1
        
        avg_goals = goals.mean()
        xg = avg_goals * 0.85
        
        recent_goals = goals[-5:].mean() if len(goals) >= 5 else avg_goals
        if recent_goals > avg_goals:
            xg *= 1.1
        
        return max(0.3, min(xg, 3.5))
    except:
        return 1.4 if is_home else 1.1

def calculate_rest_days(team, df_hist, idx):
    try:
        df_prev = df_hist[df_hist.index < idx]
        home_last = df_prev[df_prev['home_team'] == team]
        away_last = df_prev[df_prev['away_team'] == team]
        
        last_date = None
        if not home_last.empty:
            last_date_h = pd.to_datetime(home_last.iloc[-1]['date'])
            if last_date is None or last_date_h > last_date:
                last_date = last_date_h
        
        if not away_last.empty:
            last_date_a = pd.to_datetime(away_last.iloc[-1]['date'])
            if last_date is None or last_date_a > last_date:
                last_date = last_date_a
        
        if last_date is None:
            return 0.0
        
        current_date = pd.to_datetime(df_hist.iloc[idx]['date']) if idx < len(df_hist) else datetime.now()
        rest_days = (current_date - last_date).days
        
        if rest_days >= 5: return 0.3
        elif rest_days >= 3: return 0.0
        elif rest_days >= 1: return -0.2
        else: return -0.5
    except:
        return 0.0

def calculate_h2h(home, away, df_hist):
    try:
        h2h = df_hist[
            ((df_hist['home_team'] == home) & (df_hist['away_team'] == away)) |
            ((df_hist['home_team'] == away) & (df_hist['away_team'] == home))
        ]
        if len(h2h) == 0: return 0.0, 1.5, 1.3
        h2h_recent = h2h.tail(10)
        home_matches = h2h_recent[h2h_recent['home_team'] == home]
        away_matches = h2h_recent[h2h_recent['away_team'] == home]
        
        h_wins = len(home_matches[home_matches['home_goals'] > home_matches['away_goals']])
        h_wins += len(away_matches[away_matches['away_goals'] > away_matches['home_goals']])
        h_gf = home_matches['home_goals'].sum() + away_matches['away_goals'].sum()
        h_ga = home_matches['away_goals'].sum() + away_matches['home_goals'].sum()
        h_matches = len(home_matches) + len(away_matches)
        
        if h_matches == 0: return 0.0, 1.5, 1.3
        h2h_advantage = (h_wins / h_matches) - 0.33
        h2h_gf_avg = h_gf / h_matches if h_matches > 0 else 1.5
        h2h_ga_avg = h_ga / h_matches if h_matches > 0 else 1.3
        return max(-0.4, min(h2h_advantage, 0.4)), h2h_gf_avg, h2h_ga_avg
    except:
        return 0.0, 1.5, 1.3

def calculate_momentum_decay(team, df_hist, idx, is_home=True, decay_rate=0.8):
    try:
        df_prev = df_hist[df_hist.index < idx]
        if is_home: team_matches = df_prev[df_prev['home_team'] == team].tail(10)
        else: team_matches = df_prev[df_prev['away_team'] == team].tail(10)
        
        if len(team_matches) == 0: return 0.0
        
        points_weighted = []
        weights = []
        for i, (_, m) in enumerate(reversed(team_matches.iterrows())):
            weight = (decay_rate ** i)
            weights.append(weight)
            
            if is_home: gf, ga = m['home_goals'], m['away_goals']
            else: gf, ga = m['away_goals'], m['home_goals']
            
            if gf > ga: points = 3
            elif gf == ga: points = 1
            else: points = 0
            points_weighted.append(points * weight)
        
        max_possible = 3 * sum(weights)
        actual_points = sum(points_weighted)
        momentum = (actual_points / max_possible) - 0.5
        return max(-0.5, min(momentum, 0.5))
    except:
        return 0.0

def build_features_v26_enhanced(df):
    log_msg("[2] CALCOLO FEATURES V26 ENHANCED (19 → 27 FEATURES)...")
    try:
        df = compute_elo(df)
        X, y = [], []
        
        for idx, row in df.iterrows():
            try:
                if pd.isna(row.get("home_goals")): continue
                h_stats = compute_advanced_stats(df, row["home_team"], idx)
                a_stats = compute_advanced_stats(df, row["away_team"], idx)
                h_xg = calculate_xg(row["home_team"], df, idx, is_home=True)
                a_xg = calculate_xg(row["away_team"], df, idx, is_home=False)
                h_rest = calculate_rest_days(row["home_team"], df, idx)
                a_rest = calculate_rest_days(row["away_team"], df, idx)
                h2h_adv, h2h_gf, h2h_ga = calculate_h2h(row["home_team"], row["away_team"], df)
                h_momentum = calculate_momentum_decay(row["home_team"], df, idx, is_home=True)
                a_momentum = calculate_momentum_decay(row["away_team"], df, idx, is_home=False)
                
                feats = [
                    row["elo_home"], row["elo_away"],
                    h_stats['scored_overall'], h_stats['conceded_overall'],
                    a_stats['scored_overall'], a_stats['conceded_overall'],
                    h_stats['form_overall'], a_stats['form_overall'],
                    row["elo_home"] - row["elo_away"],
                    h_stats['scored_overall'] * 0.6 + a_stats['conceded_overall'] * 0.4,
                    a_stats['scored_overall'] * 0.6 + h_stats['conceded_overall'] * 0.4,
                    h_stats['home_advantage'], a_stats['home_advantage'],
                    h_stats['trend_recent'], a_stats['trend_recent'],
                    h_stats['efficiency'], a_stats['efficiency'],
                    h_stats['defense_rating'], a_stats['defense_rating'],
                    h_xg, a_xg, h_rest, a_rest, h2h_adv, h2h_gf, h2h_ga, h_momentum, a_momentum,
                ]
                X.append(feats)
                if row["home_goals"] > row["away_goals"]: y.append(2)
                elif row["home_goals"] < row["away_goals"]: y.append(0)
                else: y.append(1)
            except Exception:
                continue
        
        log_msg(f"[OK] Training Set Creato: {len(X)} campioni con 27 features (V26).")
        return np.array(X), np.array(y), df
    except Exception as e:
        log_msg(f"[ERROR] Errore build_features_v26: {e}", level="ERROR")
        return np.array([]), np.array([]), df

def train_model_v26_optimized(X, y):
    log_msg("\n[3] AI TRAINING (V26: ROBUST SCALING + FEATURE SELECTION + CALIBRATION)...")
    try:
        if len(X) == 0 or len(y) == 0:
            log_msg("[ERROR] Training set is empty!", level="ERROR")
            return None, None, None

        log_msg("[INFO] Scaling features with RobustScaler (V26)...", level="INFO")
        scaler = RobustScaler(quantile_range=(10, 90))
        X_scaled = scaler.fit_transform(X)

        split = int(len(X) * 0.85)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        log_msg(f"[INFO] Dataset split: {len(X_train)} training, {len(X_test)} test")

        log_msg("[INFO] Feature selection with SelectKBest (V26)...", level="INFO")
        k_features = min(20, X_train.shape[1])
        selector = SelectKBest(f_classif, k=k_features)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        rf = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=3, max_features='sqrt', random_state=SEED, class_weight='balanced', n_jobs=-1)
        ada = AdaBoostClassifier(n_estimators=80, learning_rate=0.05, random_state=SEED)
        gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=SEED)

        estimators = [('Random Forest', rf), ('AdaBoost', ada), ('Grad. Boosting', gb)]
        final_layer = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000, C=1.0)
        
        clf = StackingClassifier(estimators=estimators, final_estimator=final_layer, cv=5, passthrough=False, n_jobs=-1)
        log_msg("[INFO] Wrapping model with CalibratedClassifierCV (V26)...", level="INFO")
        calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)

        log_msg(f"[TRAIN] Training V26 Optimized Stacking Ensemble...")
        calibrated_clf.fit(X_train_selected, y_train)

        log_msg("-" * 60)
        log_msg(f"{'MODEL':<20} | {'ACCURACY':<10} | {'STATUS'}")
        log_msg("-" * 60)

        try:
            named_ests = calibrated_clf.estimator.named_estimators_
            for name in ['Random Forest', 'AdaBoost', 'Grad. Boosting']:
                est_clf = named_ests[name.replace(' ', '_').lower()]
                pred_single = est_clf.predict(X_test_selected)
                acc_single = accuracy_score(y_test, pred_single)
                log_msg(f"{name:<20} | {acc_single:.3f}      | [OK]")
        except Exception as e:
            pass

        preds = calibrated_clf.predict(X_test_selected)
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='weighted', zero_division=0)
        recall = recall_score(y_test, preds, average='weighted', zero_division=0)
        
        log_msg("-" * 60)
        log_msg(f"{'STACKING V26':<20} | {acc:.3f}      | [FINAL]")
        log_msg("-" * 60)

        return calibrated_clf, scaler, selector

    except Exception as e:
        log_msg(f"[ERROR] Training failed: {e}", level="ERROR")
        return None, None, None

def is_top(t): return t in TOP_TEAMS
def is_weak(t): return t in WEAK_ATTACKS

def calc_poisson_v23(h_name, a_name, h_s, h_c, a_s, a_c):
    try:
        LEAGUE_FACTOR = 1.05
        lh = ((h_s * 0.6 + a_c * 0.4) + 0.1) * LEAGUE_FACTOR
        la = ((a_s * 0.6 + h_c * 0.4) - 0.1) * LEAGUE_FACTOR
        if is_top(h_name) and is_weak(a_name): la *= 0.50
        if is_top(a_name) and is_weak(h_name): lh *= 0.65
        lh = max(0.3, lh)
        la = max(0.2, la)
        ph = 1 - poisson.pmf(0, lh)
        pa = 1 - poisson.pmf(0, la)
        p_gg = ph * pa
        p_ng = 1 - p_gg
        return p_gg, p_ng, lh, la
    except:
        return 0.5, 0.5, h_s, a_s

def predict_next_games(leagues, df_hist, model, scaler, selector=None):
    log_msg("\n[4] ANALISI PARTITE FUTURE (TUTTE LE LEGHE) - V26...")
    future_rows = []
    try:
        if model is None or scaler is None:
            log_msg("[ERROR] Modello non disponibile", level="ERROR")
            return pd.DataFrame()
        for league in leagues:
            l_code = league['code']
            l_id = league['id']
            next_md = DEBUG_MATCHDAYS.get(l_code, 10) + 1
            matches = fetch_matches(l_id, PREDICT_SEASON, league['name'])
            targets = [m for m in matches if m.get('matchday') == next_md]
            if not targets:
                targets = [m for m in matches if m['status'] == 'SCHEDULED'][:5]
            for m in targets:
                parsed = parse_match(m, PREDICT_SEASON, l_code)
                if parsed: future_rows.append(parsed)
        df_next = pd.DataFrame(future_rows)
        if df_next.empty: return df_next
        
        X_next = []
        for i, row in df_next.iterrows():
            try:
                h_stats = compute_advanced_stats(df_hist, row['home_team'], len(df_hist)+1)
                a_stats = compute_advanced_stats(df_hist, row['away_team'], len(df_hist)+1)
                last_h = df_hist[df_hist['home_team']==row['home_team']].tail(1)
                elo_h = last_h['elo_home'].values[0] if not last_h.empty else 1500
                last_a = df_hist[df_hist['away_team']==row['away_team']].tail(1)
                elo_a = last_a['elo_away'].values[0] if not last_a.empty else 1500
                
                h_xg = calculate_xg(row["home_team"], df_hist, len(df_hist)+1, is_home=True)
                a_xg = calculate_xg(row["away_team"], df_hist, len(df_hist)+1, is_home=False)
                h_rest = calculate_rest_days(row["home_team"], df_hist, len(df_hist)+1)
                a_rest = calculate_rest_days(row["away_team"], df_hist, len(df_hist)+1)
                h2h_adv, h2h_gf, h2h_ga = calculate_h2h(row["home_team"], row["away_team"], df_hist)
                h_momentum = calculate_momentum_decay(row["home_team"], df_hist, len(df_hist)+1, is_home=True)
                a_momentum = calculate_momentum_decay(row["away_team"], df_hist, len(df_hist)+1, is_home=False)
                
                feat = [
                    elo_h, elo_a, h_stats['scored_overall'], h_stats['conceded_overall'],
                    a_stats['scored_overall'], a_stats['conceded_overall'], h_stats['form_overall'], a_stats['form_overall'],
                    elo_h - elo_a, h_stats['scored_overall'] * 0.6 + a_stats['conceded_overall'] * 0.4,
                    a_stats['scored_overall'] * 0.6 + h_stats['conceded_overall'] * 0.4,
                    h_stats['home_advantage'], a_stats['home_advantage'], h_stats['trend_recent'], a_stats['trend_recent'],
                    h_stats['efficiency'], a_stats['efficiency'], h_stats['defense_rating'], a_stats['defense_rating'],
                    h_xg, a_xg, h_rest, a_rest, h2h_adv, h2h_gf, h2h_ga, h_momentum, a_momentum,
                ]
                X_next.append(feat)
            except:
                continue
                
        if not X_next: return df_next
        X_sc = scaler.transform(np.array(X_next))
        if selector is not None: X_sc = selector.transform(X_sc)
        
        probs = model.predict_proba(X_sc)
        df_next['probs'] = list(probs)
        return df_next
    except Exception as e:
        log_msg(f"[ERROR] Errore predict_next_games: {e}", level="ERROR")
        return pd.DataFrame()

def calculate_kelly_stake_advanced(prob, quota, bankroll, tier='SAFE'):
    if quota <= 1.0: return 0.0
    prob_conservative = prob * 0.80
    b = quota - 1
    p = prob_conservative
    q = 1 - p
    f_star = (b * p - q) / b
    if f_star <= 0: return 0.0
    
    tier_fractions = {'ULTRA_SAFE': 0.05, 'SAFE': 0.07, 'BALANCED': 0.10, 'VALUE': 0.12, 'AGGRESSIVE': 0.15}
    tier_caps = {'ULTRA_SAFE': 0.02, 'SAFE': 0.03, 'BALANCED': 0.04, 'VALUE': 0.05, 'AGGRESSIVE': 0.06}
    
    stake_pct = f_star * tier_fractions.get(tier, 0.08)
    cap = tier_caps.get(tier, 0.03)
    return min(stake_pct * bankroll, bankroll * cap)

def score_slip_quality(slip, accuracy=0.495):
    ev_score = min(slip['ev'] / 2.5, 1.0) * 40
    n_legs = len(slip['matches'])
    legs_score = (5 - n_legs) * 10 if n_legs <= 5 else 0
    prob_score = min(slip['prob'] * 100, 100) / 100 * 20
    return ev_score + legs_score + prob_score

def validate_bet_combination(bets):
    match_to_bets = {}
    for bet in bets:
        match_key = bet['match']
        if match_key not in match_to_bets: match_to_bets[match_key] = []
        match_to_bets[match_key].append(bet['type'])
    conflicts = [{'1', '2'}, {'1', 'X'}, {'2', 'X'}, {'1', '2X'}, {'2', '1X'}, {'GG', 'NG'}]
    for bet_types in match_to_bets.values():
        bet_set = set(bet_types)
        for conflict in conflicts:
            if conflict.issubset(bet_set): return False
    return True

def calculate_combined_probability(bets):
    if not bets: return 0.0
    adjusted_probs = [bet['prob'] * 0.90 for bet in bets]
    return np.prod(adjusted_probs)

def calculate_combined_ev(bets):
    if not bets: return 0.0
    combined_prob = calculate_combined_probability(bets)
    combined_quota = np.prod([bet['quota'] for bet in bets])
    return (combined_prob * combined_quota) - 1.0

def generate_tiered_portfolio(options, model_accuracy, budget):
    if not options: return []
    portfolio = []
    global_used_matches = set()

    def has_duplicate_match(slip, used_in_tier):
        for match in slip['matches']:
            if match in used_in_tier or match in global_used_matches: return True
        return False
        
    def add_slip_with_dedup(slip, portfolio, tier_slips):
        used_in_tier = set()
        for existing_slip in tier_slips:
            for match in existing_slip['matches']: used_in_tier.add(match)
        if not has_duplicate_match(slip, used_in_tier):
            slip['quality_score'] = score_slip_quality(slip, model_accuracy)
            tier_slips.append(slip)
            for m in slip['matches']: global_used_matches.add(m)
            return True
        return False

    t1_ops = [o for o in options if o['prob'] >= 0.75 and 1.15 <= o['quota'] <= 1.40 and o['ev'] > 1.03]
    t1_ops.sort(key=lambda x: x['ev'], reverse=True)
    best_t1 = t1_ops[:15]; random.shuffle(best_t1)
    tier1_slips = []
    for combo in itertools.combinations(best_t1, 2):
        matches = [c['match'] for c in combo]
        if any(m in global_used_matches for m in matches): continue
        if len(matches) != len(set(matches)): continue
        if not validate_bet_combination(list(combo)): continue
        slip = {'tier': 'ULTRA_SAFE', 'strategy': 'IL BUNKER 🛡️', 'matches': matches, 'types': [c['type'] for c in combo],
                'prob': calculate_combined_probability(list(combo)), 'quota': np.prod([c['quota'] for c in combo]),
                'quotas': [c['quota'] for c in combo], 'ev': calculate_combined_ev(list(combo))}
        if len(tier1_slips) < 2: add_slip_with_dedup(slip, portfolio, tier1_slips)
    portfolio.extend(tier1_slips)

    t2_ops = [o for o in options if o['prob'] >= 0.70 and 1.40 <= o['quota'] <= 1.80 and o['ev'] > 1.05 and o['match'] not in global_used_matches]
    t2_ops.sort(key=lambda x: x['ev'], reverse=True)
    best_t2 = t2_ops[:20]; random.shuffle(best_t2)
    tier2_slips = []
    for combo in itertools.combinations(best_t2, 3):
        matches = [c['match'] for c in combo]
        if not validate_bet_combination(list(combo)): continue
        c_prob = calculate_combined_probability(list(combo))
        slip = {'tier': 'SAFE', 'strategy': 'IL TRINCERONE 🏰', 'matches': matches, 'types': [c['type'] for c in combo],
                'prob': c_prob, 'quota': np.prod([c['quota'] for c in combo]), 'quotas': [c['quota'] for c in combo], 'ev': calculate_combined_ev(list(combo))}
        if c_prob >= 0.20 and len(tier2_slips) < 2: add_slip_with_dedup(slip, portfolio, tier2_slips)
    portfolio.extend(tier2_slips)

    t3_ops = [o for o in options if o['prob'] >= 0.60 and 1.70 <= o['quota'] <= 2.50 and o['ev'] > 1.07 and o['match'] not in global_used_matches]
    t3_ops.sort(key=lambda x: x['ev'], reverse=True)
    best_t3 = t3_ops[:20]; random.shuffle(best_t3)
    tier3_slips = []
    for combo in itertools.combinations(best_t3, 3):
        matches = [c['match'] for c in combo]
        if not validate_bet_combination(list(combo)): continue
        c_prob = calculate_combined_probability(list(combo))
        slip = {'tier': 'BALANCED', 'strategy': 'LA BILANCIA ⚖️', 'matches': matches, 'types': [c['type'] for c in combo],
                'prob': c_prob, 'quota': np.prod([c['quota'] for c in combo]), 'quotas': [c['quota'] for c in combo], 'ev': calculate_combined_ev(list(combo))}
        if c_prob >= 0.12 and len(tier3_slips) < 2: add_slip_with_dedup(slip, portfolio, tier3_slips)
    portfolio.extend(tier3_slips)

    t4_ops = [o for o in options if o['quota'] >= 2.20 and o['ev'] > 1.10 and o['match'] not in global_used_matches]
    t4_ops.sort(key=lambda x: x['ev'], reverse=True)
    best_t4 = t4_ops[:15]; random.shuffle(best_t4)
    tier6_slips = []
    for combo in itertools.combinations(best_t4, 2):
        matches = [c['match'] for c in combo]
        if not validate_bet_combination(list(combo)): continue
        c_prob = calculate_combined_probability(list(combo))
        slip = {'tier': 'AGGRESSIVE', 'strategy': 'IL PIRATA 🚀', 'matches': matches, 'types': [c['type'] for c in combo],
                'prob': c_prob, 'quota': np.prod([c['quota'] for c in combo]), 'quotas': [c['quota'] for c in combo], 'ev': calculate_combined_ev(list(combo))}
        if c_prob >= 0.08 and len(tier6_slips) < 1: add_slip_with_dedup(slip, portfolio, tier6_slips)
    portfolio.extend(tier6_slips)

    t5_ops = [o for o in options if o['quota'] >= 1.80 and o['ev'] > 1.10 and o['match'] not in global_used_matches]
    t5_ops.sort(key=lambda x: x['ev'], reverse=True)
    best_t5 = t5_ops[:15]; random.shuffle(best_t5)
    tier5_slips = []
    for combo in itertools.combinations(best_t5, 2):
        matches = [c['match'] for c in combo]
        if not validate_bet_combination(list(combo)): continue
        c_prob = calculate_combined_probability(list(combo))
        slip = {'tier': 'VALUE', 'strategy': 'IL CACCIATORE 💎', 'matches': matches, 'types': [c['type'] for c in combo],
                'prob': c_prob, 'quota': np.prod([c['quota'] for c in combo]), 'quotas': [c['quota'] for c in combo], 'ev': calculate_combined_ev(list(combo))}
        if c_prob >= 0.15 and len(tier5_slips) < 2: add_slip_with_dedup(slip, portfolio, tier5_slips)
    portfolio.extend(tier5_slips)

    portfolio.sort(key=lambda x: x['quality_score'], reverse=True)
    return portfolio

def print_final_strategy_v25(portfolio, budget, model_accuracy):
    log_msg("\n" + "$$" * 50)
    log_msg(f"$$   PORTAFOGLIO INTELLIGENTE V26 (AI-OPTIMIZED)   $$")
    log_msg(f"$$   BUDGET: {budget}€ | ACCURACY: {model_accuracy*100:.1f}%   $$")
    log_msg("$$" * 50)
    
    if not portfolio:
        log_msg("[!] Nessuna combinazione valida trovata.")
        return
        
    tier_allocations = {'ULTRA_SAFE': 0.70, 'SAFE': 0.15, 'BALANCED': 0.05, 'VALUE': 0.05, 'AGGRESSIVE': 0.05}
    tier_budgets = {tier: budget * alloc for tier, alloc in tier_allocations.items()}
    tier_used = {tier: 0.0 for tier in tier_allocations.keys()}
    top_picks = {}
    
    for slip in portfolio:
        tier = slip['tier']
        if tier not in top_picks: top_picks[tier] = []
        if len(top_picks[tier]) >= 2: continue
        stake = calculate_kelly_stake_advanced(slip['prob'] * 0.95, slip['quota'], tier_budgets[tier], tier=tier)
        if stake < 1.5: stake = 1.5
        if tier_used[tier] + stake > tier_budgets[tier]: continue
        potential_win = stake * slip['quota']
        top_picks[tier].append({'slip': slip, 'stake': stake, 'potential_win': potential_win, 'roi': ((potential_win - stake) / stake * 100)})
        tier_used[tier] += stake
        
    global_idx = 1
    total_used = 0.0
    tier_name_icon = {'ULTRA_SAFE': '🛡️ TIER 1: IL BUNKER', 'SAFE': '🏰 TIER 2: IL TRINCERONE', 'BALANCED': '⚖️ TIER 3: LA BILANCIA', 'VALUE': '💎 TIER 4: IL CACCIATORE', 'AGGRESSIVE': '🚀 TIER 5: IL PIRATA'}
    
    for tier in ['ULTRA_SAFE', 'SAFE', 'BALANCED', 'VALUE', 'AGGRESSIVE']:
        if tier not in top_picks or not top_picks[tier]: continue
        log_msg(f"\n{'='*100}\n{tier_name_icon[tier]}\n{'='*100}")
        log_msg(f"Budget Tier: {tier_budgets[tier]:.2f}€ | Usato: {tier_used[tier]:.2f}€ | Rimanente: {tier_budgets[tier] - tier_used[tier]:.2f}€\n")
        for pick in top_picks[tier]:
            slip, stake, potential_win, roi = pick['slip'], pick['stake'], pick['potential_win'], pick['roi']
            log_msg(f"  {global_idx}. {slip['strategy']}")
            log_msg(f"     Qualità: {slip['quality_score']:.1f}/100 | EV: {slip['ev']:.3f} | ROI Teorico: {roi:.1f}%")
            log_msg(f"     Quota: {slip['quota']:.2f} | Prob.: {slip['prob']*100:.1f}% | N° Gambe: {len(slip['matches'])}")
            log_msg(f"     💰 PUNTATA: {stake:.2f}€ → VINCITA POTENZIALE: {potential_win:.2f}€")
            log_msg(f"     {'-'*80}")
            for j, m in enumerate(slip['matches']):
                log_msg(f"     {j+1}. {m:<50} [{slip['types'][j]:<4}] @{slip['quotas'][j]:.2f}")
            log_msg("")
            global_idx += 1
            total_used += stake
            
    log_msg("="*100 + "\nRIEPILOGO PORTAFOGLIO\n" + "="*100)
    log_msg(f"Schedine Proposte: {global_idx - 1}")
    log_msg(f"Budget Totale Impiegato: {total_used:.2f}€ ({total_used/budget*100:.1f}% del budget)")
    log_msg(f"Budget Rimanente: {budget - total_used:.2f}€")
    log_msg("="*100 + "\n")

def calculate_best_bets_v25(df_next, odds_list, model_accuracy, df_hist):
    log_msg("\n[SCHEDINE V26] Generazione Portfolio Tiered Avanzato...")
    all_options = []
    for i, row in df_next.iterrows():
        try:
            q = odds_list[i] if i < len(odds_list) else {}
            pr = row['probs']
            pa, px, ph = pr[0], pr[1], pr[2]
            h_stats = compute_advanced_stats(df_hist, row['home_team'], len(df_hist)+1)
            a_stats = compute_advanced_stats(df_hist, row['away_team'], len(df_hist)+1)
            _, _, lh, la = calc_poisson_v23(row['home_team'], row['away_team'], h_stats['scored_overall'], h_stats['conceded_overall'], a_stats['scored_overall'], a_stats['conceded_overall'])
            p_0_0 = poisson.pmf(0, lh) * poisson.pmf(0, la)
            prob_ng = (poisson.pmf(0, lh) + poisson.pmf(0, la) - p_0_0)
            prob_gg = 1 - prob_ng
            match_lbl = f"[{row['league']}] {row['home_team']} vs {row['away_team']}"
            raw_bets = [
                {'type': '1', 'prob': ph, 'quota': q.get('1', 1.0)},
                {'type': 'X', 'prob': px, 'quota': q.get('X', 1.0)},
                {'type': '2', 'prob': pa, 'quota': q.get('2', 1.0)},
                {'type': '1X', 'prob': ph+px, 'quota': q.get('1X', 1.0)},
                {'type': '2X', 'prob': pa+px, 'quota': q.get('2X', 1.0)},
                {'type': 'GG', 'prob': prob_gg, 'quota': q.get('GG', 1.0)},
                {'type': 'NG', 'prob': prob_ng, 'quota': q.get('NG', 1.0)}
            ]
            for bet in raw_bets:
                if bet['quota'] <= 1.05: continue
                ev = bet['prob'] * bet['quota']
                if ev > 1.05:
                    all_options.append({'match': match_lbl, 'type': bet['type'], 'prob': bet['prob'], 'quota': bet['quota'], 'ev': ev})
        except: continue
    if not all_options:
        log_msg("[WARN] Nessuna opzione di qualità disponibile", level="WARNING")
        return
    portfolio = generate_tiered_portfolio(all_options, model_accuracy, BUDGET_TOTALE)
    print_final_strategy_v25(portfolio, BUDGET_TOTALE, model_accuracy)


# ==============================================================================
# BLOCCO ESECUZIONE AI (Richiamato dalla GUI Flet)
# ==============================================================================
def execute_ml_pipeline():
    try:
        log_msg("\n[0] INIZIO SCANSIONE EUROPA (V26 OPTIMIZED)...")
        log_msg("="*100)
        df_hist = build_global_dataset(LEAGUES_CONFIG, SEASONS_TRAIN, SEASONS_CURRENT, DEBUG_MATCHDAYS)
        
        if df_hist.empty:
            log_msg("[ERROR] Dataset vuoto, impossibile continuare", level="ERROR")
        else:
            log_msg("[1] USING V26 ENHANCED FEATURES (27 FEATURES WITH ADVANCED METRICS)...")
            X, y, df_hist = build_features_v26_enhanced(df_hist)
            
            log_msg("[2] USING V26 OPTIMIZED TRAINING (ROBUST SCALING + CALIBRATION)...")
            model_result = train_model_v26_optimized(X, y)
            
            if len(model_result) == 3:
                model, scaler, selector = model_result
            else:
                model, scaler = model_result
                selector = None
            
            if model is not None and scaler is not None:
                split = int(len(X) * 0.85)
                X_scaled = scaler.transform(X)
                
                if selector:
                    X_test_selected = selector.transform(X_scaled[split:])
                else:
                    X_test_selected = X_scaled[split:]
                
                acc = accuracy_score(y[split:], model.predict(X_test_selected))
                log_msg(f"[V26] Final Model Accuracy on Test Set: {acc:.3f} ({acc*100:.1f}%)", level="INFO")
                
                df_next = predict_next_games(LEAGUES_CONFIG, df_hist, model, scaler, selector)
                
                if not df_next.empty:
                    odds = fetch_odds_global(df_next)
                    calculate_best_bets_v25(df_next, odds, acc, df_hist)
                else:
                    log_msg("[WARN] Nessuna partita futura per l'analisi", level="WARNING")
            else:
                log_msg("[ERROR] V26 Training fallito", level="ERROR")
        
        log_msg("\n[DONE] Analisi Completata (V26).")

    except Exception as e:
        log_msg(f"\n[CRITICAL ERROR] {e}", level="ERROR")
        traceback.print_exc()
        log_msg(traceback.format_exc(), level="ERROR")


# ==============================================================================
# FLET GUI - INTERFACCIA GRAFICA (AGGIORNATO FLET V1)
# ==============================================================================
class FletTerminalOutput(io.StringIO):
    def __init__(self, page: ft.Page, log_view: ft.ListView):
        super().__init__()
        self.page = page
        self.log_view = log_view

    def write(self, string):
        if string.strip():
            self.log_view.controls.append(
                ft.Text(string.strip(), color="green400", font_family="Consolas", size=13)
            )
            self.page.update()
        return super().write(string)

def main(page: ft.Page):
    page.title = "European Football Predictor AI (V26)"
    page.theme_mode = "dark"
    page.window_width = 1100
    page.window_height = 800
    page.padding = 20

    # UI Inputs
    matchday_inputs = {
        'SA': ft.TextField(label="Serie A", value=str(DEBUG_MATCHDAYS['SA']), width=120, text_align="center"),
        'PL': ft.TextField(label="Premier League", value=str(DEBUG_MATCHDAYS['PL']), width=120, text_align="center"),
        'PD': ft.TextField(label="La Liga", value=str(DEBUG_MATCHDAYS['PD']), width=120, text_align="center"),
        'BL1': ft.TextField(label="Bundesliga", value=str(DEBUG_MATCHDAYS['BL1']), width=120, text_align="center"),
        'FL1': ft.TextField(label="Ligue 1", value=str(DEBUG_MATCHDAYS['FL1']), width=120, text_align="center"),
    }

    matchdays_row = ft.Row(controls=list(matchday_inputs.values()), alignment="spaceBetween")

    default_odds_json = json.dumps({"SA": [], "PL": [], "PD": [], "BL1": [], "FL1": []}, indent=4)
    odds_input = ft.TextField(
        label="Inserisci le Quote Aggiornate (Formato JSON) - Lascia vuoto per usare Default",
        multiline=True, min_lines=15, max_lines=15, value=default_odds_json,
        text_style=ft.TextStyle(font_family="Consolas", size=13)
    )

    terminal_view = ft.ListView(expand=True, spacing=2, auto_scroll=True)
    terminal_container = ft.Container(
        content=terminal_view, bgcolor="black", border_radius=10,
        padding=15, expand=True, border=ft.border.all(1, "blue700")
    )

    progress_ring = ft.ProgressRing(visible=False, width=24, height=24)

    def run_script_thread(e):
        btn_run.disabled = True
        progress_ring.visible = True
        page.update()

        try:
            # Aggiorniamo le variabili globali dallo stato della UI
            global DEBUG_MATCHDAYS, DYNAMIC_ODDS
            DEBUG_MATCHDAYS = {league: int(field.value) for league, field in matchday_inputs.items()}
            
            try:
                parsed_odds = json.loads(odds_input.value)
                # Verifica che non sia solo il JSON vuoto standard
                if any(len(parsed_odds.get(k, [])) > 0 for k in parsed_odds):
                    DYNAMIC_ODDS = parsed_odds
            except json.JSONDecodeError:
                terminal_view.controls.append(ft.Text("[ERRORE] JSON non valido. Uso le quote interne.", color="red400"))

            # Redirigi log su Flet
            original_stdout = sys.stdout
            sys.stdout = FletTerminalOutput(page, terminal_view)

            # Avvia la pipeline Machine Learning
            execute_ml_pipeline()

            # Ripristina stdout
            sys.stdout = original_stdout

        except Exception as ex:
            terminal_view.controls.append(ft.Text(f"[FATAL UI ERROR] {ex}", color="red500"))
        finally:
            btn_run.disabled = False
            progress_ring.visible = False
            page.update()

    def start_analysis(e):
        # Sposta la vista alla scheda del terminale automaticamente
        page.update()
        threading.Thread(target=run_script_thread, args=(e,), daemon=True).start()

    # BOTTONE FIXATO (senza "text" o "icon")
    btn_run = ft.ElevatedButton(
        content=ft.Row(
            [
                ft.Icon("play_arrow", color="white"),
                ft.Text("Avvia Analisi AI", color="white", weight="bold")
            ],
            alignment="center",
            tight=True
        ),
        on_click=start_analysis,
        style=ft.ButtonStyle(bgcolor="blue700", padding=20)
    )

    # ==============================================================================
    # STRUTTURA TABS FIXATA (Per aggirare l'errore TypeError: Tab.__init__)
    # ==============================================================================
    
    # Contenuto della prima scheda
    tab_settings_content = ft.Container(
        padding=20,
        content=ft.Column(
            controls=[
                ft.Text("Ultima Giornata Giocata (Modificabile)", size=20, weight="bold"),
                ft.Divider(), matchdays_row, ft.Container(height=20),
                ft.Text("Mappatura Quote Dinamiche", size=20, weight="bold"),
                odds_input, ft.Container(height=20),
                ft.Row([btn_run, progress_ring], alignment="center")
            ], scroll="auto"
        )
    )

    # Contenuto della seconda scheda
    tab_terminal_content = ft.Container(
        padding=20,
        content=ft.Column(expand=True, controls=[
            ft.Text("Console Esecuzione Predittiva", size=20, weight="bold"),
            terminal_container
        ])
    )

    # Nuovo sistema di schede compatibile con la V1
    tabs = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        expand=True,
        tabs=[
            ft.Tab(
                tab_content=ft.Text("⚙️ Impostazioni"),
                content=tab_settings_content
            ),
            ft.Tab(
                tab_content=ft.Text("💻 Terminale Output"),
                content=tab_terminal_content
            )
        ]
    )

    header = ft.Row([
        ft.Icon("sports_soccer", size=40, color="blue400"),
        ft.Text("AI PREDICTOR V26 - CONTROL PANEL", size=28, weight="w900", color="blue100")
    ], alignment="center")

    page.add(header, ft.Divider(height=30), tabs)

if __name__ == "__main__":
    ft.app(target=main)
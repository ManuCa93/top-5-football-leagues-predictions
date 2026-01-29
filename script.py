"""
EUROPEAN PREDICTOR 2025-26 - VERSIONE V25 (FULL INTEGRATION)
Advanced AI-Powered Sports Betting Portfolio Generator

FEATURES:
✅ 5-Tier Portfolio System (Bunker → Cacciatore → Pirata)
✅ Dynamic Budget Allocation (accuracy-adaptive)
✅ Quality Scoring System (100-point scale)
✅ Kelly Criterion Optimized Stake Sizing
✅ Correlation Conflict Detection
✅ Real-time Poisson GG/NG Calculations
✅ EV-based Bet Filtering
✅ Intelligent Portfolio Diversification
"""

import sys
import io
import os
import time
import traceback
import warnings
import random
import itertools
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
warnings.filterwarnings("ignore")

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except:
    pass

DEBUG_MODE = True

# Last matchday played
DEBUG_MATCHDAYS = {'SA': 15, 'PL': 16, 'PD': 16, 'BL1': 14, 'FL1': 16}

def get_odds_mapping():
    return {
        'SA': [
            {'home': 'Lazio', 'away': 'Cremonese', '1': 1.60, 'X': 3.80, '2': 5.75, '1X': 1.12, '2X': 2.25, 'GG': 1.95, 'NG': 1.77},
            {'home': 'Juventus', 'away': 'Roma', '1': 2.05, 'X': 3.20, '2': 3.90, '1X': 1.24, '2X': 1.73, 'GG': 1.95, 'NG': 1.77},
            {'home': 'Cagliari', 'away': 'Pisa', '1': 2.10, 'X': 3.15, '2': 3.55, '1X': 1.27, '2X': 1.67, 'GG': 1.97, 'NG': 1.77},
            {'home': 'Sassuolo', 'away': 'Torino', '1': 2.30, 'X': 3.05, '2': 3.25, '1X': 1.30, '2X': 1.57, 'GG': 1.80, 'NG': 1.90},
            {'home': 'Fiorentina', 'away': 'Udinese', '1': 2.15, 'X': 3.20, '2': 3.40, '1X': 1.30, '2X': 1.65, 'GG': 1.80, 'NG': 1.90},
            {'home': 'Genoa', 'away': 'Atalanta', '1': 3.90, 'X': 3.35, '2': 1.92, '1X': 1.80, '2X': 1.22, 'GG': 1.85, 'NG': 1.87},
            {'home': 'Napoli', 'away': 'Parma', '1': 0, 'X': 0, '2': 0, '1X': 0, '2X': 0, 'GG': 0, 'NG': 0},
            {'home': 'Inter', 'away': 'Lecce', '1': 0, 'X': 0, '2': 0, '1X': 0, '2X': 0, 'GG': 0, 'NG': 0},
            {'home': 'Verona', 'away': 'Bologna', '1': 0, 'X': 0, '2': 0, '1X': 0, '2X': 0, 'GG': 0, 'NG': 0},
            {'home': 'Como', 'away': 'Milan', '1': 0, 'X': 0, '2': 0, '1X': 0, '2X': 0, 'GG': 0, 'NG': 0},
        ],
        'PL': [
            {'home': 'Newcastle', 'away': 'Chelsea', '1': 2.65, 'X': 3.50, '2': 2.50, '1X': 1.50, '2X': 1.43, 'GG': 1.53, 'NG': 2.35},
            {'home': 'Bournemouth', 'away': 'Burnley', '1': 1.45, 'X': 4.50, '2': 6.25, '1X': 1.09, '2X': 2.60, 'GG': 1.77, 'NG': 1.92},
            {'home': 'Brighton', 'away': 'Sunderland', '1': 1.60, 'X': 3.90, '2': 5.00, '1X': 1.14, '2X': 2.20, 'GG': 1.75, 'NG': 1.97},
            {'home': 'Wolves', 'away': 'Brentford', '1': 3.60, 'X': 3.45, '2': 2.00, '1X': 1.75, '2X': 1.25, 'GG': 1.73, 'NG': 2.00},
            {'home': 'Man City', 'away': 'West Ham', '1': 1.19, 'X': 7.25, '2': 11.00, '1X': 1.02, '2X': 4.25, 'GG': 1.82, 'NG': 1.90},
            {'home': 'Tottenham', 'away': 'Liverpool', '1': 3.35, 'X': 3.70, '2': 2.00, '1X': 1.75, '2X': 1.30, 'GG': 1.53, 'NG': 2.35},
            {'home': 'Leeds', 'away': 'Crystal Palace', '1': 2.55, 'X': 3.25, '2': 2.70, '1X': 1.40, '2X': 1.47, 'GG': 1.77, 'NG': 1.95},
            {'home': 'Everton', 'away': 'Arsenal', '1': 5.50, 'X': 3.85, '2': 1.60, '1X': 2.20, '2X': 1.12, 'GG': 2.00, 'NG': 1.73},
            {'home': 'Aston Villa', 'away': 'Man United', '1': 2.10, 'X': 3.60, '2': 3.20, '1X': 1.30, '2X': 1.67, 'GG': 1.50, 'NG': 2.40},
            {'home': 'Fulham', 'away': 'Nottingham', '1': 2.25, 'X': 3.30, '2': 3.10, '1X': 1.33, '2X': 1.60, 'GG': 1.73, 'NG': 2.00},
        ],
        'PD': [
            {'home': 'Valencia', 'away': 'Mallorca', '1': 1.97, 'X': 3.25, '2': 4.00, '1X': 1.21, '2X': 1.77, 'GG': 1.95, 'NG': 1.77},
            {'home': 'Oviedo', 'away': 'Celta', '1': 3.65, 'X': 3.40, '2': 2.00, '1X': 1.75, '2X': 1.25, 'GG': 1.90, 'NG': 1.82},
            {'home': 'Levante', 'away': 'Real Sociedad', '1': 3.20, 'X': 3.40, '2': 2.15, '1X': 1.65, '2X': 1.30, 'GG': 1.67, 'NG': 2.05},
            {'home': 'Osasuna', 'away': 'Alaves', '1': 2.25, 'X': 3.00, '2': 3.45, '1X': 1.27, '2X': 1.60, 'GG': 2.05, 'NG': 1.67},
            {'home': 'Real Madrid', 'away': 'Sevilla', '1': 1.19, 'X': 6.75, '2': 12.00, '1X': 1.01, '2X': 4.25, 'GG': 1.87, 'NG': 1.85},
            {'home': 'Girona', 'away': 'Atletico', '1': 4.60, 'X': 4.00, '2': 1.65, '1X': 2.10, '2X': 1.17, 'GG': 1.67, 'NG': 2.05},
            {'home': 'Villarreal', 'away': 'Barcelona', '1': 3.25, 'X': 4.50, '2': 1.87, '1X': 1.85, '2X': 1.30, 'GG': 1.30, 'NG': 3.30},
            {'home': 'Elche', 'away': 'Rayo', '1': 2.40, 'X': 3.10, '2': 3.00, '1X': 1.35, '2X': 1.50, 'GG': 1.93, 'NG': 1.77},
            {'home': 'Real Betis', 'away': 'Getafe', '1': 1.70, 'X': 3.45, '2': 5.25, '1X': 1.13, '2X': 2.05, 'GG': 2.15, 'NG': 1.65},
            {'home': 'Athletic Club', 'away': 'Espanyol', '1': 1.70, 'X': 3.50, '2': 5.00, '1X': 1.14, '2X': 2.05, 'GG': 2.05, 'NG': 1.70},
        ],
        'BL1': [
            {'home': 'Dortmund', 'away': 'Gladbach', '1': 1.47, 'X': 4.60, '2': 5.75, '1X': 1.11, '2X': 2.55, 'GG': 1.60, 'NG': 2.20},
            {'home': 'Augsburg', 'away': 'Werder Bremen', '1': 2.25, 'X': 3.45, '2': 3.00, '1X': 1.35, '2X': 1.60, 'GG': 1.57, 'NG': 2.25},
            {'home': 'Wolfsburg', 'away': 'Freiburg', '1': 2.40, 'X': 3.45, '2': 2.75, '1X': 1.40, '2X': 1.50, 'GG': 1.60, 'NG': 2.20},
            {'home': 'Hamburger SV', 'away': 'Frankfurt', '1': 2.60, 'X': 3.50, '2': 2.50, '1X': 1.50, '2X': 1.45, 'GG': 1.50, 'NG': 2.40},
            {'home': 'Stuttgart', 'away': 'Hoffenheim', '1': 1.95, 'X': 3.85, '2': 3.35, '1X': 1.30, '2X': 1.77, 'GG': 1.40, 'NG': 2.65},
            {'home': 'Koln', 'away': 'Union Berlin', '1': 2.35, 'X': 3.25, '2': 3.00, '1X': 1.35, '2X': 1.55, 'GG': 1.70, 'NG': 2.05},
            {'home': 'Leipzig', 'away': 'Leverkusen', '1': 2.05, 'X': 3.75, '2': 3.20, '1X': 1.30, '2X': 1.70, 'GG': 1.45, 'NG': 2.60},
            {'home': 'Mainz', 'away': 'St. Pauli', '1': 1.90, 'X': 3.35, '2': 4.00, '1X': 1.21, '2X': 1.82, 'GG': 1.88, 'NG': 1.83},
            {'home': 'Heidenheim', 'away': 'Bayern', '1': 14.00, 'X': 8.75, '2': 1.13, '1X': 5.25, '2X': 1.04, 'GG': 1.85, 'NG': 1.87},
        ],
        'FL1': [
            {'home': 'Toulouse', 'away': 'Lens', '1': 2.75, 'X': 3.30, '2': 2.40, '1X': 1.50, '2X': 1.40, 'GG': 1.63, 'NG': 2.10},
            {'home': 'Monaco', 'away': 'Lyon', '1': 1.90, 'X': 3.70, '2': 3.45, '1X': 1.27, '2X': 1.77, 'GG': 1.53, 'NG': 2.35},
            {'home': 'Nice', 'away': 'Strasbourg', '1': 2.70, 'X': 3.40, '2': 2.40, '1X': 1.50, '2X': 1.40, 'GG': 1.55, 'NG': 2.30},
            {'home': 'Lille', 'away': 'Rennes', '1': 1.83, 'X': 3.55, '2': 4.00, '1X': 1.20, '2X': 1.87, 'GG': 1.55, 'NG': 2.30},
            {'home': 'Marseille', 'away': 'Nantes', '1': 1.22, 'X': 6.25, '2': 10.00, '1X': 1.02, '2X': 3.75, 'GG': 1.95, 'NG': 1.75},
            {'home': 'Le Havre', 'away': 'Angers', '1': 2.05, 'X': 3.15, '2': 3.65, '1X': 1.23, '2X': 1.67, 'GG': 1.95, 'NG': 1.75},
            {'home': 'Brest', 'away': 'Auxerre', '1': 1.85, 'X': 3.30, '2': 4.25, '1X': 1.18, '2X': 1.85, 'GG': 1.75, 'NG': 1.95},
            {'home': 'Lorient', 'away': 'Metz', '1': 1.70, 'X': 3.70, '2': 4.40, '1X': 1.17, '2X': 2.00, 'GG': 1.70, 'NG': 2.00},
            {'home': 'PSG', 'away': 'Paris FC', '1': 1.21, 'X': 6.50, '2': 10.00, '1X': 1.02, '2X': 3.85, 'GG': 1.87, 'NG': 1.80},
        ]
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

log_msg("="*100)
log_msg("[START] EUROPEAN PREDICTOR - V25 (ADVANCED PORTFOLIO)")
log_msg("="*100)

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
# =======================
# TEAM NAME NORMALIZATION (AGGIORNATO)
# =======================
TEAM_ALIASES = {
    # SERIE A
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
    
    # PREMIER LEAGUE
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
    
    # LA LIGA (CORREZIONI CRITICHE QUI)
    'Real Madrid CF': 'Real Madrid', 'Real Madrid': 'Real Madrid', 'FC Barcelona': 'Barcelona',
    'Barcelona': 'Barcelona', 'Atlético Madrid': 'Atletico', 'Atletico Madrid': 'Atletico',
    'Atleti': 'Atletico', 'Real Betis Balompié': 'Real Betis', 'Real Betis': 'Real Betis',
    'Betis': 'Real Betis', 'Villarreal CF': 'Villarreal', 'Villarreal': 'Villarreal',
    'Athletic Club': 'Athletic Club', 'Athletic Bilbao': 'Athletic Club', 'Athletic': 'Athletic Club',
    'Real Sociedad': 'Real Sociedad', 'Real Sociedad de Fútbol': 'Real Sociedad',
    'Girona FC': 'Girona', 'Girona': 'Girona', 'Getafe CF': 'Getafe', 'Getafe': 'Getafe',
    'Sevilla FC': 'Sevilla', 'Sevilla': 'Sevilla', 'Valencia CF': 'Valencia', 'Valencia': 'Valencia',
    'Rayo Vallecano': 'Rayo', 'Vallecano': 'Rayo', 
    'RCD Espanyol de Barcelona': 'Espanyol', 'RCD Espanyol': 'Espanyol', 'Espanyol': 'Espanyol', # FIX
    'Real Oviedo': 'Oviedo', 'Oviedo': 'Oviedo', 'RCD Mallorca': 'Mallorca', 'Mallorca': 'Mallorca',
    'Elche CF': 'Elche', 'Elche': 'Elche', 'Celta Vigo': 'Celta', 'Celta de Vigo': 'Celta',
    'Celta': 'Celta', 'Levante UD': 'Levante', 'Levante': 'Levante', 'CA Osasuna': 'Osasuna',
    'Osasuna': 'Osasuna', 'Las Palmas': 'Las Palmas', 'CD Leganés': 'Leganes',
    'Deportivo Alavés': 'Alaves', 'Alavés': 'Alaves', 'Alaves': 'Alaves', # FIX
    'Leganes': 'Leganes', 'Real Valladolid': 'Valladolid', 'Valladolid': 'Valladolid',
    
    # BUNDESLIGA (CORREZIONI CRITICHE QUI)
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
    '1. FC Köln': 'Koln', '1. FC Koeln': 'Koln', 'Köln': 'Koln', # FIX
    'Hamburger SV': 'Hamburger SV',
    'Holstein Kiel': 'Holstein Kiel', 'Kiel': 'Holstein Kiel',
    
    # LIGUE 1 (CORREZIONI CRITICHE QUI)
    # LIGUE 1 (AGGIORNATO CON PARIS FC)
    'Paris Saint-Germain FC': 'PSG', 'Paris Saint-Germain': 'PSG', 'Paris SG': 'PSG',
    'Paris FC': 'Paris FC', 'PFC': 'Paris FC', # <--- QUESTA RIGA È FONDAMENTALE
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
    
    # Quote di default (neutre)
    default_odds = {'1': 1.01, 'X': 1.01, '2': 1.01, '1X': 1.01, '2X': 1.01, 'GG': 1.01, 'NG': 1.01}

    try:
        matches_matched = 0
        for idx, row in df_matches.iterrows():
            league = row['league']
            home_api = row['home_team']
            away_api = row['away_team']
            
            league_odds_data = odds_mapping.get(league, [])
            
            found_odds = None
            
            # Normalizzazione stringhe per il confronto (rimuove FC, 1., spazi)
            h_api_clean = home_api.lower().replace("fc", "").replace("1.", "").strip()
            a_api_clean = away_api.lower().replace("fc", "").replace("1.", "").strip()

            for stored_match in league_odds_data:
                if 'home' not in stored_match or 'away' not in stored_match:
                    continue
                
                h_store = stored_match['home'].lower().replace("fc", "").strip()
                a_store = stored_match['away'].lower().replace("fc", "").strip()
                
                # Check 1: Similarità (Fuzzy)
                h_ratio = SequenceMatcher(None, h_api_clean, h_store).ratio()
                a_ratio = SequenceMatcher(None, a_api_clean, a_store).ratio()
                
                # Check 2: Contenimento (es. "Lens" in "Racing Club de Lens")
                h_sub = h_store in h_api_clean or h_api_clean in h_store
                a_sub = a_store in a_api_clean or a_api_clean in a_store

                # Logica rilassata: Se uno dei due metodi funziona per entrambe le squadre
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
        import traceback
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
    
    # --- PARTE 1: GESTIONE STORICO (CACHE) ---
    if os.path.exists(CACHE_FILE):
        log_msg(f"[CACHE] Trovato file '{CACHE_FILE}'. Caricamento dati storici da locale...")
        try:
            df_train = pd.read_csv(CACHE_FILE)
            # Riconvertiamo la data in datetime perché il CSV la salva come stringa
            df_train["date"] = pd.to_datetime(df_train["date"])
            log_msg(f"[CACHE] Caricati {len(df_train)} match storici dal file.")
        except Exception as e:
            log_msg(f"[ERROR] Errore lettura cache ({e}). Riscarico tutto.", level="ERROR")
            df_train = pd.DataFrame()
    else:
        df_train = pd.DataFrame()

    # Se il dataframe storic è vuoto (o non esisteva il file), scarichiamo dall'API
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
            # Salviamo su file per la prossima volta
            try:
                df_train.to_csv(CACHE_FILE, index=False)
                log_msg(f"[CACHE] Salvato file '{CACHE_FILE}' con {len(df_train)} match.")
            except Exception as e:
                log_msg(f"[WARN] Impossibile salvare cache: {e}", level="WARNING")

    # --- PARTE 2: GESTIONE STAGIONE CORRENTE (SEMPRE FRESH) ---
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
                        # Carichiamo solo fino alla giornata indicata o tutto se finito
                        if m.get("matchday", 0) <= current_md:
                            parsed = parse_match(m, s, l_code)
                            if parsed:
                                all_curr_rows.append(parsed)
                                count_curr += 1
                log_msg(f" -> {league['name']}: {count_curr} partite correnti scaricate.")
    except Exception as e:
        log_msg(f"[ERROR] Errore scaricamento current: {e}", level="ERROR")

    df_curr = pd.DataFrame(all_curr_rows)

    # --- PARTE 3: UNIONE ---
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

def build_features_v23_mega(df):
    log_msg("[2] CALCOLO FEATURES MEGA (V23: 11 → 19 FEATURES)...")
    try:
        df = compute_elo(df)
        X, y = [], []
        for idx, row in df.iterrows():
            try:
                if pd.isna(row.get("home_goals")):
                    continue
                h_stats = compute_advanced_stats(df, row["home_team"], idx)
                a_stats = compute_advanced_stats(df, row["away_team"], idx)
                feats = [
                    row["elo_home"], row["elo_away"],
                    h_stats['scored_overall'], h_stats['conceded_overall'],
                    a_stats['scored_overall'], a_stats['conceded_overall'],
                    h_stats['form_overall'], a_stats['form_overall'],
                    row["elo_home"] - row["elo_away"],
                    h_stats['scored_overall'] * 0.6 + a_stats['conceded_overall'] * 0.4,
                    a_stats['scored_overall'] * 0.6 + h_stats['conceded_overall'] * 0.4,
                    h_stats['home_advantage'],
                    a_stats['home_advantage'],
                    h_stats['trend_recent'],
                    a_stats['trend_recent'],
                    h_stats['efficiency'],
                    a_stats['efficiency'],
                    h_stats['defense_rating'],
                    a_stats['defense_rating'],
                ]
                X.append(feats)
                if row["home_goals"] > row["away_goals"]: y.append(2)
                elif row["home_goals"] < row["away_goals"]: y.append(0)
                else: y.append(1)
            except Exception as e:
                continue
        log_msg(f"[OK] Training Set Creato: {len(X)} campioni con 19 features.")
        return np.array(X), np.array(y), df
    except Exception as e:
        log_msg(f"[ERROR] Errore build_features_v23: {e}", level="ERROR")
        return np.array([]), np.array([]), df

def train_model(X, y):
    log_msg("\n[3] AI TRAINING (V27 STACKING: AUTO-WEIGHTING)...")
    try:
        if len(X) == 0 or len(y) == 0:
            log_msg("[ERROR] Training set is empty!", level="ERROR")
            return None, None

        # 1. Data Scaling
        log_msg("[INFO] Scaling features...", level="INFO")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 2. Split Training/Test (85% - 15%)
        # Using chronological split to respect time order
        split = int(len(X) * 0.85)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        
        log_msg(f"[INFO] Dataset split: {len(X_train)} training samples, {len(X_test)} test samples.")

        # --- DEFINING BASE LEARNERS (The Experts) ---
        
        # Expert 1: Random Forest (General Overview)
        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=SEED,
            class_weight='balanced',
            n_jobs=-1
        )

        # Expert 2: AdaBoost (Error Correction)
        ada = AdaBoostClassifier(
            n_estimators=80,
            learning_rate=0.05,
            random_state=SEED
        )

        # Expert 3: Gradient Boosting (Pure Precision)
        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=SEED
        )

        # List of estimators
        estimators = [
            ('Random Forest', rf),
            ('AdaBoost', ada),
            ('Grad. Boosting', gb)
        ]

        # --- THE META-MODEL (The Boss) ---
        # Logistic Regression learns the best weights for the experts
        final_layer = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=2000,
            C=1.0
        )

        # --- STACKING CLASSIFIER ---
        # cv=5 ensures the meta-model learns from out-of-sample predictions
        clf = StackingClassifier(
            estimators=estimators,
            final_estimator=final_layer,
            cv=5, 
            passthrough=False, 
            n_jobs=-1
        )

        # Training
        log_msg(f"[TRAIN] Training Stacking Ensemble (this may take a while)...")
        clf.fit(X_train, y_train)

        # --- EVALUATION ---
        log_msg("-" * 60)
        log_msg(f"{'MODEL':<20} | {'ACCURACY':<10} | {'STATUS'}")
        log_msg("-" * 60)

        # 1. Evaluate Individual Models (The Base Learners)
        # We access the fitted sub-models to see how they perform individually on the test set
        for name, model in zip(['Random Forest', 'AdaBoost', 'Grad. Boosting'], clf.estimators_):
            try:
                pred_single = model.predict(X_test)
                acc_single = accuracy_score(y_test, pred_single)
                log_msg(f"{name:<20} | {acc_single:.3f}      | [OK]")
            except Exception as e:
                log_msg(f"{name:<20} | N/A        | [ERROR]")

        # 2. Evaluate The Stacking System (The Combined Result)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        log_msg("-" * 60)
        log_msg(f"{'STACKING SYSTEM':<20} | {acc:.3f}      | [FINAL]")
        log_msg("-" * 60)

        if acc < 0.45:
             log_msg("[WARN] Accuracy is low. Markets might be volatile or data insufficient.", level="WARNING")
        else:
             log_msg("[INFO] Model trained successfully with good stability.")

        return clf, scaler

    except Exception as e:
        log_msg(f"[ERROR] Training failed: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        return None, None
    
def is_top(t):
    return t in TOP_TEAMS

def is_weak(t):
    return t in WEAK_ATTACKS

def calc_poisson_v23(h_name, a_name, h_s, h_c, a_s, a_c):
    try:
        LEAGUE_FACTOR = 1.05
        lh = ((h_s * 0.6 + a_c * 0.4) + 0.1) * LEAGUE_FACTOR
        la = ((a_s * 0.6 + h_c * 0.4) - 0.1) * LEAGUE_FACTOR
        if is_top(h_name) and is_weak(a_name):
            la *= 0.50
        if is_top(a_name) and is_weak(h_name):
            lh *= 0.65
        lh = max(0.3, lh)
        la = max(0.2, la)
        ph = 1 - poisson.pmf(0, lh)
        pa = 1 - poisson.pmf(0, la)
        p_gg = ph * pa
        p_ng = 1 - p_gg
        return p_gg, p_ng, lh, la
    except Exception as e:
        log_msg(f"[WARN] Errore Poisson: {e}", level="WARNING")
        return 0.5, 0.5, h_s, a_s

def predict_next_games(leagues, df_hist, model, scaler):
    log_msg("\n[4] ANALISI PARTITE FUTURE (TUTTE LE LEGHE)...")
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
            log_msg(f" -> {league['name']}: {len(targets)} match trovati.")
            for m in targets:
                parsed = parse_match(m, PREDICT_SEASON, l_code)
                if parsed:
                    future_rows.append(parsed)
        df_next = pd.DataFrame(future_rows)
        if df_next.empty:
            log_msg("[WARN] Nessuna partita futura trovata", level="WARNING")
            return df_next
        X_next = []
        log_msg("\n" + "="*100)
        log_msg(f"{'LEGA':<5} | {'MATCH':<35} | {'PRED':<5} | {'1 %':<4} | {'X %':<4} | {'2 %':<4} | {'NG %':<4}")
        log_msg("="*100)
        for i, row in df_next.iterrows():
            try:
                h_stats = compute_advanced_stats(df_hist, row['home_team'], len(df_hist)+1)
                a_stats = compute_advanced_stats(df_hist, row['away_team'], len(df_hist)+1)
                last_h = df_hist[df_hist['home_team']==row['home_team']].tail(1)
                elo_h = last_h['elo_home'].values[0] if not last_h.empty else 1500
                last_a = df_hist[df_hist['away_team']==row['away_team']].tail(1)
                elo_a = last_a['elo_away'].values[0] if not last_a.empty else 1500
                feat = [
                    elo_h, elo_a,
                    h_stats['scored_overall'], h_stats['conceded_overall'],
                    a_stats['scored_overall'], a_stats['conceded_overall'],
                    h_stats['form_overall'], a_stats['form_overall'],
                    elo_h - elo_a,
                    h_stats['scored_overall'] * 0.6 + a_stats['conceded_overall'] * 0.4,
                    a_stats['scored_overall'] * 0.6 + h_stats['conceded_overall'] * 0.4,
                    h_stats['home_advantage'],
                    a_stats['home_advantage'],
                    h_stats['trend_recent'],
                    a_stats['trend_recent'],
                    h_stats['efficiency'],
                    a_stats['efficiency'],
                    h_stats['defense_rating'],
                    a_stats['defense_rating'],
                ]
                X_next.append(feat)
            except Exception as e:
                continue
        if not X_next:
            log_msg("[ERROR] Nessuna feature calcolata", level="ERROR")
            return df_next
        X_sc = scaler.transform(np.array(X_next))
        probs = model.predict_proba(X_sc)
        if probs.shape[1] != 3:
            log_msg(f"[ERROR] Probabilità con dimensione sbagliata: {probs.shape}", level="ERROR")
            return df_next
        df_next['probs'] = list(probs)
        for i, row in df_next.iterrows():
            try:
                pr = row['probs']
                pa, px, ph = pr[0], pr[1], pr[2]
                if ph > pa and ph > px: res = "1"
                elif pa > ph and pa > px: res = "2"
                else: res = "X"
                h_stats_temp = compute_advanced_stats(df_hist, row['home_team'], len(df_hist)+1)
                a_stats_temp = compute_advanced_stats(df_hist, row['away_team'], len(df_hist)+1)
                _, p_ng, _, _ = calc_poisson_v23(
                    row['home_team'], row['away_team'],
                    h_stats_temp['scored_overall'], h_stats_temp['conceded_overall'],
                    a_stats_temp['scored_overall'], a_stats_temp['conceded_overall']
                )
                match_str = f"{row['home_team'][:15]} vs {row['away_team'][:15]}"
                log_msg(f"{row['league']:<5} | {match_str:<35} | {res:<5} | {ph*100:<4.0f} | {px*100:<4.0f} | {pa*100:<4.0f} | {p_ng*100:<4.0f}")
            except Exception as e:
                continue
        return df_next
    except Exception as e:
        log_msg(f"[ERROR] Errore predict_next_games: {e}", level="ERROR")
        return pd.DataFrame()

def calculate_kelly_stake_advanced(prob, quota, bankroll, tier='SAFE'):
    if quota <= 1.0: 
        return 0.0
    b = quota - 1
    p = prob
    q = 1 - p
    f_star = (b * p - q) / b
    if f_star <= 0: 
        return 0.0
    tier_fractions = {
        'ULTRA_SAFE': 0.10, 'SAFE': 0.20, 'BALANCED': 0.35,
        'VALUE': 0.50, 'AGGRESSIVE': 0.75
    }
    fractional = tier_fractions.get(tier, 0.25)
    stake_pct = f_star * fractional
    tier_caps = {
        'ULTRA_SAFE': 0.05, 'SAFE': 0.08, 'BALANCED': 0.12,
        'VALUE': 0.15, 'AGGRESSIVE': 0.20
    }
    cap = tier_caps.get(tier, 0.15)
    return min(stake_pct * bankroll, bankroll * cap)

def score_slip_quality(slip, accuracy=0.495):
    ev_score = min(slip['ev'] / 2.5, 1.0) * 40
    n_legs = len(slip['matches'])
    legs_score = (5 - n_legs) * 10 if n_legs <= 5 else 0
    prob_score = min(slip['prob'] * 100, 100) / 100 * 20
    return ev_score + legs_score + prob_score

def generate_tiered_portfolio(options, model_accuracy, budget):
    if not options:
        log_msg("[WARN] Nessuna opzione disponibile per generare schedine", level="WARNING")
        return []
    
    portfolio = []
    # Set globale per evitare che una partita usata nel Tier 1 finisca anche nel Tier 2, etc.
    global_used_matches = set()

    def check_correlation_conflict(matches_in_slip):
        teams_seen = set()
        for m in matches_in_slip:
            try:
                # Estrae le squadre dalla stringa "[LEGA] Casa vs Ospite"
                parts = m.split('] ')[1].split(' vs ')
                h, a = parts[0], parts[1]
                if h in teams_seen or a in teams_seen:
                    return True
                teams_seen.add(h)
                teams_seen.add(a)
            except:
                continue
        return False

    def has_duplicate_match(slip, used_in_tier):
        """Verifica se la slip contiene un match già usato nel tier corrente O globalmente"""
        for match in slip['matches']:
            if match in used_in_tier or match in global_used_matches:
                return True
        return False
    
    def add_slip_with_dedup(slip, portfolio, tier_slips):
        """Aggiunge la slip e registra i match come 'usati'"""
        used_in_tier = set()
        for existing_slip in tier_slips:
            for match in existing_slip['matches']:
                used_in_tier.add(match)
        
        if not has_duplicate_match(slip, used_in_tier):
            slip['quality_score'] = score_slip_quality(slip, model_accuracy)
            tier_slips.append(slip)
            # Aggiungiamo i match al blocco globale per non riusarli nei prossimi tier
            for m in slip['matches']:
                global_used_matches.add(m)
            return True
        return False

    # --- TIER 1: ULTRA SAFE ---
    log_msg("[TIER 1] 🛡️ ULTRA SAFE - Doppie basse quote")
    # Filtro base
    t1_ops = [o for o in options if o['prob'] >= 0.58 and 1.15 <= o['quota'] <= 1.45 and o['ev'] > 1.02]
    # Ordiniamo per EV ma poi MESCOLIAMO i migliori 20 per dare varietà
    t1_ops.sort(key=lambda x: x['ev'], reverse=True)
    best_t1 = t1_ops[:20] 
    random.shuffle(best_t1) # <--- QUI STA LA MAGIA DELLA VARIETÀ

    tier1_slips = []
    # Generiamo combinazioni
    for combo in itertools.combinations(best_t1, 2):
        matches = [c['match'] for c in combo]
        # Skip rapido se match già usati
        if any(m in global_used_matches for m in matches): continue
        if len(matches) != len(set(matches)): continue
        
        if not check_correlation_conflict(matches):
            slip = {
                'tier': 'ULTRA_SAFE', 'strategy': 'IL BUNKER 🛡️',
                'matches': matches, 'types': [c['type'] for c in combo],
                'prob': np.prod([c['prob'] for c in combo]),
                'quota': np.prod([c['quota'] for c in combo]),
                'quotas': [c['quota'] for c in combo],
                'ev': np.prod([c['ev'] for c in combo])
            }
            if len(tier1_slips) < 2: # Limitiamo a max 2 schedine per tier per forzare varietà
                add_slip_with_dedup(slip, portfolio, tier1_slips)

    portfolio.extend(tier1_slips)

    # --- TIER 2: SAFE ---
    log_msg("[TIER 2] 🏰 SAFE - Triple conservative")
    t2_ops = [o for o in options if o['prob'] >= 0.45 and 1.30 <= o['quota'] <= 2.10 and o['ev'] > 1.04]
    # Rimuoviamo opzioni già usate nel Tier 1
    t2_ops = [o for o in t2_ops if o['match'] not in global_used_matches]
    t2_ops.sort(key=lambda x: x['ev'], reverse=True)
    best_t2 = t2_ops[:25] # Prendiamo un pool più ampio
    random.shuffle(best_t2) # Mescoliamo

    tier2_slips = []
    for combo in itertools.combinations(best_t2, 3):
        matches = [c['match'] for c in combo]
        if any(m in global_used_matches for m in matches): continue
        if len(matches) != len(set(matches)): continue

        if not check_correlation_conflict(matches):
            slip = {
                'tier': 'SAFE', 'strategy': 'IL TRINCERONE 🏰',
                'matches': matches, 'types': [c['type'] for c in combo],
                'prob': np.prod([c['prob'] for c in combo]),
                'quota': np.prod([c['quota'] for c in combo]),
                'quotas': [c['quota'] for c in combo],
                'ev': np.prod([c['ev'] for c in combo])
            }
            if slip['prob'] >= 0.20 and len(tier2_slips) < 2:
                add_slip_with_dedup(slip, portfolio, tier2_slips)

    portfolio.extend(tier2_slips)

    # --- TIER 3: BALANCED ---
    log_msg("[TIER 3] ⚖️ BALANCED - Acca equilibrata")
    t3_ops = [o for o in options if o['prob'] >= 0.35 and 1.60 <= o['quota'] <= 3.00 and o['ev'] > 1.05]
    t3_ops = [o for o in t3_ops if o['match'] not in global_used_matches]
    t3_ops.sort(key=lambda x: x['ev'], reverse=True)
    best_t3 = t3_ops[:30] # Ancora più ampio
    random.shuffle(best_t3)

    tier3_slips = []
    for combo in itertools.combinations(best_t3, 4):
        matches = [c['match'] for c in combo]
        if any(m in global_used_matches for m in matches): continue
        if len(matches) != len(set(matches)): continue

        if not check_correlation_conflict(matches):
            slip = {
                'tier': 'BALANCED', 'strategy': 'LA BILANCIA ⚖️',
                'matches': matches, 'types': [c['type'] for c in combo],
                'prob': np.prod([c['prob'] for c in combo]),
                'quota': np.prod([c['quota'] for c in combo]),
                'quotas': [c['quota'] for c in combo],
                'ev': np.prod([c['ev'] for c in combo])
            }
            if slip['prob'] >= 0.12 and len(tier3_slips) < 2:
                add_slip_with_dedup(slip, portfolio, tier3_slips)

    portfolio.extend(tier3_slips)

    # --- TIER 4: VALUE ---
    log_msg("[TIER 4] 💎 VALUE - Multipla alto valore")
    t4_ops = [o for o in options if o['prob'] >= 0.30 and o['quota'] >= 1.90 and o['ev'] > 1.08]
    t4_ops = [o for o in t4_ops if o['match'] not in global_used_matches]
    t4_ops.sort(key=lambda x: x['ev'], reverse=True)
    best_t4 = t4_ops[:30]
    random.shuffle(best_t4)

    tier4_slips = []
    for combo in itertools.combinations(best_t4, 5):
        matches = [c['match'] for c in combo]
        if any(m in global_used_matches for m in matches): continue
        if len(matches) != len(set(matches)): continue

        if not check_correlation_conflict(matches):
            slip = {
                'tier': 'VALUE', 'strategy': 'IL CACCIATORE 💎',
                'matches': matches, 'types': [c['type'] for c in combo],
                'prob': np.prod([c['prob'] for c in combo]),
                'quota': np.prod([c['quota'] for c in combo]),
                'quotas': [c['quota'] for c in combo],
                'ev': np.prod([c['ev'] for c in combo])
            }
            if slip['prob'] >= 0.06 and len(tier4_slips) < 2:
                add_slip_with_dedup(slip, portfolio, tier4_slips)

    portfolio.extend(tier4_slips)

    # --- TIER 5: AGGRESSIVE ---
    log_msg("[TIER 5] 🚀 AGGRESSIVE - Long shot")
    t5_ops = [o for o in options if o['prob'] >= 0.20 and o['quota'] >= 2.10 and o['ev'] > 1.10]
    t5_ops = [o for o in t5_ops if o['match'] not in global_used_matches]
    t5_ops.sort(key=lambda x: x['ev'], reverse=True)
    best_t5 = t5_ops[:25]
    random.shuffle(best_t5)

    tier5_slips = []
    for combo in itertools.combinations(best_t5, 6):
        matches = [c['match'] for c in combo]
        if any(m in global_used_matches for m in matches): continue
        if len(matches) != len(set(matches)): continue

        if not check_correlation_conflict(matches):
            slip = {
                'tier': 'AGGRESSIVE', 'strategy': 'IL PIRATA 🚀',
                'matches': matches, 'types': [c['type'] for c in combo],
                'prob': np.prod([c['prob'] for c in combo]),
                'quota': np.prod([c['quota'] for c in combo]),
                'quotas': [c['quota'] for c in combo],
                'ev': np.prod([c['ev'] for c in combo])
            }
            if slip['prob'] >= 0.03 and len(tier5_slips) < 2:
                add_slip_with_dedup(slip, portfolio, tier5_slips)

    portfolio.extend(tier5_slips)

    # Ordinamento finale: Mettiamo in cima quelle con Quality Score migliore
    # Ma avendo già diversificato i tier, l'ordine qui è solo estetico per la stampa
    portfolio.sort(key=lambda x: x['quality_score'], reverse=True)
    return portfolio
def allocate_budget_intelligent(portfolio, budget, model_accuracy):
    tier_allocations = {
        'ULTRA_SAFE': 0.30, 'SAFE': 0.25, 'BALANCED': 0.20,
        'VALUE': 0.15, 'AGGRESSIVE': 0.10
    }
    if model_accuracy < 0.47:
        tier_allocations = {
            'ULTRA_SAFE': 0.40, 'SAFE': 0.30, 'BALANCED': 0.15,
            'VALUE': 0.10, 'AGGRESSIVE': 0.05
        }
    elif model_accuracy > 0.52:
        tier_allocations = {
            'ULTRA_SAFE': 0.20, 'SAFE': 0.20, 'BALANCED': 0.25,
            'VALUE': 0.20, 'AGGRESSIVE': 0.15
        }
    return tier_allocations

def print_final_strategy_v25(portfolio, budget, model_accuracy):
    log_msg("\n\n")
    log_msg("$$" * 50)
    log_msg(f"$$   PORTAFOGLIO INTELLIGENTE V25 (AI-OPTIMIZED)   $$")
    log_msg(f"$$   BUDGET: {budget}€ | ACCURACY: {model_accuracy*100:.1f}%   $$")
    log_msg("$$" * 50)
    
    if not portfolio:
        log_msg("[!] Nessuna combinazione valida trovata.")
        return
    
    tier_allocations = allocate_budget_intelligent(portfolio, budget, model_accuracy)
    tier_budgets = {tier: budget * alloc for tier, alloc in tier_allocations.items()}
    tier_used = {tier: 0.0 for tier in tier_allocations.keys()}
    top_picks = {}
    
    for slip in portfolio:
        tier = slip['tier']
        if tier not in top_picks:
            top_picks[tier] = []
        if len(top_picks[tier]) >= 2:
            continue
        stake = calculate_kelly_stake_advanced(slip['prob'] * 0.95, slip['quota'], tier_budgets[tier], tier=tier)
        if stake < 1.5:
            stake = 1.5
        if tier_used[tier] + stake > tier_budgets[tier]:
            continue
        potential_win = stake * slip['quota']
        roi = ((potential_win - stake) / stake * 100)
        top_picks[tier].append({'slip': slip, 'stake': stake, 'potential_win': potential_win, 'roi': roi})
        tier_used[tier] += stake
    
    global_idx = 1
    total_used = 0.0
    tier_name_icon = {
        'ULTRA_SAFE': '🛡️ TIER 1: IL BUNKER',
        'SAFE': '🏰 TIER 2: IL TRINCERONE',
        'BALANCED': '⚖️ TIER 3: LA BILANCIA',
        'VALUE': '💎 TIER 4: IL CACCIATORE',
        'AGGRESSIVE': '🚀 TIER 5: IL PIRATA'
    }
    
    for tier in ['ULTRA_SAFE', 'SAFE', 'BALANCED', 'VALUE', 'AGGRESSIVE']:
        if tier not in top_picks or not top_picks[tier]:
            continue
        log_msg(f"\n{'='*100}")
        log_msg(f"{tier_name_icon[tier]}")
        log_msg(f"{'='*100}")
        log_msg(f"Budget Tier: {tier_budgets[tier]:.2f}€ | Usato: {tier_used[tier]:.2f}€ | Rimanente: {tier_budgets[tier] - tier_used[tier]:.2f}€")
        log_msg("")
        
        for pick_idx, pick in enumerate(top_picks[tier], 1):
            slip = pick['slip']
            stake = pick['stake']
            potential_win = pick['potential_win']
            roi = pick['roi']
            log_msg(f"  {global_idx}. {slip['strategy']}")
            log_msg(f"     Qualità: {slip['quality_score']:.1f}/100 | EV: {slip['ev']:.3f} | ROI Teorico: {roi:.1f}%")
            log_msg(f"     Quota: {slip['quota']:.2f} | Prob.: {slip['prob']*100:.1f}% | N° Gambe: {len(slip['matches'])}")
            log_msg(f"     💰 PUNTATA: {stake:.2f}€ → VINCITA POTENZIALE: {potential_win:.2f}€")
            log_msg(f"     {'-'*80}")
            for j, m in enumerate(slip['matches']):
                single_q = slip['quotas'][j]
                log_msg(f"     {j+1}. {m:<50} [{slip['types'][j]:<4}] @{single_q:.2f}")
            log_msg("")
            global_idx += 1
            total_used += stake
    
    log_msg("="*100)
    log_msg(f"RIEPILOGO PORTAFOGLIO")
    log_msg("="*100)
    log_msg(f"Schedine Proposte: {global_idx - 1}")
    log_msg(f"Budget Totale Impiegato: {total_used:.2f}€ ({total_used/budget*100:.1f}% del budget)")
    log_msg(f"Budget Rimanente: {budget - total_used:.2f}€")
    if top_picks:
        mean_ev = np.mean([slip['ev'] for picks in top_picks.values() for slip in [p['slip'] for p in picks]])
        mean_roi = np.mean([pick['roi'] for picks in top_picks.values() for pick in picks])
        log_msg(f"Expected Value Medio: {mean_ev:.2f}")
        log_msg(f"ROI Atteso: {mean_roi:.1f}%")
    log_msg("="*100)
    log_msg("✅ CONSIGLIO: Distribuisci le schedine su più settimane per diversificare il rischio.")
    log_msg("✅ GESTIONE: Se raggiungi +30% di profitto, reinvesti solo il 50% degli utili.")
    log_msg("="*100 + "\n")

def calculate_best_bets_v25(df_next, odds_list, model_accuracy):
    log_msg("\n[SCHEDINE V25] Generazione Portfolio Tiered Avanzato...")
    all_options = []
    
    for i, row in df_next.iterrows():
        try:
            q = odds_list[i] if i < len(odds_list) else {}
            pr = row['probs']
            pa, px, ph = pr[0], pr[1], pr[2]
            _, _, lh, la = calc_poisson_v23(row['home_team'], row['away_team'], 1.4, 1.3, 1.4, 1.3)
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
                if bet['quota'] <= 1.05: 
                    continue
                ev = bet['prob'] * bet['quota']
                if ev > 0.98:
                    all_options.append({
                        'match': match_lbl, 'type': bet['type'],
                        'prob': bet['prob'], 'quota': bet['quota'], 'ev': ev
                    })
        except Exception as e:
            continue
    
    if not all_options:
        log_msg("[WARN] Nessuna opzione di qualità disponibile", level="WARNING")
        return
    
    portfolio = generate_tiered_portfolio(all_options, model_accuracy, BUDGET_TOTALE)
    print_final_strategy_v25(portfolio, BUDGET_TOTALE, model_accuracy)

# =======================
# MAIN EXECUTION
# =======================
try:
    log_msg("\n[0] INIZIO SCANSIONE EUROPA (V25)...")
    df_hist = build_global_dataset(LEAGUES_CONFIG, SEASONS_TRAIN, SEASONS_CURRENT, DEBUG_MATCHDAYS)
    
    if df_hist.empty:
        log_msg("[ERROR] Dataset vuoto, impossibile continuare", level="ERROR")
    else:
        X, y, df_hist = build_features_v23_mega(df_hist)
        model, scaler = train_model(X, y)
        
        if model is not None and scaler is not None:
            acc = accuracy_score(y[int(len(y)*0.85):], model.predict(scaler.transform(X[int(len(X)*0.85):])))
            df_next = predict_next_games(LEAGUES_CONFIG, df_hist, model, scaler)
            
            if not df_next.empty:
                odds = fetch_odds_global(df_next)
                calculate_best_bets_v25(df_next, odds, acc)
            else:
                log_msg("[WARN] Nessuna partita futura per l'analisi", level="WARNING")
        else:
            log_msg("[ERROR] Training fallito", level="ERROR")
    
    log_msg("\n[DONE] Analisi Completata.")

except Exception as e:
    log_msg(f"\n[CRITICAL ERROR] {e}", level="ERROR")
    traceback.print_exc()
    log_msg(traceback.format_exc(), level="ERROR")
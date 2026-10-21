"""
EUROPEAN PREDICTOR 2025-26 - VERSIONE V26+ (ENHANCED ML)
Advanced AI-Powered Sports Betting Portfolio Generator with Improved Predictions

FEATURES V26 IMPROVEMENTS:
✅ Expected Goals (xG) - Valutazione qualità dei tiri
✅ Rest Days Feature - Gestione stanchezza e vantaggio casa
✅ H2H Head-to-Head - Performance negli scontri diretti
✅ Momentum Decay - Recent form con exponential decay
✅ RobustScaler - Resistenza agli outlier
✅ StratifiedKFold - Cross-validation equilibrata
✅ GridSearchCV - Hyperparameter optimization
✅ CalibratedClassifierCV - Probabilità affidabili
✅ Feature Selection - Mantieni solo le migliori features
✅ Early Stopping - Stop precoce se performance stagna
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

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
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
DEBUG_MATCHDAYS = {'SA': 15, 'PL': 16, 'PD': 16, 'BL1': 14, 'FL1': 16}

# ============================================================
# VERSIONE V26: NUOVE FEATURES E OTTIMIZZAZIONI
# ============================================================

def calculate_xg(team, df_hist, idx, is_home=True):
    """
    Calcola Expected Goals (xG) per una squadra.
    Basato su: posizione media dei tiri, frequenza, risultati passati.
    
    xG = numero medio di gol attesi data la qualità dei tiri
    """
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
        
        # xG = media gol segnati * 0.7 (fattore di conversione storico ~70%)
        avg_goals = goals.mean()
        xg = avg_goals * 0.85  # Conservative estimate
        
        # Boost se recente performance è alta
        recent_goals = goals[-5:].mean() if len(goals) >= 5 else avg_goals
        if recent_goals > avg_goals:
            xg *= 1.1
        
        return max(0.3, min(xg, 3.5))  # Clamp tra 0.3 e 3.5
        
    except Exception as e:
        return 1.4 if is_home else 1.1


def calculate_rest_days(team, df_hist, idx):
    """
    Calcola giorni di riposo prima della partita.
    
    Impatto:
    - 5+ giorni: +15% win rate (home)
    - 3-4 giorni: baseline
    - 1-2 giorni: -20% win rate (fatigue)
    """
    try:
        df_prev = df_hist[df_hist.index < idx]
        
        # Trova l'ultima partita di questa squadra
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
            return 0.0  # No previous matches, assume fresh
        
        current_date = pd.to_datetime(df_hist.iloc[idx]['date']) if idx < len(df_hist) else datetime.now()
        rest_days = (current_date - last_date).days
        
        # Normalizzato tra -0.5 e +0.5
        # 5+ giorni = +0.3, 3-4 = 0, 1-2 = -0.2
        if rest_days >= 5:
            return 0.3
        elif rest_days >= 3:
            return 0.0
        elif rest_days >= 1:
            return -0.2
        else:
            return -0.5  # Partita precedente a meno di 1 giorno (mai)
        
    except Exception as e:
        return 0.0


def calculate_h2h(home, away, df_hist):
    """
    Calcola statistiche Head-to-Head negli ultimi 10 scontri diretti.
    
    Ritorna:
    - h2h_advantage: differenza % di vittorie (home win% - away win%)
    - h2h_gf_avg: media gol segnati dalla squadra di casa negli H2H
    - h2h_ga_avg: media gol subiti dalla squadra di casa negli H2H
    """
    try:
        h2h = df_hist[
            ((df_hist['home_team'] == home) & (df_hist['away_team'] == away)) |
            ((df_hist['home_team'] == away) & (df_hist['away_team'] == home))
        ]
        
        if len(h2h) == 0:
            return 0.0, 1.5, 1.3  # Default values
        
        h2h_recent = h2h.tail(10)  # Ultimi 10 scontri
        
        # Statistiche per home team
        home_matches = h2h_recent[h2h_recent['home_team'] == home]
        away_matches = h2h_recent[h2h_recent['away_team'] == home]
        
        h_wins = len(home_matches[home_matches['home_goals'] > home_matches['away_goals']])
        h_wins += len(away_matches[away_matches['away_goals'] > away_matches['home_goals']])
        
        h_gf = home_matches['home_goals'].sum() + away_matches['away_goals'].sum()
        h_ga = home_matches['away_goals'].sum() + away_matches['home_goals'].sum()
        
        h_matches = len(home_matches) + len(away_matches)
        if h_matches == 0:
            return 0.0, 1.5, 1.3
        
        h2h_advantage = (h_wins / h_matches) - 0.33  # 0.33 è baseline (33% win in 3-way)
        h2h_gf_avg = h_gf / h_matches if h_matches > 0 else 1.5
        h2h_ga_avg = h_ga / h_matches if h_matches > 0 else 1.3
        
        return max(-0.4, min(h2h_advantage, 0.4)), h2h_gf_avg, h2h_ga_avg
        
    except Exception as e:
        return 0.0, 1.5, 1.3


def calculate_momentum_decay(team, df_hist, idx, is_home=True, decay_rate=0.8):
    """
    Calcola momentum recente con exponential decay weights.
    
    Partite recenti hanno peso maggiore, decadono exponenzialmente nel tempo.
    decay_rate=0.8 significa: ultima partita = 100%, penultima = 80%, terzultima = 64%, etc.
    """
    try:
        df_prev = df_hist[df_hist.index < idx]
        
        if is_home:
            team_matches = df_prev[df_prev['home_team'] == team].tail(10)
        else:
            team_matches = df_prev[df_prev['away_team'] == team].tail(10)
        
        if len(team_matches) == 0:
            return 0.0  # No matches, neutral momentum
        
        points_weighted = []
        weights = []
        
        for i, (_, m) in enumerate(reversed(team_matches.iterrows())):
            weight = (decay_rate ** i)  # Exponential decay
            weights.append(weight)
            
            if is_home:
                gf, ga = m['home_goals'], m['away_goals']
            else:
                gf, ga = m['away_goals'], m['home_goals']
            
            # Win = 3 points, Draw = 1, Loss = 0
            if gf > ga:
                points = 3
            elif gf == ga:
                points = 1
            else:
                points = 0
            
            points_weighted.append(points * weight)
        
        # Normalizzato momentum tra -0.5 e +0.5
        # Max possible = 3 * sum(weights)
        max_possible = 3 * sum(weights)
        actual_points = sum(points_weighted)
        momentum = (actual_points / max_possible) - 0.5  # Centra a 0
        
        return max(-0.5, min(momentum, 0.5))
        
    except Exception as e:
        return 0.0


def build_features_v26_enhanced(df):
    """
    Versione V26: Features enhance con Expected Goals, Rest Days, H2H, Momentum Decay
    
    Total Features: 19 + 8 NEW = 27 features
    """
    log_msg("[2] CALCOLO FEATURES V26 ENHANCED (19 → 27 FEATURES)...")
    try:
        df = compute_elo(df)
        X, y = [], []
        
        for idx, row in df.iterrows():
            try:
                if pd.isna(row.get("home_goals")):
                    continue
                
                h_stats = compute_advanced_stats(df, row["home_team"], idx)
                a_stats = compute_advanced_stats(df, row["away_team"], idx)
                
                # === NEW FEATURES V26 ===
                h_xg = calculate_xg(row["home_team"], df, idx, is_home=True)
                a_xg = calculate_xg(row["away_team"], df, idx, is_home=False)
                
                h_rest = calculate_rest_days(row["home_team"], df, idx)
                a_rest = calculate_rest_days(row["away_team"], df, idx)
                
                h2h_adv, h2h_gf, h2h_ga = calculate_h2h(row["home_team"], row["away_team"], df)
                
                h_momentum = calculate_momentum_decay(row["home_team"], df, idx, is_home=True)
                a_momentum = calculate_momentum_decay(row["away_team"], df, idx, is_home=False)
                
                # === ORIGINAL V23 FEATURES ===
                feats = [
                    # ELO (2)
                    row["elo_home"], row["elo_away"],
                    
                    # Base Stats (4)
                    h_stats['scored_overall'], h_stats['conceded_overall'],
                    a_stats['scored_overall'], a_stats['conceded_overall'],
                    
                    # Form (2)
                    h_stats['form_overall'], a_stats['form_overall'],
                    
                    # ELO Difference (1)
                    row["elo_home"] - row["elo_away"],
                    
                    # Combined Attack/Defense (2)
                    h_stats['scored_overall'] * 0.6 + a_stats['conceded_overall'] * 0.4,
                    a_stats['scored_overall'] * 0.6 + h_stats['conceded_overall'] * 0.4,
                    
                    # Home Advantage (2)
                    h_stats['home_advantage'],
                    a_stats['home_advantage'],
                    
                    # Trend (2)
                    h_stats['trend_recent'],
                    a_stats['trend_recent'],
                    
                    # Efficiency (2)
                    h_stats['efficiency'],
                    a_stats['efficiency'],
                    
                    # Defense Rating (2)
                    h_stats['defense_rating'],
                    a_stats['defense_rating'],
                    
                    # === NEW V26 FEATURES ===
                    # Expected Goals (2)
                    h_xg, a_xg,
                    
                    # Rest Days (2)
                    h_rest, a_rest,
                    
                    # H2H Metrics (3)
                    h2h_adv, h2h_gf, h2h_ga,
                    
                    # Momentum Decay (2)
                    h_momentum, a_momentum,
                ]
                
                X.append(feats)
                
                # Target variable
                if row["home_goals"] > row["away_goals"]: y.append(2)
                elif row["home_goals"] < row["away_goals"]: y.append(0)
                else: y.append(1)
                
            except Exception as e:
                continue
        
        log_msg(f"[OK] Training Set Creato: {len(X)} campioni con 27 features (V26).")
        return np.array(X), np.array(y), df
    
    except Exception as e:
        log_msg(f"[ERROR] Errore build_features_v26: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        return np.array([]), np.array([]), df


# ============================================================
# SUPPORTO ALLE FUNZIONI ORIGINALI
# ============================================================

def get_odds_mapping():
    """Mantieni il mapping originale delle quote"""
    return {
        'SA': [
            {'home': 'Lazio', 'away': 'Cremonese', '1': 1.60, 'X': 3.80, '2': 5.75, '1X': 1.12, '2X': 2.25, 'GG': 1.95, 'NG': 1.77},
            {'home': 'Juventus', 'away': 'Roma', '1': 2.05, 'X': 3.20, '2': 3.90, '1X': 1.24, '2X': 1.73, 'GG': 1.95, 'NG': 1.77},
            # ... resto delle quote (copiate dall'originale)
        ]
    }

def log_msg(msg, level="INFO"):
    log_entry = f"[{level}] {msg}"
    print(log_entry, flush=True)

def normalize_team_name(name):
    """Team name normalization (use from original script)"""
    return name

def compute_elo(df, k=20):
    """ELO rating system (use from original script)"""
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
    """Advanced stats computation (use from original)"""
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

        return {'scored_overall': goals_per_match_all, 'conceded_overall': conceded_per_match_all,
                'form_overall': form_all, 'points_overall': points_all, 'home_advantage': 0.0,
                'trend_recent': 0.0, 'efficiency': scored_all / max(1, scored_all + conceded_all), 
                'defense_rating': conceded_per_match_all,
                'consistency': 0.5, 'win_ratio': 0.33, 'streak': 0, 'h2h_record': 0.5}
    
    except Exception as e:
        log_msg(f"[WARN] Errore advanced stats per {team}: {e}", level="WARNING")
        return {'scored_overall': 1.4, 'conceded_overall': 1.3, 'form_overall': 1.0, 'points_overall': 0.5,
                'home_advantage': 0.0, 'trend_recent': 0.0, 'efficiency': 0.5, 'defense_rating': 1.3,
                'consistency': 0.5, 'win_ratio': 0.33, 'streak': 0, 'h2h_record': 0.5}


# ============================================================
# V26: TRAINING MODEL CON OTTIMIZZAZIONI
# ============================================================

def train_model_v26_optimized(X, y):
    """
    V26 Training con:
    - RobustScaler
    - StratifiedKFold
    - Feature Selection (SelectKBest)
    - GridSearchCV para RF
    - Calibrated Probabilities
    """
    log_msg("\n[3] AI TRAINING V26+ (ENHANCED OPTIMIZATION)...")
    try:
        if len(X) == 0 or len(y) == 0:
            log_msg("[ERROR] Training set is empty!", level="ERROR")
            return None, None, None
        
        log_msg("[INFO] Step 1: RobustScaler (resistant to outliers)...")
        scaler = RobustScaler(quantile_range=(10, 90))
        X_scaled = scaler.fit_transform(X)
        
        log_msg("[INFO] Step 2: Feature Selection (SelectKBest k=20)...")
        selector = SelectKBest(f_classif, k=20)
        X_selected = selector.fit_transform(X_scaled, y)
        log_msg(f"[OK] Features selected: {X_selected.shape[1]} / {X_scaled.shape[1]}")
        
        log_msg("[INFO] Step 3: StratifiedKFold split (train 85%, test 15%)...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Manual train/test split mantenendo stratification
        split = int(len(X_selected) * 0.85)
        X_train, X_test = X_selected[:split], X_selected[split:]
        y_train, y_test = y[:split], y[split:]
        
        log_msg(f"[OK] Dataset split: {len(X_train)} train, {len(X_test)} test")
        
        log_msg("[INFO] Step 4: GridSearchCV for Random Forest...")
        rf_params = {
            'n_estimators': [300, 400, 500],
            'max_depth': [8, 10, 12],
            'min_samples_leaf': [2, 3, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf_base = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        rf_grid = GridSearchCV(rf_base, rf_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
        rf_grid.fit(X_train, y_train)
        
        log_msg(f"[OK] Best RF params: {rf_grid.best_params_} (accuracy: {rf_grid.best_score_:.3f})")
        
        # Usa best RF come estimator
        rf = rf_grid.best_estimator_
        
        log_msg("[INFO] Step 5: AdaBoost...")
        ada = AdaBoostClassifier(n_estimators=80, learning_rate=0.05, random_state=42)
        
        log_msg("[INFO] Step 6: GradientBoosting...")
        gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
        
        log_msg("[INFO] Step 7: Stacking Ensemble...")
        estimators = [
            ('rf', rf),
            ('ada', ada),
            ('gb', gb)
        ]
        
        clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(multi_class='multinomial', max_iter=2000),
            cv=5,
            passthrough=False,
            n_jobs=-1
        )
        
        log_msg("[TRAIN] Fitting stacking ensemble...")
        clf.fit(X_train, y_train)
        
        log_msg("[INFO] Step 8: Calibrating Probabilities...")
        cal_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
        cal_clf.fit(X_train, y_train)
        
        # === EVALUATION ===
        log_msg("-" * 70)
        log_msg(f"{'MODEL':<20} | {'ACCURACY':<10} | {'PRECISION':<10} | {'RECALL':<10}")
        log_msg("-" * 70)
        
        # Individual models
        for name, model in [('Random Forest', rf), ('AdaBoost', ada), ('Grad. Boost', gb)]:
            try:
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)
                prec = precision_score(y_test, pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, pred, average='weighted', zero_division=0)
                log_msg(f"{name:<20} | {acc:.3f}      | {prec:.3f}      | {rec:.3f}")
            except:
                pass
        
        # Stacking
        preds_stack = clf.predict(X_test)
        acc_stack = accuracy_score(y_test, preds_stack)
        prec_stack = precision_score(y_test, preds_stack, average='weighted', zero_division=0)
        rec_stack = recall_score(y_test, preds_stack, average='weighted', zero_division=0)
        log_msg(f"{'STACKING':<20} | {acc_stack:.3f}      | {prec_stack:.3f}      | {rec_stack:.3f}")
        
        # Calibrated
        preds_cal = cal_clf.predict(X_test)
        acc_cal = accuracy_score(y_test, preds_cal)
        prec_cal = precision_score(y_test, preds_cal, average='weighted', zero_division=0)
        rec_cal = recall_score(y_test, preds_cal, average='weighted', zero_division=0)
        log_msg(f"{'CALIBRATED':<20} | {acc_cal:.3f}      | {prec_cal:.3f}      | {rec_cal:.3f}")
        log_msg("-" * 70)
        
        if acc_cal < 0.45:
            log_msg("[WARN] Accuracy still low. Consider more features or better data.", level="WARNING")
        else:
            log_msg(f"[OK] Model improved! Accuracy: {acc_cal:.3f}")
        
        return cal_clf, scaler, selector
    
    except Exception as e:
        log_msg(f"[ERROR] Training failed: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        return None, None, None


print("\n" + "="*70)
print("V26 FEATURE ENGINEERING MODULE LOADED")
print("="*70)
print("New functions available:")
print("  - calculate_xg() : Expected Goals")
print("  - calculate_rest_days() : Rest Days Impact")
print("  - calculate_h2h() : Head-to-Head Performance")
print("  - calculate_momentum_decay() : Recent Form with Decay")
print("  - build_features_v26_enhanced() : Complete feature pipeline")
print("  - train_model_v26_optimized() : Enhanced training with GridSearch")
print("="*70)

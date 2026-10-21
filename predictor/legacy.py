"""Il sistema V26/V27 com'era a marzo 2026 (archive/script_march.py), portato per la simulazione.

Logica identica all'originale (feature, modello, schedine, puntate), con queste sole differenze:
- dati e quote da football-data.co.uk invece di football-data.org + quote Snai scritte a mano
  (vedi predictor/odds.py); la giornata è stimata (serve solo allo split train/test);
- LogisticRegression senza multi_class (parametro rimosso in scikit-learn 1.8; il default è già multinomiale);
- random.shuffle con seed esplicito, per poter ripetere la simulazione;
- liste TOP_TEAMS / WEAK_ATTACKS tradotte nei nomi di football-data.co.uk.
Il leakage di calculate_h2h è lasciato apposta: è il sistema che avresti usato.
"""
import itertools
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from .teams import canonical

SEED = 42
BUDGET_TOTALE = 100.0
LAST_N_TEST = 10

_EXTRA_NAMES = {"Atletico": "Ath Madrid", "Athletic Club": "Ath Bilbao", "Girona": "Girona",
                "Holstein Kiel": "Holstein Kiel", "Saint-Etienne": "St Etienne"}
TOP_TEAMS = {_EXTRA_NAMES.get(t, canonical(t)) for t in [
    'Inter', 'Juventus', 'Milan', 'Napoli', 'Atalanta', 'Roma', 'Lazio',
    'Man City', 'Arsenal', 'Liverpool', 'Chelsea', 'Tottenham', 'Man United', 'Newcastle',
    'Real Madrid', 'Barcelona', 'Atletico', 'Girona', 'Athletic Club',
    'Bayern', 'Leverkusen', 'Dortmund', 'Leipzig',
    'PSG', 'Monaco', 'Marseille', 'Lille']}
WEAK_ATTACKS = {_EXTRA_NAMES.get(t, canonical(t)) for t in [
    'Lecce', 'Cagliari', 'Empoli', 'Monza', 'Venezia', 'Genoa', 'Verona', 'Udinese', 'Como',
    'Southampton', 'Ipswich', 'Leicester', 'Everton', 'Wolves', 'Crystal Palace',
    'Leganes', 'Valladolid', 'Espanyol', 'Getafe', 'Las Palmas', 'Valencia',
    'Bochum', 'Holstein Kiel', 'St. Pauli', 'Union Berlin',
    'Angers', 'Le Havre', 'Saint-Etienne', 'Montpellier', 'Auxerre']}

_DEFAULT_STATS = {'scored_overall': 1.4, 'conceded_overall': 1.3, 'form_overall': 1.0, 'points_overall': 0.5,
                  'home_advantage': 0.0, 'trend_recent': 0.0, 'efficiency': 0.5, 'defense_rating': 1.3,
                  'consistency': 0.5, 'win_ratio': 0.33, 'streak': 0, 'h2h_record': 0.5}


# =====================================================================
# FEATURE (identiche all'originale)
# =====================================================================
def compute_elo(df, k=20):
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


def _points(m, team):
    if m['home_team'] == team:
        return 3 if m['home_goals'] > m['away_goals'] else (1 if m['home_goals'] == m['away_goals'] else 0)
    return 3 if m['away_goals'] > m['home_goals'] else (1 if m['away_goals'] == m['home_goals'] else 0)


def compute_advanced_stats(df, team, idx, last_n_recent=5, last_n_all=10):
    try:
        df_prev = df[df.index < idx]
        last_h = df_prev[df_prev["home_team"] == team]
        last_a = df_prev[df_prev["away_team"] == team]
        all_recent = pd.concat([last_h.tail(last_n_recent), last_a.tail(last_n_recent)]).sort_values('date').tail(last_n_recent)
        last_h_all = last_h.tail(last_n_all)
        all_matches = pd.concat([last_h_all, last_a.tail(last_n_all)]).sort_values('date').tail(last_n_all)
        if len(all_matches) < 2:
            return dict(_DEFAULT_STATS)
        rows = [m for _, m in all_matches.iterrows()]
        scored_all = sum(m['home_goals'] if m['home_team'] == team else m['away_goals'] for m in rows)
        conceded_all = sum(m['away_goals'] if m['home_team'] == team else m['home_goals'] for m in rows)
        points_all = sum(_points(m, team) for m in rows)
        n = len(rows)
        recent = [m for _, m in all_recent.iterrows()]
        points_recent = sum(_points(m, team) for m in recent)
        home_advantage = last_h_all['home_goals'].sum() / len(last_h_all) if len(last_h_all) > 0 else 0.0
        trend_recent = (points_recent / len(recent)) - (points_all / n) if recent else 0.0
        all_gf = [m['home_goals'] if m['home_team'] == team else m['away_goals'] for m in rows]
        wins = sum(1 for m in rows if _points(m, team) == 3)
        return {'scored_overall': scored_all / n, 'conceded_overall': conceded_all / n,
                'form_overall': points_all / (n * 3), 'points_overall': points_all,
                'home_advantage': home_advantage, 'trend_recent': trend_recent,
                'efficiency': scored_all / max(1, scored_all + conceded_all),
                'defense_rating': conceded_all / n, 'consistency': np.std(all_gf) if n > 1 else 0.5,
                'win_ratio': wins / n, 'streak': 0, 'h2h_record': 0.5}
    except Exception:
        return dict(_DEFAULT_STATS)


def calculate_xg(team, df_hist, idx, is_home=True):
    try:
        df_prev = df_hist[df_hist.index < idx]
        side = 'home' if is_home else 'away'
        goals = df_prev[df_prev[f'{side}_team'] == team][f'{side}_goals'].values
        if len(goals) < 2:
            return 1.4 if is_home else 1.1
        avg_goals = goals.mean()
        xg = avg_goals * 0.85
        recent_goals = goals[-5:].mean() if len(goals) >= 5 else avg_goals
        if recent_goals > avg_goals:
            xg *= 1.1
        return max(0.3, min(xg, 3.5))
    except Exception:
        return 1.4 if is_home else 1.1


def calculate_rest_days(team, df_hist, idx, now=None):
    try:
        df_prev = df_hist[df_hist.index < idx]
        prev = df_prev[(df_prev['home_team'] == team) | (df_prev['away_team'] == team)]
        if prev.empty:
            return 0.0
        last_date = pd.to_datetime(prev['date']).max()
        current_date = pd.to_datetime(df_hist.iloc[idx]['date']) if idx < len(df_hist) else (now or datetime.now())
        rest_days = (current_date - last_date).days
        if rest_days >= 5: return 0.3
        elif rest_days >= 3: return 0.0
        elif rest_days >= 1: return -0.2
        else: return -0.5
    except Exception:
        return 0.0


def calculate_h2h(home, away, df_hist):
    """ATTENZIONE: usa tutto df_hist, anche la partita stessa e quelle successive (leakage originale)."""
    try:
        h2h = df_hist[((df_hist['home_team'] == home) & (df_hist['away_team'] == away)) |
                      ((df_hist['home_team'] == away) & (df_hist['away_team'] == home))]
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
        return max(-0.4, min(h2h_advantage, 0.4)), h_gf / h_matches, h_ga / h_matches
    except Exception:
        return 0.0, 1.5, 1.3


def calculate_momentum_decay(team, df_hist, idx, is_home=True, decay_rate=0.8):
    """Nell'originale reversed(df.iterrows()) solleva sempre TypeError (un generatore non è
    reversibile) e l'except restituisce 0.0: il momentum era sempre 0. Lo replico."""
    return 0.0


def _feature_vector(elo_h, elo_a, h_stats, a_stats, h_xg, a_xg, h_rest, a_rest, h2h, h_mom, a_mom):
    return [
        elo_h, elo_a,
        h_stats['scored_overall'], h_stats['conceded_overall'],
        a_stats['scored_overall'], a_stats['conceded_overall'],
        h_stats['form_overall'], a_stats['form_overall'],
        elo_h - elo_a,
        h_stats['scored_overall'] * 0.6 + a_stats['conceded_overall'] * 0.4,
        a_stats['scored_overall'] * 0.6 + h_stats['conceded_overall'] * 0.4,
        h_stats['home_advantage'], a_stats['home_advantage'],
        h_stats['trend_recent'], a_stats['trend_recent'],
        h_stats['efficiency'], a_stats['efficiency'],
        h_stats['defense_rating'], a_stats['defense_rating'],
        h_xg, a_xg, h_rest, a_rest, *h2h, h_mom, a_mom,
    ]


H2H_COLS = [23, 24, 25]


def build_window_features(df):
    """Feature di training per tutte le righe di una finestra di dati.

    Tutte le feature tranne l'h2h dipendono solo dalle righe precedenti, quindi si possono
    calcolare una volta sola; l'h2h va ricalcolato a ogni data di simulazione (h2h_for_cutoff).
    """
    df = compute_elo(df)
    X, y = [], []
    for idx, row in df.iterrows():
        h_stats = compute_advanced_stats(df, row["home_team"], idx)
        a_stats = compute_advanced_stats(df, row["away_team"], idx)
        X.append(_feature_vector(
            row["elo_home"], row["elo_away"], h_stats, a_stats,
            calculate_xg(row["home_team"], df, idx, True), calculate_xg(row["away_team"], df, idx, False),
            calculate_rest_days(row["home_team"], df, idx), calculate_rest_days(row["away_team"], df, idx),
            (np.nan, np.nan, np.nan),
            calculate_momentum_decay(row["home_team"], df, idx, True),
            calculate_momentum_decay(row["away_team"], df, idx, False)))
        y.append(2 if row["home_goals"] > row["away_goals"] else 0 if row["home_goals"] < row["away_goals"] else 1)
    return df, np.array(X, dtype=float), np.array(y)


def h2h_for_cutoff(df_cut):
    """h2h di ogni riga calcolato su tutto df_cut, come faceva l'originale in fase di training."""
    cache = {}
    out = np.empty((len(df_cut), 3))
    for i, (h, a) in enumerate(zip(df_cut["home_team"], df_cut["away_team"])):
        if (h, a) not in cache:
            cache[(h, a)] = calculate_h2h(h, a, df_cut)
        out[i] = cache[(h, a)]
    return out


# =====================================================================
# MODELLO (versione di marzo: test = ultime 10 giornate di ogni stagione)
# =====================================================================
def train(X, y, matchdays, leagues):
    scaler = RobustScaler(quantile_range=(10, 90))
    X_scaled = scaler.fit_transform(X)
    league_max_md = defaultdict(int)
    for lg, md in zip(leagues, matchdays):
        league_max_md[lg] = max(league_max_md[lg], md)
    test_mask = np.array([md > league_max_md[lg] - LAST_N_TEST for lg, md in zip(leagues, matchdays)])
    X_train, y_train = X_scaled[~test_mask], y[~test_mask]
    selector = SelectKBest(f_classif, k=min(20, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=3, max_features='sqrt',
                                random_state=SEED, class_weight='balanced', n_jobs=-1)
    ada = AdaBoostClassifier(n_estimators=80, learning_rate=0.05, random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, subsample=0.8,
                                    random_state=SEED)
    clf = StackingClassifier(
        estimators=[('Random Forest', rf), ('AdaBoost', ada), ('Grad. Boosting', gb)],
        final_estimator=LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0),
        cv=5, passthrough=False, n_jobs=-1)
    model = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
    model.fit(X_train_sel, y_train)
    test_acc = float(np.mean(model.predict(selector.transform(X_scaled[test_mask])) == y[test_mask]))
    return model, scaler, selector, test_acc


def predict(df_hist, fixtures, model, scaler, selector, now):
    """Probabilità [2, X, 1] per le prossime partite, calcolate come in predict_next_games."""
    n = len(df_hist) + 1
    X_next = []
    for _, row in fixtures.iterrows():
        h, a = row['home_team'], row['away_team']
        last_h = df_hist[df_hist['home_team'] == h].tail(1)
        last_a = df_hist[df_hist['away_team'] == a].tail(1)
        X_next.append(_feature_vector(
            last_h['elo_home'].values[0] if not last_h.empty else 1500,
            last_a['elo_away'].values[0] if not last_a.empty else 1500,
            compute_advanced_stats(df_hist, h, n), compute_advanced_stats(df_hist, a, n),
            calculate_xg(h, df_hist, n, True), calculate_xg(a, df_hist, n, False),
            calculate_rest_days(h, df_hist, n, now), calculate_rest_days(a, df_hist, n, now),
            calculate_h2h(h, a, df_hist),
            calculate_momentum_decay(h, df_hist, n, True), calculate_momentum_decay(a, df_hist, n, False)))
    X_sc = selector.transform(scaler.transform(np.array(X_next, dtype=float)))
    return model.predict_proba(X_sc)


# =====================================================================
# SCHEDINE (identiche alla versione di marzo)
# =====================================================================
def is_top(t): return t in TOP_TEAMS
def is_weak(t): return t in WEAK_ATTACKS


def calc_poisson_v23(h_name, a_name, h_s, h_c, a_s, a_c):
    LEAGUE_FACTOR = 1.05
    lh = ((h_s * 0.6 + a_c * 0.4) + 0.1) * LEAGUE_FACTOR
    la = ((a_s * 0.6 + h_c * 0.4) - 0.1) * LEAGUE_FACTOR
    if is_top(h_name) and is_weak(a_name): la *= 0.50
    if is_top(a_name) and is_weak(h_name): lh *= 0.65
    return max(0.3, lh), max(0.2, la)


def score_slip_quality(slip):
    ev_score = min(slip['ev'] / 2.5, 1.0) * 40
    n_legs = len(slip['matches'])
    legs_score = (5 - n_legs) * 10 if n_legs <= 5 else 0
    return ev_score + legs_score + min(slip['prob'] * 100, 100) / 100 * 20


def validate_bet_combination(bets):
    by_match = defaultdict(set)
    for b in bets:
        by_match[b['match']].add(b['type'])
    conflicts = [{'1', '2'}, {'1', 'X'}, {'2', 'X'}, {'1', 'X2'}, {'2', '1X'}, {'GG', 'NG'}]
    return not any(c.issubset(s) for s in by_match.values() for c in conflicts)


def combined_probability(bets):
    return float(np.prod([b['prob'] * 0.90 for b in bets]))


def combined_ev(bets):
    return combined_probability(bets) * float(np.prod([b['quota'] for b in bets])) - 1.0


# (tier, filtro, gambe, max schedine nel tier, prob. combinata minima)
_TIERS = [
    ('ULTRA_SAFE', lambda o: o['prob'] >= 0.75 and 1.15 <= o['quota'] <= 1.40 and o['ev'] > 1.03, 15, 2, 4, None),
    ('SAFE', lambda o: o['prob'] >= 0.70 and 1.40 <= o['quota'] <= 1.80 and o['ev'] > 1.05, 20, 3, 1, 0.20),
    ('BALANCED', lambda o: o['prob'] >= 0.60 and 1.70 <= o['quota'] <= 2.50 and o['ev'] > 1.07, 20, 3, 1, 0.12),
    ('AGGRESSIVE', lambda o: o['quota'] >= 2.20 and o['ev'] > 1.10, 15, 2, 0, 0.08),
    ('VALUE', lambda o: o['quota'] >= 1.80 and o['ev'] > 1.10, 15, 2, 1, 0.15),
]


def generate_tiered_portfolio(options, rng):
    """Riproduce generate_tiered_portfolio di marzo: (tier, top-N per EV, gambe, soglia su len(slips), prob minima)."""
    portfolio = []
    used = set()
    for tier, filt, top_n, legs, max_idx, min_prob in _TIERS:
        ops = [o for o in options if filt(o) and o['match'] not in used]
        ops.sort(key=lambda x: x['ev'], reverse=True)
        best = ops[:top_n]
        rng.shuffle(best)
        slips = []
        for combo in itertools.combinations(best, legs):
            combo = list(combo)
            matches = [c['match'] for c in combo]
            if any(m in used for m in matches) or len(matches) != len(set(matches)):
                continue
            if not validate_bet_combination(combo):
                continue
            prob = combined_probability(combo)
            if min_prob is not None and prob < min_prob:
                continue
            if len(slips) > max_idx:
                continue
            slip = {'tier': tier, 'legs': combo, 'matches': matches, 'prob': prob,
                    'quota': float(np.prod([c['quota'] for c in combo])), 'ev': combined_ev(combo)}
            slip['quality_score'] = score_slip_quality(slip)
            slips.append(slip)
            used.update(matches)
        portfolio.extend(slips)
    portfolio.sort(key=lambda x: x['quality_score'], reverse=True)
    return portfolio


STAKING = {
    # versione di marzo (quella usata il 22/3): minimo 5€, Kelly pieno sui tier sicuri
    "march": {"fractions": {'ULTRA_SAFE': 1.0, 'SAFE': 1.0, 'BALANCED': 0.10, 'VALUE': 0.12, 'AGGRESSIVE': 0.15},
              "caps": {'ULTRA_SAFE': 0.20, 'SAFE': 0.12, 'BALANCED': 0.04, 'VALUE': 0.05, 'AGGRESSIVE': 0.06},
              "alloc": {'ULTRA_SAFE': 0.50, 'SAFE': 0.30, 'BALANCED': 0.15, 'VALUE': 0.05, 'AGGRESSIVE': 0.00},
              "min_stake": 5.0, "prob_factor": 1.0, "kelly_on_tier_budget": False},
    # versione di febbraio: minimo 1.5€, frazioni prudenti
    "february": {"fractions": {'ULTRA_SAFE': 0.05, 'SAFE': 0.07, 'BALANCED': 0.10, 'VALUE': 0.12, 'AGGRESSIVE': 0.15},
                 "caps": {'ULTRA_SAFE': 0.02, 'SAFE': 0.03, 'BALANCED': 0.04, 'VALUE': 0.05, 'AGGRESSIVE': 0.06},
                 "alloc": {'ULTRA_SAFE': 0.70, 'SAFE': 0.15, 'BALANCED': 0.05, 'VALUE': 0.05, 'AGGRESSIVE': 0.05},
                 "min_stake": 1.5, "prob_factor": 0.95, "kelly_on_tier_budget": True},
}


def kelly_stake(prob, quota, bankroll, tier, cfg):
    if quota <= 1.0:
        return 0.0
    p = prob * 0.80
    b = quota - 1
    f_star = (b * p - (1 - p)) / b
    if f_star <= 0:
        return 0.0
    return min(f_star * cfg["fractions"].get(tier, 0.08) * bankroll, bankroll * cfg["caps"].get(tier, 0.03))


def choose_stakes(portfolio, cfg, budget=BUDGET_TOTALE):
    """Come print_final_strategy_v25: max 2 schedine per tier, puntata minima forzata."""
    tier_budgets = {t: budget * a for t, a in cfg["alloc"].items()}
    tier_used = defaultdict(float)
    count = defaultdict(int)
    picks = []
    for slip in portfolio:
        tier = slip['tier']
        if count[tier] >= 2:
            continue
        bank = tier_budgets[tier] if cfg["kelly_on_tier_budget"] else budget
        stake = max(kelly_stake(slip['prob'] * cfg["prob_factor"], slip['quota'], bank, tier, cfg), cfg["min_stake"])
        if tier_used[tier] + stake > tier_budgets[tier]:
            continue
        picks.append({**slip, 'stake': stake})
        tier_used[tier] += stake
        count[tier] += 1
    return picks

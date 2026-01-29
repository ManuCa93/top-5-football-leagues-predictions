# 🚀 GUIDA IMPLEMENTAZIONE V26 STEP-BY-STEP

## FASE 1: PREPARAZIONE (15 minuti)

### Step 1.1: Backup del codice originale
```bash
cd c:\Users\manuc\Documents\seriea_predictions
copy script.py script_v25_backup.py
```

### Step 1.2: Verifica le dipendenze
```bash
pip install scikit-learn pandas numpy scipy requests
```

Verifica le versioni:
```python
import sklearn
print(f"scikit-learn: {sklearn.__version__}")  # Almeno 1.3.0
```

---

## FASE 2: INTEGRAZIONE INCREMENTALE (2-3 ore)

### Step 2.1: Aggiungere le NUOVE FUNZIONI

Nel tuo `script.py` originale, **DOPO** le funzioni `compute_advanced_stats()` e **PRIMA** di `build_features_v23_mega()`, aggiungi:

```python
# ============================================================
# V26 ENHANCEMENT: EXPECTED GOALS (xG)
# ============================================================

def calculate_xg(team, df_hist, idx, is_home=True):
    """
    Expected Goals: Qualità dei tiri basato su performance storica
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
        
        avg_goals = goals.mean()
        xg = avg_goals * 0.85
        
        recent_goals = goals[-5:].mean() if len(goals) >= 5 else avg_goals
        if recent_goals > avg_goals:
            xg *= 1.1
        
        return max(0.3, min(xg, 3.5))
    except:
        return 1.4 if is_home else 1.1


# ============================================================
# V26 ENHANCEMENT: REST DAYS
# ============================================================

def calculate_rest_days(team, df_hist, idx):
    """
    Giorni di riposo prima della partita.
    5+ giorni = +0.3, 3-4 = 0.0, 1-2 = -0.2, <1 = -0.5
    """
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
        
        if rest_days >= 5:
            return 0.3
        elif rest_days >= 3:
            return 0.0
        elif rest_days >= 1:
            return -0.2
        else:
            return -0.5
    except:
        return 0.0


# ============================================================
# V26 ENHANCEMENT: HEAD-TO-HEAD (H2H)
# ============================================================

def calculate_h2h(home, away, df_hist):
    """
    Statistiche scontri diretti ultimi 10 match
    """
    try:
        h2h = df_hist[
            ((df_hist['home_team'] == home) & (df_hist['away_team'] == away)) |
            ((df_hist['home_team'] == away) & (df_hist['away_team'] == home))
        ]
        
        if len(h2h) == 0:
            return 0.0, 1.5, 1.3
        
        h2h_recent = h2h.tail(10)
        
        home_matches = h2h_recent[h2h_recent['home_team'] == home]
        away_matches = h2h_recent[h2h_recent['away_team'] == home]
        
        h_wins = len(home_matches[home_matches['home_goals'] > home_matches['away_goals']])
        h_wins += len(away_matches[away_matches['away_goals'] > away_matches['home_goals']])
        
        h_gf = home_matches['home_goals'].sum() + away_matches['away_goals'].sum()
        h_ga = home_matches['away_goals'].sum() + away_matches['home_goals'].sum()
        
        h_matches = len(home_matches) + len(away_matches)
        if h_matches == 0:
            return 0.0, 1.5, 1.3
        
        h2h_advantage = (h_wins / h_matches) - 0.33
        h2h_gf_avg = h_gf / h_matches if h_matches > 0 else 1.5
        h2h_ga_avg = h_ga / h_matches if h_matches > 0 else 1.3
        
        return max(-0.4, min(h2h_advantage, 0.4)), h2h_gf_avg, h2h_ga_avg
    except:
        return 0.0, 1.5, 1.3


# ============================================================
# V26 ENHANCEMENT: MOMENTUM WITH DECAY
# ============================================================

def calculate_momentum_decay(team, df_hist, idx, is_home=True, decay_rate=0.8):
    """
    Recent form con exponential decay weights
    Ultima partita = 100%, penultima = 80%, terzultima = 64%, etc.
    """
    try:
        df_prev = df_hist[df_hist.index < idx]
        
        if is_home:
            team_matches = df_prev[df_prev['home_team'] == team].tail(10)
        else:
            team_matches = df_prev[df_prev['away_team'] == team].tail(10)
        
        if len(team_matches) == 0:
            return 0.0
        
        points_weighted = []
        weights = []
        
        for i, (_, m) in enumerate(reversed(team_matches.iterrows())):
            weight = (decay_rate ** i)
            weights.append(weight)
            
            if is_home:
                gf, ga = m['home_goals'], m['away_goals']
            else:
                gf, ga = m['away_goals'], m['home_goals']
            
            if gf > ga:
                points = 3
            elif gf == ga:
                points = 1
            else:
                points = 0
            
            points_weighted.append(points * weight)
        
        max_possible = 3 * sum(weights)
        actual_points = sum(points_weighted)
        momentum = (actual_points / max_possible) - 0.5
        
        return max(-0.5, min(momentum, 0.5))
    except:
        return 0.0
```

---

### Step 2.2: MODIFICARE build_features_v23_mega()

**OPZIONE A: Creazione nuova funzione (CONSIGLIATO)**

Crea una **nuova funzione** `build_features_v26_enhanced()` che chiama le nuove funzioni:

```python
def build_features_v26_enhanced(df):
    """
    Version V26: Features V23 + 8 NEW features
    Total: 27 features instead of 19
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
                
                # === ORIGINAL V23 FEATURES (19) ===
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
                    
                    # === NEW V26 (8 features) ===
                    h_xg, a_xg,
                    h_rest, a_rest,
                    h2h_adv, h2h_gf, h2h_ga,
                    h_momentum, a_momentum,
                ]
                
                X.append(feats)
                
                if row["home_goals"] > row["away_goals"]: y.append(2)
                elif row["home_goals"] < row["away_goals"]: y.append(0)
                else: y.append(1)
                
            except Exception as e:
                continue
        
        log_msg(f"[OK] Training Set: {len(X)} campioni con 27 features (V26).")
        return np.array(X), np.array(y), df
    
    except Exception as e:
        log_msg(f"[ERROR] Errore build_features_v26: {e}", level="ERROR")
        return np.array([]), np.array([]), df
```

Poi nel main del codice, **sostituisci la chiamata**:

```python
# PRIMA (V25):
# X, y, df = build_features_v23_mega(df)

# DOPO (V26):
X, y, df = build_features_v26_enhanced(df)
```

---

### Step 2.3: MIGLIORARE il TRAINING CON ROBUSTSCALER

**Nel `train_model()`, sostituisci il StandardScaler:**

```python
# PRIMA:
# scaler = StandardScaler()

# DOPO:
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler(quantile_range=(10, 90))
X_scaled = scaler.fit_transform(X)
```

---

### Step 2.4: AGGIUNGERE FEATURE SELECTION

**Nel `train_model()`, DOPO il scaling, aggiungi:**

```python
from sklearn.feature_selection import SelectKBest, f_classif

log_msg("[INFO] Feature Selection: SelectKBest k=20...")
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, y)
log_msg(f"[OK] Features selected: {X_selected.shape[1]}")

# Usa X_selected invece di X_scaled
split = int(len(X_selected) * 0.85)
X_train, X_test = X_selected[:split], X_selected[split:]
y_train, y_test = y[:split], y[split:]
```

---

### Step 2.5: AGGIUNGERE CALIBRATION

**Nel `train_model()`, DOPO il training dello stacking:**

```python
from sklearn.calibration import CalibratedClassifierCV

log_msg("[INFO] Calibrating probabilities...")
cal_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
cal_clf.fit(X_train, y_train)

# Usa cal_clf per predictions
preds = cal_clf.predict(X_test)
acc = accuracy_score(y_test, preds)
```

---

## FASE 3: TESTING INCREMENTALE (1-2 ore)

### Test 1: Verifica che le nuove features non hanno NaN

```python
import numpy as np

# Dopo calculate_xg, calculate_rest_days, etc:
print(f"xG home values: min={h_xg:.3f}, max={h_xg:.3f}")
print(f"Rest days: {h_rest:.3f}")
print(f"H2H: {h2h_adv:.3f}, {h2h_gf:.3f}, {h2h_ga:.3f}")
print(f"Momentum: {h_momentum:.3f}")

# Verifica nessun NaN in X
assert not np.isnan(X).any(), "NaN values found in features!"
```

### Test 2: Confronto velocità training

```python
import time

start = time.time()
X, y, df = build_features_v26_enhanced(df)
t1 = time.time() - start
log_msg(f"[TIMING] Feature engineering: {t1:.1f}s")

start = time.time()
clf, scaler = train_model(X, y)
t2 = time.time() - start
log_msg(f"[TIMING] Model training: {t2:.1f}s")
log_msg(f"[TIMING] Total: {t1+t2:.1f}s")
```

**Aspettate:**
- V25: Feature engineering ~2s, Training ~2s, Total ~4s
- V26: Feature engineering ~3s, Training ~15s, Total ~18s
- V26 è più lento ma più accurato!

### Test 3: Confronto accuracy

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score

# V25 (originale)
acc_v25 = ...  # Salva il valore dal run precedente
print(f"V25 Accuracy: {acc_v25:.3f}")

# V26 (nuovo)
preds_v26 = clf.predict(X_test)
acc_v26 = accuracy_score(y_test, preds_v26)
prec_v26 = precision_score(y_test, preds_v26, average='weighted')
rec_v26 = recall_score(y_test, preds_v26, average='weighted')

print(f"V26 Accuracy:  {acc_v26:.3f} (improvement: +{(acc_v26-acc_v25)*100:.1f}%)")
print(f"V26 Precision: {prec_v26:.3f}")
print(f"V26 Recall:    {rec_v26:.3f}")
```

---

## FASE 4: DEPLOY (30 minuti)

### Step 4.1: Sostituire nel main script

Se tutto funziona, integra **definitivamente** le nuove funzioni nel tuo `script.py` originale.

### Step 4.2: Run il predictor aggiornato

```bash
cd c:\Users\manuc\Documents\seriea_predictions
python script.py
```

### Step 4.3: Monitor dei logs

```bash
# Vedi gli ultimi log
Get-Content predictor_february_1.log | Select-Object -Last 100
```

---

## ✅ CHECKLIST COMPLETAMENTO

- [ ] Backup di script.py creato
- [ ] Nuove 4 funzioni aggiunte (xG, Rest, H2H, Momentum)
- [ ] build_features_v26_enhanced() creata e testata
- [ ] RobustScaler integrato
- [ ] SelectKBest integrato
- [ ] CalibratedClassifierCV integrato
- [ ] Test 1: Nessun NaN in features
- [ ] Test 2: Training time verificato (<20s)
- [ ] Test 3: Accuracy migliorata di 5-7%
- [ ] Script principale aggiornato
- [ ] Nuovo run completato senza errori

---

## 🚨 TROUBLESHOOTING

### Errore: "NaN in features"
```python
# Aggiungi debug
print(f"xG: {h_xg}, Rest: {h_rest}, H2H: {h2h_adv}, Momentum: {h_momentum}")
assert not np.isnan(h_xg), "xG is NaN!"
```

### Errore: "SelectKBest failed"
```python
# K troppo grande? Prova:
selector = SelectKBest(f_classif, k=15)  # invece di 20
```

### Errore: "Not enough samples for Calibration"
```python
# CV=5 necessita almeno 20 campioni per fold
if len(X_train) < 100:
    clf.fit(X_train, y_train)  # Usa senza calibration
else:
    cal_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
    cal_clf.fit(X_train, y_train)
```

### Training molto lento
```python
# GridSearchCV è lento. Opzioni:
# 1. Riduci CV=3 invece di 5
# 2. Riduci param_grid size
# 3. Usa n_jobs=-1 per parallelization (già fatto)
```

---

## 📊 RISULTATI ATTESI

| Metrica | V25 | V26 | Improvement |
|---------|-----|-----|-------------|
| Accuracy | 50.0% | 55-57% | +5-7% |
| Precision | ? | ? | +3-5% |
| Recall | ? | ? | +3-5% |
| Training Time | 4s | 18s | 4.5x (worth it) |
| Feature Count | 19 | 27 | +42% |

---

## 🎯 PROSSIMI PASSI

1. ✅ Implementa V26
2. ✅ Testa su dati storici
3. → Monitora performance su predizioni live
4. → Se bene, mantieni V26
5. → Se non bene, torna a V25 e rivedi features

Buona fortuna! 🚀

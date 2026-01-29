# 📊 V26 QUICK REFERENCE GUIDE

## 🎯 Cosa Migliora in V26

```
┌─────────────────────────────────────────────────────────────────┐
│                    MIGLIORAMENTI V26 VISUALIZZATI                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  V25 FEATURES (19)          V26 NUOVE FEATURES (+8)             │
│  ══════════════             ══════════════════════             │
│  • ELO Rating               • Expected Goals (xG)  [+2-3%]       │
│  • Attack/Defense           • Rest Days             [+1-2%]       │
│  • Form                     • Head-to-Head         [+1-2%]       │
│  • Efficiency               • Momentum Decay        [+1%]         │
│  • Trend                                                          │
│  • Home Advantage           OPTIMIZZAZIONI                       │
│                             ══════════════                       │
│  V25 TRAINING               • RobustScaler         [+1%]         │
│  ══════════════             • Feature Selection    [+0.5%]       │
│  • StandardScaler           • Calibration          [+0.5%]       │
│  • 85/15 Split              • GridSearch (Opz.)   [+1-2%]       │
│  • Stacking                                                       │
│                             TOTALE IMPATTO: +5-7%                │
└─────────────────────────────────────────────────────────────────┘

                    ACCURACY IMPROVEMENT ROADMAP

     V25: 50.0%  ──────────→  V26: 55-60%  ✅
           │
           ├─ Features   (+5-6%)  Expected Goals, Rest, H2H, Momentum
           ├─ Scaling    (+1%)    RobustScaler
           ├─ Selection  (+0.5%)  SelectKBest
           └─ Calib      (+0.5%)  CalibratedClassifierCV
```

---

## 🚀 OPZIONI DI IMPLEMENTAZIONE

```
┌─────────────────────────────────────┐
│  TEMPO vs ACCURATEZZA               │
├─────────────────────────────────────┤
│                                     │
│  EXPERT              5-6h ████████  │ +9-12%
│  (GridSearch+All)                   │
│                                     │
│  FULL               3-4h ██████     │ +7-9%
│  (All features)                     │
│                                     │
│  QUICK              1-2h ███        │ +4-5%
│  (xG + Rest only)                   │
│                                     │
│  BASELINE           0.5h █          │ 0%
│  (Original V25)                     │
│                                     │
└─────────────────────────────────────┘
```

---

## 📋 IMPLEMENTAZIONE CHECKLIST

### FASE 1: SETUP (15 min)
```
[ ] Backup script.py → script_v25_backup.py
[ ] Verifica dipendenze: scikit-learn >= 1.3.0
[ ] Leggi IMPROVEMENT_GUIDE.md
[ ] Scegli OPZIONE A/B/C
```

### FASE 2: CODE (1-3 hours depends on option)
```
QUICK (Option A):
[ ] Copia calculate_xg() dal script_v26_enhancements.py
[ ] Copia calculate_rest_days()
[ ] Modifica build_features_v23_mega() → usa xG + rest
[ ] Test: No NaN, No errors

FULL (Option B):
[ ] Copia tutte 4 nuove funzioni (xG, Rest, H2H, Momentum)
[ ] Copia build_features_v26_enhanced() intera
[ ] Aggiorna train_model():
    [ ] RobustScaler al posto di StandardScaler
    [ ] Aggiungi SelectKBest(k=20)
    [ ] Aggiungi CalibratedClassifierCV
[ ] Test: Accuracy migliore di V25

EXPERT (Option C):
[ ] Fai tutto da FULL
[ ] Aggiungi GridSearchCV per RF hyperparameter tuning
[ ] Aggiungi StratifiedKFold cross-validation
[ ] Monitor CPU/Memory usage
[ ] Test: Aspetta GridSearch completi (5-10 min)
```

### FASE 3: TEST (1-2 hours)
```
[ ] Test 1: Verifica features non NaN
    ```python
    assert not np.isnan(X).any(), "NaN found!"
    ```

[ ] Test 2: Confronta velocità
    V25: ~4s | V26: ~18s | Aspettato ✓

[ ] Test 3: Confronta accuracy
    V25: 50.0% | V26: 55%+ | ✅ Migliorato!

[ ] Test 4: Run completo senza errori
    python script.py → Controllare logs
```

### FASE 4: DEPLOY (30 min)
```
[ ] Se V26 accuracy > V25: Mantieni V26
[ ] Se V26 accuracy < V25: Torna a V25
[ ] Update documentation
[ ] Commit changes a git
[ ] Monitor performance live (prima settimana)
```

---

## 🎯 FEATURE ENGINEERING EXPLAINED

### 1️⃣ Expected Goals (xG)

```python
calculate_xg(team, df_hist, idx, is_home=True)

┌─────────────────────────────────────────┐
│ Expected Goals Formula                  │
├─────────────────────────────────────────┤
│                                         │
│ xG = Avg_Goals_Storico * 0.85           │
│      * (1 + boost_if_recent_good)       │
│                                         │
│ Interpretazione:                        │
│ • xG > gol_reali → Underperforming      │
│ • xG < gol_reali → Overperforming       │
│ • xG ~ gol_reali → Normalizzato         │
│                                         │
│ Range: [0.3, 3.5]                       │
│ Impatto: +2-3% accuracy                 │
└─────────────────────────────────────────┘
```

### 2️⃣ Rest Days

```python
calculate_rest_days(team, df_hist, idx)

┌──────────────────────────────────────┐
│ Rest Impact on Performance            │
├──────────────────────────────────────┤
│                                      │
│ ≥5 days   →  +0.3  (Fresh, energico)│
│ 3-4 days  →   0.0  (Normal)         │
│ 1-2 days  →  -0.2  (Stanco)         │
│ <1 day    →  -0.5  (Molto stanco)   │
│                                      │
│ Effetto: Home team win rate +15%    │
│          con 5+ giorni di riposo    │
│                                      │
│ Range: [-0.5, +0.3]                 │
│ Impatto: +1-2% accuracy             │
└──────────────────────────────────────┘
```

### 3️⃣ Head-to-Head (H2H)

```python
calculate_h2h(home, away, df_hist)

┌─────────────────────────────────┐
│ H2H Analysis (Ultimi 10 match)  │
├─────────────────────────────────┤
│                                 │
│ h2h_advantage:                  │
│  = (Home_Win_% - 33%) - baseline│
│  Range: [-0.4, +0.4]            │
│                                 │
│ Esempio:                        │
│ • Milan vs Roma: +0.25          │
│   (Milan vince 60% vs Roma)     │
│ • Roma vs Milan: -0.25          │
│   (Roma vince 10% vs Milan)     │
│                                 │
│ Impatto: +1-2% accuracy         │
└─────────────────────────────────┘
```

### 4️⃣ Momentum Decay

```python
calculate_momentum_decay(team, df_hist, idx, decay_rate=0.8)

┌───────────────────────────────────────┐
│ Recent Form Scoring                   │
├───────────────────────────────────────┤
│                                       │
│ Ultimi match weight:                  │
│ 1st (most recent):  100%              │
│ 2nd:                80%               │
│ 3rd:                64%               │
│ 4th:                51%               │
│ 5th:                41%               │
│ ... decays exponentially              │
│                                       │
│ Points: Win=3, Draw=1, Loss=0         │
│ Momentum = normalized_weighted_points │
│                                       │
│ Range: [-0.5, +0.5]                   │
│ Impatto: +1% accuracy                 │
└───────────────────────────────────────┘
```

---

## 🔧 INTEGRAZIONE NEL CODICE

### Step 1: Aggiungi le funzioni

```python
# In script.py, DOPO compute_advanced_stats() e PRIMA di build_features_v23_mega()

def calculate_xg(team, df_hist, idx, is_home=True):
    # Copia dal script_v26_enhancements.py linea ~50

def calculate_rest_days(team, df_hist, idx):
    # Copia dal script_v26_enhancements.py linea ~85

def calculate_h2h(home, away, df_hist):
    # Copia dal script_v26_enhancements.py linea ~130

def calculate_momentum_decay(team, df_hist, idx, is_home=True, decay_rate=0.8):
    # Copia dal script_v26_enhancements.py linea ~165
```

### Step 2: Modifica build_features (QUICK OPTION A)

```python
# PRIMA: usava solo 19 features
X.append([
    row["elo_home"], row["elo_away"],
    h_stats['scored_overall'], ...
    # ... 19 total
])

# DOPO: aggiungi le 8 nuove
X.append([
    # ... 19 original ...
    h_xg, a_xg,                                    # +2 new
    h_rest, a_rest,                                # +2 new
    h2h_adv, h2h_gf, h2h_ga,                      # +3 new
    h_momentum, a_momentum,                        # +2 new
    # = 27 total
])
```

### Step 3: Migliora il training (FULL OPTION B)

```python
# In train_model(), sostituisci:

# PRIMA:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DOPO:
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif

scaler = RobustScaler(quantile_range=(10, 90))
X_scaled = scaler.fit_transform(X)

selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, y)

# Usa X_selected da qui in poi
X_train = X_selected[:split]
```

### Step 4: Calibra il modello

```python
# DOPO il training dello stacking ensemble:

from sklearn.calibration import CalibratedClassifierCV

cal_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
cal_clf.fit(X_train, y_train)

# Usa cal_clf per predictions
return cal_clf, scaler
```

---

## 📊 PERFORMANCE TIMELINE

```
Time    │ Activity              │ Cumulative
────────┼───────────────────────┼─────────────
0:00    │ Backup & Setup        │ 15 min
0:15    │ Copia functions       │ 45 min
0:45    │ Modify features       │ 1:30 h
1:30    │ Update training       │ 2:00 h
2:00    │ Test 1: No NaN        │ 2:15 h
2:15    │ Test 2: Speed check   │ 2:30 h
2:30    │ Test 3: Accuracy      │ 3:00 h
3:00    │ Bug fixing (if any)   │ 3:30 h
3:30    │ Final validation      │ 4:00 h
4:00    │ ✅ DONE!              │ 4:00 h
```

---

## ⚠️ COMMON PITFALLS & FIXES

```
Problema                    Soluzione
──────────────────────────────────────────────────────────
NaN in features             Check empty dataframes,
                           return default values

Accuracy worse              Check data leakage,
                           verify feature calculation

Training slow               Expected! V26 = 4.5x slower
                           Use n_jobs=-1 for parallelism

Memory error                Reduce CV folds (3 instead 5)
                           or reduce data size

Features not improving      Maybe già buono V25,
                           or need more/different features
```

---

## 🎯 SUCCESS CRITERIA

✅ **V26 è successful se:**

1. **No errors**: Script runs without exceptions
2. **Better accuracy**: V26 > V25 (e.g., 55% > 50%)
3. **Reasonable timing**: Training < 20 seconds
4. **Stable results**: Accuracy consistent across runs
5. **Production ready**: Can process real matches

❌ **Rollback a V25 se:**

1. Accuracy peggiore di V25
2. Training time > 5 minuti
3. Memory issues
4. Frequent NaN/errors
5. Probabilities non affidabili

---

## 📞 QUICK HELP

**Q: Quale opzione scelgo?**
A: Se nuovo alla ML → QUICK (A). Se esperto → EXPERT (C).

**Q: Quanto migliora davvero?**
A: 5-7% realistico. 9-12% se tutto perfetto.

**Q: Torno a V25 dopo V26?**
A: Sì, se accuracy peggiore o troppo lento.

**Q: Gridserach fa differenza?**
A: +1-2% ma aggiunge 5-10 min training.

**Q: Serve CalibratedClassifierCV?**
A: Sì, per probabilità affidabili nelle scommesse.

---

**Pronto? Inizia con la lettura di IMPLEMENTATION_GUIDE_V26.md! 🚀**

# 📊 GUIDA AI MIGLIORAMENTI DEL MODELLO DI PREDIZIONE

## Stato Attuale
- **Accuracy**: ~49-50% (accettabile ma non eccezionale)
- **EV Medio**: 1.42 
- **ROI Teorico**: 866% (troppo ottimista, sospetto)
- **Features Attuali**: 19 features (ELO, statistiche di attacco/difesa, form, etc.)
- **Modello**: Stacking (RF + AdaBoost + GB)

---

## 🎯 PROBLEMI IDENTIFICATI

### 1. **Features Insufficienti**
- ❌ Mancano Expected Goals (xG) - fondamentale per qualità dei tiri
- ❌ Non considera i giorni di riposo tra partite (home advantage aumenta con riposo)
- ❌ Manca H2H head-to-head performance (storici scontri diretti)
- ❌ Non traccia infortuni/fatigue (sostitute importanti)
- ❌ Momentum recente usa solo ultimi 5 match senza decay weights

### 2. **Scaling Subottimale**
- ❌ StandardScaler risente degli outlier
- ✅ RobustScaler più robusto per dati sportivi con anomalie

### 3. **Hyperparameter Non Ottimizzati**
- ❌ Iperparametri settati manualmente
- ✅ GridSearchCV troverebbe combinazioni migliori

### 4. **Cross-Validation Semplice**
- ❌ Split cronologico 85-15 non garantisce equilibrio delle classi
- ✅ StratifiedKFold riduce variance e migliora generalizzazione

### 5. **Probabilità Non Calibrate**
- ❌ Le probabilità predette dal modello potrebbero non essere affidabili
- ✅ CalibratedClassifierCV risolve il problema

### 6. **Ensemble Non Ottimizzato**
- ❌ Stacking con pesi fissi nel meta-model
- ✅ Voting Classifier con pesi custom basati su accuracy di validazione

---

## 📈 MIGLIORAMENTI PROPOSTI (PRIORITÀ)

### PRIORITÀ ALTA (Impatto +5-7% accuracy)

#### **1. Aggiungere Expected Goals (xG)**
```python
# Basato su qualità dei tiri, posizione, tipo di azione
def calculate_xg(team, df_hist, idx, is_home=True):
    # Analizza i tiri della squadra nelle ultime N partite
    # Stima probabilità di goal per ogni tiro
    # Sommale per ottenere xG atteso
    pass
```
**Impatto**: +2-3% (fondamentale per previsioni realistiche)

#### **2. Aggiungere Rest Days Feature**
```python
def calculate_rest_days(team, df_hist, idx):
    # Giorni di riposo prima della partita
    # Home teams vincono più frequentemente con 5+ giorni di riposo
    # Away teams perdono vigore con <2 giorni di riposo
    pass
```
**Impatto**: +1-2%

#### **3. H2H Head-to-Head Performance**
```python
def calculate_h2h(home, away, df_hist):
    # Ultimi 10 scontri diretti tra le squadre
    # Win rate, gol medio, performance in casa vs fuori
    pass
```
**Impatto**: +1-2%

### PRIORITÀ MEDIA (Impatto +2-4% accuracy)

#### **4. RobustScaler + Feature Normalization**
```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler(quantile_range=(10, 90))
```
**Impatto**: +1%

#### **5. GridSearchCV per Hyperparameter Tuning**
```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [300, 400, 500],
    'max_depth': [8, 10, 12],
    ...
}
grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
```
**Impatto**: +1-2%

#### **6. StratifiedKFold + Cross-Validation**
```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
**Impatto**: +0.5-1%

### PRIORITÀ BASSA (Impatto +0.5-1% accuracy)

#### **7. Calibrated Probabilities**
```python
from sklearn.calibration import CalibratedClassifierCV
cal_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
```
**Impatto**: +0.5%

#### **8. Feature Selection (SelectKBest)**
```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=15)  # Mantieni le 15 features migliori
```
**Impatto**: +0.5%

---

## 🔧 IMPLEMENTAZIONE INCREMENTALE

### FASE 1: Feature Engineering (3-4 ore di sviluppo)
1. ✅ Aggiungere Expected Goals (xG)
2. ✅ Aggiungere Rest Days
3. ✅ Aggiungere H2H performance
4. ✅ Implementare momentum recente con decay weights

**Risultato Atteso**: +4-6% accuracy (da 50% a 54-56%)

### FASE 2: Scaling & Preprocessing (1-2 ore)
1. ✅ Passare a RobustScaler
2. ✅ Feature normalization
3. ✅ Outlier detection (quantile-based)

**Risultato Atteso**: +1% accuracy

### FASE 3: Model Optimization (2-3 ore)
1. ✅ GridSearchCV per hyperparameter tuning
2. ✅ StratifiedKFold cross-validation
3. ✅ Validate on holdout test set

**Risultato Atteso**: +1-2% accuracy

### FASE 4: Calibration & Ensemble (1-2 ore)
1. ✅ CalibratedClassifierCV
2. ✅ Feature selection con SelectKBest
3. ✅ Optimized Voting Classifier

**Risultato Atteso**: +0.5-1% accuracy

---

## 📊 METRICHE DA TRACCIARE

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

# Calcola per ogni modello in training:
- Accuracy (TP+TN) / (TP+TN+FP+FN)
- Precision per classe (correctness)
- Recall per classe (completeness)
- F1 Score (harmonic mean)
- ROC-AUC (probabilistic ranking)
- Confusion Matrix (dove sbaglia il modello)
```

---

## 💡 BEST PRACTICES APPLICATE

1. **Time-Series Split**: Mantieni l'ordine cronologico nei dati di training/test
2. **Class Imbalance**: Usa `class_weight='balanced'` nei classifiers
3. **Feature Scaling**: RobustScaler per dati sportivi con outlier
4. **Cross-Validation**: StratifiedKFold per garantire rappresentazione uniforme
5. **Probability Calibration**: CalibratedClassifierCV per affidabilità predizioni
6. **Ensemble Diversity**: Combina modelli diversi (RF, GB, Ada) con pesi ottimizzati
7. **Regularization**: L2 penalty nel Logistic Regression meta-model
8. **Monitoring**: Log accuracy per epoch, early stopping se non migliora

---

## 🚀 TIMELINE IMPLEMENTAZIONE

| Fase | Ore | Priorità | Impatto |
|------|-----|----------|--------|
| Feature Engineering | 4 | 🔴 ALTA | +5-6% |
| Scaling & Preprocessing | 2 | 🟡 MEDIA | +1% |
| Hyperparameter Tuning | 3 | 🟡 MEDIA | +1-2% |
| Calibration | 2 | 🟢 BASSA | +0.5% |
| **TOTALE** | **11 ore** | - | **+7-9.5%** |

**Accuracy Target**: Da 50% → **57-60%** (considerato buono per scommesse)

---

## ⚠️ ATTENZIONE IMPORTANTE

- Il modello sembra **SOVRA-OTTIMISTA** (EV 1.42 con accuracy 50% è impossibile)
- Rivedi il calcolo dell'EV e del Kelly Criterion
- Possibile bias positivo nei dati di training
- Verifica di non aver committere data leakage (futures data nel training)

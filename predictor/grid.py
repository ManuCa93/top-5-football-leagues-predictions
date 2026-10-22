"""Grid search su DEV (2020/21 - 2024/25): quali feature, con quali parametri, migliorano le quote di mercato.

Fasi (ogni configurazione è un walk-forward completo, solo dati passati):
  0. finestra di training e peso delle stagioni vecchie, solo quote
  1. ogni famiglia di feature da sola, con tutti i suoi parametri
  2. selezione in avanti: una famiglia entra solo se migliora il log loss in modo chiaro
  3. regolarizzazione
Il risultato è salvato in data/research/. Le regole di gioco si scelgono dopo (predictor.betgrid).

  python -m predictor.grid       # 1X2
  python -m predictor.grid gg    # Goal/No Goal
"""
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .walkforward import DEV_SEASONS
from .research import (ADV_GG_STATS, ADV_STRENGTH_STATS, DAYS_GRID, ELO_GRID, EWM_GRID, GG_STATS, LASTN_GRID, RESEARCH_DIR, STRENGTH_STATS, SW_GRID,
                       SW_GRID_GG, TARGETS, design, feature_bank, fit_predict, load_base, timer)

N_JOBS = 3
MIN_GAIN = 0.0002  # miglioramento minimo del log loss per tenere una famiglia (circa 1 errore standard)
LEAKY = "infortuni_now"  # assenti al calcio d'inizio: dopo le quote del venerdì, solo per confronto


def families(bank, target="1x2"):
    fam = {}

    def add(f, label, cols):
        if all(c in bank.columns for c in cols):
            fam.setdefault(f, {})[label] = cols

    if target == "gg":  # attacco e difesa delle due squadre, non la differenza
        for stat in GG_STATS + ADV_GG_STATS:
            for m1, m2 in SW_GRID_GG:
                add(f"stagioni_{stat}", f"m1={m1} m2={m2}", [f"sw_{stat}_{m1}_{m2}_h", f"sw_{stat}_{m1}_{m2}_a"])
                add(f"casa_trasf_{stat}", f"m1={m1} m2={m2}", [f"swv_{stat}_{m1}_{m2}_h", f"swv_{stat}_{m1}_{m2}_a"])
            for h in EWM_GRID:
                add(f"decadimento_{stat}", f"mezza vita {h} partite", [f"ewm_{stat}_{h}_h", f"ewm_{stat}_{h}_a"])
            for n in (5, 10):
                add(f"ultime_n_{stat}", f"n={n}", [f"last{n}_{stat}_h", f"last{n}_{stat}_a"])
    for stat in STRENGTH_STATS + ADV_STRENGTH_STATS if target == "1x2" else ():
        for m1, m2 in SW_GRID:
            add(f"stagioni_{stat}", f"m1={m1} m2={m2}", [f"sw_{stat}_{m1}_{m2}_diff"])
            add(f"casa_trasf_{stat}", f"m1={m1} m2={m2}", [f"swv_{stat}_{m1}_{m2}_diff"])
        for h in EWM_GRID:
            add(f"decadimento_{stat}", f"mezza vita {h} partite", [f"ewm_{stat}_{h}_diff"])
        for n in LASTN_GRID:
            add(f"ultime_n_{stat}", f"n={n}", [f"last{n}_{stat}_diff"])
    for d in DAYS_GRID:
        add("ultimi_giorni_pts", f"{d} giorni", [f"days{d}_pts_diff"])
        add("ultimi_giorni_res", f"{d} giorni", [f"days{d}_res_diff"])
        add("partite_ultimi_giorni_campionato", f"{d} giorni", [f"days{d}_n_h", f"days{d}_n_a"])
    add("riposo_campionato", "-", ["rest_lg_h", "rest_lg_a"])
    add("riposo_campionato_europa", "-", ["rest_all_h", "rest_all_a"])
    add("partite_14_giorni_con_europa", "-", ["n14_all_h", "n14_all_a"])
    add("europa_prima_dopo", "-", ["euro_prev_h", "euro_prev_a", "euro_next_h", "euro_next_a"])
    for k, h, c in ELO_GRID:
        add("elo", f"K={k} casa={h} carry={c}", [f"elo_{k}_{h}_{c}_diff"])
    add("riposo_tutte_comp", "-", ["rest_all_h", "rest_all_a"])
    add("carico_7_14_giorni", "-", ["n7_all_h", "n7_all_a", "n14_all_h", "n14_all_a"])
    add("europa_coppe_prima_dopo", "-", ["euro_prev_h", "euro_prev_a", "euro_next_h", "euro_next_a",
                                         "cup_next_h", "cup_next_a"])
    for d in (14, 21, 30):
        add("forma_tutte_comp", f"{d} giorni", [f"form{d}_all_diff"])
    for lb in (5, 10, 20):
        add("infortuni_partita_prima", f"minuti ultime {lb}", [f"inj_prev{lb}_h", f"inj_prev{lb}_a"])
        add("squalificati", f"minuti ultime {lb}", [f"susp{lb}_h", f"susp{lb}_a"])
        add(LEAKY, f"minuti ultime {lb}", [f"inj_now{lb}_h", f"inj_now{lb}_a", f"dbt_now{lb}_h", f"dbt_now{lb}_a"])
    return fam


class Evaluator:
    def __init__(self, base, bank, train_from, halflife_years, refit_months=3, C=1.0, target="1x2", seasons=DEV_SEASONS):
        self.meta, self.X0 = design(base, None, [], train_from, target)
        self.bank = bank.reindex(self.meta.match_id)
        self.kw = {"halflife_years": halflife_years, "refit_months": refit_months, "C": C, "target": target,
                   "seasons": seasons}
        self.classes = TARGETS[target]

    def X(self, cols):
        return np.column_stack([self.X0, self.bank[list(cols)].values]) if cols else self.X0

    def losses(self, pred):
        p = pred[[f"m_{c}" for c in self.classes]].values[np.arange(len(pred)), pred.y.values]
        return pd.Series(-np.log(p), index=pred.match_id.values)

    def run(self, configs, **override):
        kw = {**self.kw, **override}
        preds = Parallel(n_jobs=N_JOBS)(delayed(fit_predict)(self.meta, self.X(cols), **kw) for cols in configs)
        return [self.losses(p) for p in preds]


def compare(loss, ref):
    d = (loss - ref.reindex(loss.index)).dropna()
    return float(d.mean()), float(d.std() / np.sqrt(len(d)))


def market_losses(base, index, target="1x2"):
    b = base.set_index("match_id").loc[index]
    if target == "1x2":
        p = b[["p_1", "p_X", "p_2"]].values[np.arange(len(b)), b.y.values]
    else:
        p = np.where((b.home_goals > 0) & (b.away_goals > 0), b.p_GG, b.p_NG)
    return pd.Series(-np.log(p), index=index)


EUROPE_FAMILIES = ("riposo_campionato_europa", "partite_14_giorni_con_europa", "europa_prima_dopo")


def main(target="1x2", sofa=True, seasons=DEV_SEASONS, tag="", europe=True):
    warnings.filterwarnings("ignore")
    t = timer()
    base = load_base()
    bank = feature_bank(base, sofa=sofa)
    print(f"dati e feature pronti: {bank.shape[1]} colonne ({t()})", flush=True)
    rows = []

    # fase 0: finestra di training e peso delle stagioni vecchie
    first = min(seasons)
    stage0 = [(tf, hl) for tf in sorted({2006, max(2006, first - 8), max(2006, first - 4)}) for hl in (None, 2, 4, 8)]
    best0 = None
    for tf, hl in stage0:
        ev = Evaluator(base, bank, tf, hl, target=target, seasons=seasons)
        loss = ev.run([[]])[0]
        mk = market_losses(base, loss.index, target)
        d, se = compare(loss, mk)
        rows.append({"fase": 0, "famiglia": "solo quote", "parametri": f"training dal {tf}, mezza vita {hl} anni",
                     "logloss": loss.mean(), "delta_vs_mercato": d, "se": se})
        print(f"fase 0  dal {tf} mezza vita {hl}: {loss.mean():.5f} (mercato {mk.mean():.5f}, {d:+.5f} ± {se:.5f})",
              flush=True)
        if best0 is None or loss.mean() < best0[0]:
            best0 = (loss.mean(), tf, hl, loss)
    _, tf, hl, ref = best0
    print(f"-> training dal {tf}, mezza vita {hl} ({t()})", flush=True)

    # fase 1: ogni famiglia da sola
    ev = Evaluator(base, bank, tf, hl, target=target, seasons=seasons)
    fam = families(bank, target)
    if not europe:  # il calendario europeo della stagione in corso non è disponibile il venerdì
        fam = {f: v for f, v in fam.items() if f not in EUROPE_FAMILIES}
    labels = [(f, lab) for f, opts in fam.items() for lab in opts]
    losses = ev.run([fam[f][lab] for f, lab in labels])
    single = {}
    for (f, lab), loss in zip(labels, losses):
        d, se = compare(loss, ref)
        rows.append({"fase": 1, "famiglia": f, "parametri": lab, "logloss": loss.mean(), "delta_vs_quote": d, "se": se})
        if f not in single or d < single[f][1]:
            single[f] = (lab, d, se)
    print(f"fase 1 fatta: {len(labels)} configurazioni ({t()})", flush=True)
    for f, (lab, d, se) in sorted(single.items(), key=lambda kv: kv[1][1]):
        print(f"  {f:38s} {lab:28s} {d:+.5f} ± {se:.5f}", flush=True)

    # fase 2: selezione in avanti (le famiglie "now" sono escluse: usano notizie successive alle quote)
    chosen, current = [], ref
    candidates = [f for f, (_, d, _) in sorted(single.items(), key=lambda kv: kv[1][1]) if d < 0 and f != LEAKY]
    improved = True
    while improved and candidates:
        improved = False
        cols_now = [c for f in chosen for c in fam[f][single[f][0]]]
        trial = ev.run([cols_now + fam[f][single[f][0]] for f in candidates])
        gains = [(compare(l, current)[0], f, l) for f, l in zip(candidates, trial)]
        g, f, l = min(gains, key=lambda x: x[0])
        if g < -MIN_GAIN:
            chosen.append(f)
            candidates.remove(f)
            current = l
            improved = True
            d, se = compare(l, ref)
            rows.append({"fase": 2, "famiglia": " + ".join(chosen), "parametri": single[f][0],
                         "logloss": l.mean(), "delta_vs_quote": d, "se": se})
            print(f"fase 2 + {f} ({single[f][0]}): {g:+.5f}, totale vs quote {d:+.5f} ± {se:.5f} ({t()})", flush=True)
    cols = [c for f in chosen for c in fam[f][single[f][0]]]

    # fase 3: regolarizzazione, mezza vita, aggiornamento mensile
    best3 = None
    for C in (0.01, 0.1, 1.0, 10.0):
        for h in sorted({hl, None, 4}, key=lambda v: -1 if v is None else v):
            loss = ev.run([cols], C=C, halflife_years=h, refit_months=1)[0]
            d, se = compare(loss, market_losses(base, loss.index, target))
            rows.append({"fase": 3, "famiglia": "finale", "parametri": f"C={C} mezza vita {h}",
                         "logloss": loss.mean(), "delta_vs_mercato": d, "se": se})
            print(f"fase 3 C={C} mezza vita {h}: vs mercato {d:+.5f} ± {se:.5f}", flush=True)
            if best3 is None or loss.mean() < best3[0]:
                best3 = (loss.mean(), C, h)
    _, C, h = best3

    config = {"target": target, "train_from": tf, "halflife_years": h, "C": C, "families": {f: single[f][0] for f in chosen},
              "columns": cols}
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    (RESEARCH_DIR / f"best_config_{target}{tag}.json").write_text(json.dumps(config, indent=1, ensure_ascii=False))
    pd.DataFrame(rows).to_csv(RESEARCH_DIR / f"grid_results_{target}{tag}.csv", index=False)
    print(f"\nconfigurazione scelta: {json.dumps(config, ensure_ascii=False)} ({t()})")


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--all" in sys.argv:  # scelta finale: tutte le stagioni prima del holdout
        main(args[0] if args else "1x2", sofa="--no-sofa" not in sys.argv, seasons=range(2014, 2025), tag="_all",
             europe="--no-europe" not in sys.argv)
    elif "--pre16" in sys.argv:  # controllo incrociato con gli xG: Understat parte dal 2014/15
        main(args[0] if args else "1x2", sofa="--no-sofa" not in sys.argv, seasons=range(2016, 2020), tag="_pre16",
             europe="--no-europe" not in sys.argv)
    elif "--pre" in sys.argv:  # controllo della procedura: scelgo sul 2014-2019 e verifico sul DEV
        main(args[0] if args else "1x2", sofa="--no-sofa" not in sys.argv, seasons=range(2014, 2020), tag="_pre")
    else:
        main(args[0] if args else "1x2", sofa="--no-sofa" not in sys.argv)

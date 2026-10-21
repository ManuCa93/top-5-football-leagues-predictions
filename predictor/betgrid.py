"""Grid search delle regole di gioco sulle previsioni walk-forward del DEV.

Per ogni regola: numero di giocate, ROI reale (con errore standard) e valore atteso secondo le quote di
chiusura (CLV). Il CLV misura se la quota presa era migliore di quella finale del mercato: è il modo in cui
chi scommette di mestiere capisce se ha un vantaggio, ed è molto meno rumoroso del P&L.
Le regole si scelgono per CLV con almeno MIN_BETS giocate, il ROI reale serve da controllo.

  python -m predictor.betgrid
"""
import itertools
import json
import warnings

import numpy as np
import pandas as pd

from .odds import outcome_wins
from .research import RESEARCH_DIR, bets_summary, design, feature_bank, fit_predict, load_base, timer, value_bets
from .slips import SLIP_TYPES, build_slips, candidate_legs
from .walkforward import DEV_SEASONS, HOLDOUT_SEASONS

MIN_BETS = 150  # almeno ~1 giocata ogni 2 settimane sui 5 anni di DEV
MARKET_SETS = {"1X2": ("1", "X", "2"), "doppie chance": ("1X", "X2"), "tutti": ("1", "X", "2", "1X", "X2")}
MARKET_SETS_GG = {"GG/NG": ("GG", "NG"), "solo GG": ("GG",), "solo NG": ("NG",)}
THRESHOLDS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15)
MAX_ODDS = (2.5, 4.0, 10.0)


def predictions(base, bank, config, seasons):
    target = config.get("target", "1x2")
    meta, X = design(base, bank, config["columns"], config["train_from"], target)
    return fit_predict(meta, X, C=config["C"], seasons=seasons, halflife_years=config["halflife_years"],
                       target=target)


def market_as_pred(base, seasons, target="1x2"):
    b = base[base.season.isin(seasons) & base.p_1.notna() & base.p_GG.notna()]
    meta = b[["match_id", "date", "season", "league", "y"]]
    if target == "gg":
        return meta.assign(m_GG=b.p_GG.values, m_NG=b.p_NG.values)
    return meta.assign(m_1=b.p_1.values, m_X=b.p_X.values, m_2=b.p_2.values)


def single_grid(pred, base, label):
    rows = []
    sets = MARKET_SETS_GG if "m_GG" in pred else MARKET_SETS
    for (mset, markets), t, mo in itertools.product(sets.items(), THRESHOLDS, MAX_ODDS):
        s = bets_summary(value_bets(pred, base, t, markets, mo))
        rows.append({"modello": label, "tipo": "singole", "mercati": mset, "soglia": t, "quota_max": mo, **s})
    return rows


def slip_backtest(pred, base, min_leg_edge=None, slip_types=SLIP_TYPES, budget=5.0):
    """Schedine settimanali con le probabilità del modello. min_leg_edge: ogni gamba deve avere
    p x quota > 1 + min_leg_edge (None = come oggi, la combinazione che perde meno)."""
    b = base.set_index("match_id").loc[pred.match_id].reset_index()
    for k, v in (("1", pred.m_1.values), ("X", pred.m_X.values), ("2", pred.m_2.values)):
        b[f"p_{k}"] = v
    b["p_1X"], b["p_X2"] = b.p_1 + b.p_X, b.p_2 + b.p_X
    b = b.drop(columns=["p_GG", "p_NG"])  # GG/NG: niente modello, fuori dalle schedine di valore
    rows = []
    b["friday"] = b.date - pd.to_timedelta((b.date.dt.weekday - 4) % 7, unit="D")
    b["friday"] = b.friday.dt.normalize()
    for friday, week in b.groupby("friday"):
        week = week[week.date < friday + pd.Timedelta(days=4)]
        recs = [{**r, "label": r["match_id"]} for r in week.to_dict("records")]
        legs = candidate_legs(recs)
        if min_leg_edge is not None:
            legs = [l for l in legs if l["ev"] > 1 + min_leg_edge]
        res = {r["match_id"]: r for r in recs}
        for s in build_slips(legs, budget, slip_types=slip_types):
            won = all(outcome_wins(l["market"], res[l["match_id"]]["home_goals"], res[l["match_id"]]["away_goals"])
                      for l in s["legs"])
            close = np.prod([res[l["match_id"]][f"c_{l['market']}"] * l["q"] for l in s["legs"]]) - 1
            rows.append({"friday": friday, "type": s["type"], "stake": s["stake"], "odds": s["odds"],
                         "won": won, "pnl": s["stake"] * (s["odds"] - 1) if won else -s["stake"],
                         "ev_model": s["ev"], "ev_close": close})
    return pd.DataFrame(rows)


def slip_summary(sl, label, edge):
    if sl.empty:
        return {"modello": label, "tipo": "schedine", "soglia": edge, "bets": 0}
    per = sl.pnl / sl.stake
    return {"modello": label, "tipo": "schedine", "mercati": "tutti", "soglia": edge, "bets": len(sl),
            "roi": sl.pnl.sum() / sl.stake.sum(), "roi_se": per.std() / np.sqrt(len(sl)),
            "ev_close": float(np.average(sl.ev_close, weights=sl.stake)),
            "ev_model": float(np.average(sl.ev_model, weights=sl.stake))}


def main(target="1x2", holdout=False, sofa=True):
    warnings.filterwarnings("ignore")
    t = timer()
    base = load_base()
    bank = feature_bank(base, sofa=sofa)
    config = json.loads((RESEARCH_DIR / f"best_config_{target}.json").read_text())
    seasons = HOLDOUT_SEASONS if holdout else DEV_SEASONS
    preds = {"modello": predictions(base, bank, config, seasons),
             "solo quote": market_as_pred(base, seasons, target)}
    tag = f"{target}_{'holdout' if holdout else 'dev'}"
    preds["modello"].to_pickle(RESEARCH_DIR / f"pred_{tag}.pkl")
    print(f"previsioni pronte ({t()})", flush=True)
    rows = []
    for label, pred in preds.items():
        rows += single_grid(pred, base, label)
        if target == "1x2":
            for edge in (None, 0.0, 0.03, 0.06):
                rows.append(slip_summary(slip_backtest(pred, base, edge), label, edge))
    res = pd.DataFrame(rows)
    res.to_csv(RESEARCH_DIR / f"betgrid_{tag}.csv", index=False)
    ok = res[res.bets >= MIN_BETS].sort_values("ev_close", ascending=False)
    pd.set_option("display.width", 200)
    print(ok.head(25).round(4).to_string(index=False))
    print(f"\n({t()})")


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "1x2", holdout="--holdout" in sys.argv, sofa="--no-sofa" not in sys.argv)

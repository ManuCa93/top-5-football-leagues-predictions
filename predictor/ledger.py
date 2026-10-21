"""Registro delle schedine (data/ledger.csv) e bankroll (data/bankroll.json).

Nel CSV puoi correggere a mano due colonne:
- giocata: 1 se l'hai giocata (default), 0 se no -> non conta nel bankroll
- quota_presa: la quota totale che ti ha dato Snai, se diversa dalla stimata
"""
import json
from datetime import datetime, timedelta

import pandas as pd

from .config import DATA_DIR
from .odds import outcome_wins

LEDGER = DATA_DIR / "ledger.csv"
BANKROLL = DATA_DIR / "bankroll.json"
POSTPONED_AFTER_DAYS = 3  # gamba senza risultato dopo 3 giorni = rinviata, vale quota 1


def load_ledger(path=LEDGER):
    if not path.exists():
        return pd.DataFrame(columns=["slip_id", "creata", "tipo", "puntata", "quota", "quota_presa", "prob", "ev",
                                     "gambe", "stato", "pnl", "giocata"])
    return pd.read_csv(path, dtype={"slip_id": str})


def save_ledger(ledger, path=LEDGER):
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(path, index=False)


def init_bankroll(initial):
    if not BANKROLL.exists():
        BANKROLL.parent.mkdir(parents=True, exist_ok=True)
        BANKROLL.write_text(json.dumps({"iniziale": initial, "dal": datetime.now().strftime("%Y-%m-%d")}))


def bankroll_initial():
    return json.loads(BANKROLL.read_text())["iniziale"] if BANKROLL.exists() else None


def current_bankroll(ledger):
    initial = bankroll_initial() or 0.0
    done = ledger[(ledger.stato != "aperta") & (ledger.giocata == 1)]
    return initial + done.pnl.sum()


def add_slips(ledger, slips, created):
    rows = []
    for i, s in enumerate(slips, 1):
        legs = [{"league": l["league"], "home": l["home_team"], "away": l["away_team"],
                 "date": l["date"].strftime("%Y-%m-%d %H:%M"), "market": l["market"], "q": round(l["q"], 2),
                 "p": round(l["p"], 4)} for l in s["legs"]]
        rows.append({"slip_id": f"{created:%Y-%m-%d}-{i}", "creata": f"{created:%Y-%m-%d}", "tipo": s["type"],
                     "puntata": s["stake"], "quota": round(s["odds"], 2), "quota_presa": None,
                     "prob": round(s["prob"], 4), "ev": round(s["ev"], 4), "gambe": json.dumps(legs),
                     "stato": "aperta", "pnl": 0.0, "giocata": 1})
    new = pd.DataFrame(rows)
    ledger = ledger[~ledger.slip_id.isin(new.slip_id)]  # rilanciare nella stessa giornata sostituisce le proposte
    return pd.concat([ledger, new], ignore_index=True) if len(ledger) else new


def _leg_result(leg, played, now):
    date = datetime.strptime(leg["date"], "%Y-%m-%d %H:%M")
    m = played[(played.league == leg["league"]) & (played.home_team == leg["home"]) & (played.away_team == leg["away"])
               & (played.date >= date - timedelta(days=2))]
    if len(m):
        r = m.iloc[0]
        return "vinta" if outcome_wins(leg["market"], r.home_goals, r.away_goals) else "persa"
    if now - date > timedelta(days=POSTPONED_AFTER_DAYS):
        return "rinviata"
    return None


def settle(ledger, played, now=None):
    now = now or datetime.now()
    ledger = ledger.copy()
    for i, row in ledger[ledger.stato == "aperta"].iterrows():
        legs = json.loads(row.gambe)
        results = [_leg_result(l, played, now) for l in legs]
        if "persa" in results:
            ledger.loc[i, ["stato", "pnl"]] = ["persa", -row.puntata]
        elif None not in results:
            quota = row.quota_presa if pd.notna(row.quota_presa) else row.quota
            for l, res in zip(legs, results):
                if res == "rinviata":
                    quota /= l["q"]
            ledger.loc[i, ["stato", "pnl"]] = ["vinta", round(row.puntata * (quota - 1), 2)]
    return ledger

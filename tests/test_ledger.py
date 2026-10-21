"""Il registro chiude le schedine con i risultati veri. Esegui: python tests/test_ledger.py"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from predictor import ledger as L  # noqa: E402
from predictor.data import load_matches  # noqa: E402
from predictor.odds import add_market, outcome_wins  # noqa: E402
from predictor.slips import build_slips, candidate_legs  # noqa: E402

FRIDAY = datetime(2026, 10, 16, 12)


def test_settle_matches_real_results():
    m = load_matches(refresh=False)
    week = add_market(m[(m.date >= FRIDAY) & (m.date < datetime(2026, 10, 20, 12))])
    recs = [{**r, "label": r["match_id"]} for r in week.to_dict("records")]
    slips = build_slips(candidate_legs(recs), 5.0)
    ledger = L.add_slips(L.load_ledger(Path("_inesistente.csv")), slips, FRIDAY)
    assert (ledger.stato == "aperta").all()

    # prima delle partite resta tutto aperto
    early = L.settle(ledger, m[m.date < FRIDAY], now=FRIDAY)
    assert (early.stato == "aperta").all()

    settled = L.settle(ledger, m, now=datetime(2026, 10, 21))
    assert (settled.stato != "aperta").all()
    for s, row in zip(slips, settled.itertuples()):
        res = {r.match_id: (r.home_goals, r.away_goals) for r in week.itertuples()}
        won = all(outcome_wins(l["market"], *res[l["match_id"]]) for l in s["legs"])
        assert row.stato == ("vinta" if won else "persa")
        expected = round(s["stake"] * (round(s["odds"], 2) - 1), 2) if won else -s["stake"]
        assert abs(row.pnl - expected) < 1e-6
        assert len(json.loads(row.gambe)) == len(s["legs"])


def test_postponed_leg_counts_as_one():
    m = load_matches(refresh=False)
    legs = [{"league": "SA", "home": "Squadra Finta", "away": "Altra Finta", "date": "2026-10-17 15:00",
             "market": "1", "q": 2.0, "p": 0.5}]
    ledger = L.load_ledger(Path("_inesistente.csv"))
    ledger.loc[0] = {"slip_id": "x", "creata": "2026-10-16", "tipo": "t", "puntata": 2.0, "quota": 2.0,
                     "quota_presa": None, "prob": 0.5, "ev": 0.0, "gambe": json.dumps(legs), "stato": "aperta",
                     "pnl": 0.0, "giocata": 1}
    assert L.settle(ledger, m, now=datetime(2026, 10, 18)).stato.iloc[0] == "aperta"
    done = L.settle(ledger, m, now=datetime(2026, 10, 25))
    assert done.stato.iloc[0] == "vinta" and abs(done.pnl.iloc[0]) < 1e-9  # rimborso: quota 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)

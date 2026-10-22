"""Comando settimanale: aggiorna i dati, chiude le schedine passate, propone le nuove e apre il report.

  python run.py                     # uso normale (meglio il venerdì pomeriggio)
  python run.py --bankroll 100      # solo al primo avvio: bankroll di partenza
  python run.py --as-of 2026-10-16  # prova come se fosse quel venerdì (non tocca il registro)
  python run.py --due-doppie        # 2 doppie a giornata invece di doppia + tripla + quota alta
  python run.py --maggiorata Inter 1 2.60 --bankroll 100   # conviene la quota maggiorata Snai?
"""
import argparse
import sys
import webbrowser
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from predictor import ledger as L
from predictor.boost import check
from predictor.config import REPORTS_DIR
from predictor.context import fixture_context
from predictor.data import load_fixtures, load_matches
from predictor.odds import add_market
from predictor.report import render
from predictor.slips import DUE_DOPPIE, SLIP_TYPES, build_slips, candidate_legs

DAYS_AHEAD = 7
MIN_WEEKLY_BUDGET = 3.0


def upcoming(matches, now, demo):
    if demo:  # partite di quel weekend, fingendo di non conoscerne il risultato
        fx = matches[(matches.date >= now) & (matches.date < now + timedelta(days=4))].copy()
        fx[["home_goals", "away_goals"]] = np.nan
        return fx
    fx = load_fixtures()
    if fx.empty:
        return fx
    fx = fx[(fx.date >= now - timedelta(hours=2)) & (fx.date < now + timedelta(days=DAYS_AHEAD))].copy()
    fx["match_id"] = fx.league + "_" + fx.date.dt.strftime("%Y%m%d") + "_" + fx.home_team + "_" + fx.away_team
    return fx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bankroll", type=float, help="bankroll iniziale (primo avvio)")
    ap.add_argument("--budget-pct", type=float, default=5.0, help="%% del bankroll da giocare a settimana (default 5)")
    ap.add_argument("--as-of", help="data AAAA-MM-GG: prova su un venerdì passato")
    ap.add_argument("--no-open", action="store_true", help="non aprire il report nel browser")
    ap.add_argument("--due-doppie", action="store_true", help="2 doppie (quota 1.6-2.6) invece dei 3 tipi")
    ap.add_argument("--maggiorata", nargs=3, metavar=("SQUADRA", "ESITO", "QUOTA"),
                    help="conviene questa quota maggiorata? es. --maggiorata Inter 1 2.60")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")  # console Windows non UTF-8: niente crash sui simboli

    if args.maggiorata:
        team, market, odds = args.maggiorata
        print(check(team, market, float(odds.replace(",", ".")), args.bankroll))
        return

    demo = args.as_of is not None
    now = datetime.fromisoformat(args.as_of).replace(hour=12) if demo else datetime.now()
    print("Aggiorno i dati...", flush=True)
    matches = load_matches(refresh=not demo)
    played = matches[matches.date < now]

    if demo:
        ledger = L.load_ledger(path=L.DATA_DIR / "_nessun_registro.csv")
        initial = args.bankroll or 100.0
    else:
        L.init_bankroll(args.bankroll or 100.0)
        initial = L.bankroll_initial()
        ledger = L.settle(L.load_ledger(), played, now)
    bankroll = initial + ledger[(ledger.stato != "aperta") & (ledger.giocata == 1)].pnl.sum()
    weekly_budget = max(MIN_WEEKLY_BUDGET, round(bankroll * args.budget_pct / 100 * 2) / 2)

    fx = upcoming(matches, now, demo)
    if len(fx):
        fx = add_market(fx).dropna(subset=["p_1"])
    if len(fx):
        fx = fx.merge(fixture_context(played, fx, now), on="match_id")
        recs = [{**r, "label": f"{r['home_team']} - {r['away_team']}"} for r in fx.to_dict("records")]
        slips = build_slips(candidate_legs(recs), weekly_budget,
                            slip_types=DUE_DOPPIE if args.due_doppie else SLIP_TYPES)
    else:
        slips = []

    if not demo:
        ledger = L.add_slips(ledger, slips, now) if slips else ledger
        L.save_ledger(ledger)

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"{'prova-' if demo else ''}{now:%Y-%m-%d}.html"
    out.write_text(render(now, fx if len(fx) else pd.DataFrame(), slips, ledger, bankroll, initial, weekly_budget, demo),
                   encoding="utf-8")

    print(f"\nBankroll {bankroll:.2f} € · budget settimana {weekly_budget:.2f} € · {len(fx)} partite con quote")
    for s in slips:
        print(f"\n{s['type']}: quota {s['odds']:.2f}, prob. {s['prob']:.0%}, puntata {s['stake']:.2f} €, "
              f"costo atteso {s['stake'] * s['ev']:+.2f} €")
        for l in s["legs"]:
            print(f"   {l['home_team']} - {l['away_team']}  [{l['market']}] @{l['q']:.2f}  (prob. {l['p']:.0%})")
    print(f"\nReport: {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()

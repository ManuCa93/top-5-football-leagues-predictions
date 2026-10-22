"""Conviene la quota maggiorata? Confronta la quota Snai con la quota equa (Pinnacle o Betfair Exchange senza
margine, altrimenti la media di mercato) della prossima partita della squadra, dai fixtures di football-data.co.uk.

Uso: python run.py --maggiorata Inter 1 2.60     (mercati: 1 X 2 1X X2 12 GG NG O U; O/U = 2.5)
"""
import numpy as np
import pandas as pd
import requests

from .data import FIXTURES_URL, _read_csv_bytes
from .odds import fair_probs, implied_goal_rates, market_summary, score_matrix

MARKETS = ("1", "X", "2", "1X", "X2", "12", "GG", "NG", "O", "U")
KELLY_FRACTION = 0.25


def _first(row, *cols):
    vals = [row.get(c) for c in cols]
    return vals if all(pd.notna(v) and v > 1 for v in vals) else None


def fair_market_probs(row):
    """Probabilità eque di tutti i mercati per una riga dei fixtures, e la fonte usata."""
    for src, cols in (("Pinnacle", ("PSH", "PSD", "PSA")), ("Betfair Exchange", ("BFEH", "BFED", "BFEA")),
                      ("media mercato", ("AvgH", "AvgD", "AvgA"))):
        o1x2 = _first(row, *cols)
        if o1x2:
            break
    else:
        return None, None
    ou = _first(row, "P>2.5", "P<2.5") or _first(row, "BFE>2.5", "BFE<2.5") or _first(row, "Avg>2.5", "Avg<2.5")
    p1, px, p2 = fair_probs(*o1x2)
    p = {"1": p1, "X": px, "2": p2, "1X": p1 + px, "X2": px + p2, "12": p1 + p2}
    if ou:
        p["O"], p["U"] = fair_probs(*ou)
    lh, la = implied_goal_rates(*o1x2, *(ou or (np.nan, np.nan)))
    p["GG"] = market_summary(score_matrix(lh, la))["GG"]
    p["NG"] = 1 - p["GG"]
    return p, src


def check(team, market, odds, bankroll=None):
    market = market.upper()
    if market not in MARKETS:
        return f"Mercato '{market}' non valido: usa uno di {' '.join(MARKETS)}"
    raw = _read_csv_bytes(requests.get(FIXTURES_URL, timeout=30).content)
    t = team.lower()
    hits = raw[raw.HomeTeam.str.lower().str.contains(t, regex=False) |
               raw.AwayTeam.str.lower().str.contains(t, regex=False)]
    if hits.empty:
        return f"Nessuna partita in programma per '{team}' nei fixtures di football-data.co.uk (escono ~venerdì e ~martedì)."
    row = hits.iloc[0].to_dict()
    p_all, src = fair_market_probs(row)
    if p_all is None or market not in p_all:
        return f"{row['HomeTeam']} - {row['AwayTeam']}: quote non ancora disponibili per {market}."
    p = p_all[market]
    ev = p * odds - 1
    lines = [f"{row['HomeTeam']} - {row['AwayTeam']} ({row['Date']}), esito {market}",
             f"  probabilità equa {p:.1%} ({src}), quota equa {1 / p:.2f}, quota Snai {odds:.2f}",
             f"  valore atteso: {ev:+.1%} per euro giocato"]
    if ev > 0:
        kelly = KELLY_FRACTION * ev / (odds - 1)
        stake = f" = {kelly * bankroll:.2f} €" if bankroll else ""
        lines.append(f"  CONVIENE. Puntata suggerita (1/4 Kelly): {kelly:.1%} del bankroll{stake}")
    else:
        lines.append("  NON conviene: la quota è sotto quella equa.")
    return "\n".join(lines)

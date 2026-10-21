"""Quote "stile Snai" ricostruite dalle quote di mercato di football-data.co.uk.

Tarature misurate su 185 partite 2025/26 con quote Snai vere (tools/extract_snai_odds.py):
- 1X2 Snai ~ media di mercato (Avg), rapporto mediano 1.00
- doppia chance Snai ~ 0.99 x quota ricavata dall'1X2
- GG/NG: P(GG) dai gol attesi impliciti nel mercato (1X2 + O/U 2.5, Dixon-Coles rho=-0.08),
  +0.0065 di correzione, margine Snai 8%. Errore medio sulla quota ~2.5%.
La correzione e il margine sono tarati su quote, non su esiti: nessuna informazione sui risultati.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import poisson

SNAI_DC_FACTOR = 0.99
SNAI_GGNG_MARGIN = 0.08
GG_PROB_SHIFT = 0.0065
DC_RHO = -0.08

_GOALS = np.arange(11)
_TOTAL = np.add.outer(_GOALS, _GOALS)


def score_matrix(lh, la, rho=DC_RHO):
    m = np.outer(poisson.pmf(_GOALS, lh), poisson.pmf(_GOALS, la))
    m[0, 0] *= 1 - lh * la * rho
    m[0, 1] *= 1 + lh * rho
    m[1, 0] *= 1 + la * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def market_summary(m):
    return {
        "1": np.tril(m, -1).sum(), "X": np.trace(m), "2": np.triu(m, 1).sum(),
        "O25": m[_TOTAL > 2.5].sum(), "GG": m[1:, 1:].sum(),
    }


def normalized(*odds):
    p = 1 / np.asarray(odds, dtype=float)
    return p / p.sum()


def fair_probs(*odds):
    """Probabilità senza margine col metodo "power": p_i = (1/q_i)^k con somma 1.
    Sul DEV è meglio calibrato del proporzionale (il margine pesa di più sulle quote alte)."""
    q = 1 / np.asarray(odds, dtype=float)
    k = brentq(lambda k: (q ** k).sum() - 1, 0.3, 3.0)
    return q ** k


def implied_goal_rates(o1, ox, o2, o_over, o_under):
    """Gol attesi (casa, trasferta) coerenti con le quote 1X2 e Over/Under 2.5."""
    p1, px, p2 = normalized(o1, ox, o2)
    target = [p1, px, p2]
    if np.isfinite(o_over) and np.isfinite(o_under):
        target.append(normalized(o_over, o_under)[0])

    def loss(x):
        s = market_summary(score_matrix(*np.exp(x)))
        pred = [s["1"], s["X"], s["2"], s["O25"]][: len(target)]
        return sum((a - b) ** 2 for a, b in zip(pred, target))

    res = minimize(loss, np.log([1.5, 1.15]), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-9})
    return tuple(np.exp(res.x))


def market_row(o1, ox, o2, o_over=np.nan, o_under=np.nan):
    """Per una partita: probabilità "vere" p_<mercato> (senza margine) e quote Snai stimate q_<mercato>."""
    p1, px, p2 = fair_probs(o1, ox, o2)
    lh, la = implied_goal_rates(o1, ox, o2, o_over, o_under)
    p_gg = market_summary(score_matrix(lh, la))["GG"]
    p = {"1": p1, "X": px, "2": p2, "1X": p1 + px, "X2": p2 + px, "GG": p_gg, "NG": 1 - p_gg}
    snai_gg = min(max(p_gg + GG_PROB_SHIFT, 0.01), 0.99)
    q = {"1": o1, "X": ox, "2": o2,
         "1X": SNAI_DC_FACTOR / (1 / o1 + 1 / ox), "X2": SNAI_DC_FACTOR / (1 / o2 + 1 / ox),
         "GG": 1 / (snai_gg * (1 + SNAI_GGNG_MARGIN)), "NG": 1 / ((1 - snai_gg) * (1 + SNAI_GGNG_MARGIN))}
    out = {f"p_{k}": v for k, v in p.items()} | {f"q_{k}": v for k, v in q.items()}
    out["lambda_home"], out["lambda_away"] = lh, la
    return out


def add_market(df):
    """Aggiunge a ogni partita con quote le colonne p_* e q_* di market_row."""
    rows = []
    for r in df.itertuples():
        ok = all(np.isfinite(v) for v in (r.odds_1, r.odds_x, r.odds_2))
        rows.append(market_row(r.odds_1, r.odds_x, r.odds_2, r.odds_over25, r.odds_under25) if ok else {})
    extra = pd.DataFrame(rows, index=df.index)
    return df.join(extra)


def outcome_wins(market, home_goals, away_goals):
    return {
        "1": home_goals > away_goals, "X": home_goals == away_goals, "2": home_goals < away_goals,
        "1X": home_goals >= away_goals, "X2": home_goals <= away_goals,
        "GG": home_goals > 0 and away_goals > 0, "NG": home_goals == 0 or away_goals == 0,
    }[market]

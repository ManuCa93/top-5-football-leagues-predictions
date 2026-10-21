"""Modello dei gol: Poisson con forza d'attacco/difesa per squadra, home advantage e
decadimento esponenziale nel tempo, più correzione Dixon-Coles sui punteggi bassi.

Si addestra SOLO su partite con data < cutoff: chi lo chiama passa già i dati tagliati.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from .odds import market_summary, score_matrix

MARKETS = ("1", "X", "2", "1X", "X2", "O25", "U25", "GG", "NG")


class GoalModel:
    def __init__(self, half_life_days=240, alpha=1e-3, target="goals", max_days=3 * 365):
        self.half_life_days = half_life_days
        self.alpha = alpha
        self.target = target  # "goals" oppure "sot" (tiri in porta)
        self.max_days = max_days
        self.params = {}

    def fit(self, past, cutoff):
        """past: partite giocate prima di cutoff (tutte le leghe)."""
        assert (past.date < cutoff).all(), "dati futuri nel training"
        past = past[past.date >= cutoff - pd.Timedelta(days=self.max_days)]
        for league, g in past.groupby("league"):
            self.params[league] = self._fit_league(g, cutoff)
        return self

    def _fit_league(self, g, cutoff):
        hcol, acol = ("home_goals", "away_goals") if self.target == "goals" else ("home_sot", "away_sot")
        g = g.dropna(subset=[hcol, acol])
        teams = sorted(set(g.home_team) | set(g.away_team))
        ix = {t: i for i, t in enumerate(teams)}
        n, k = len(g), len(teams)
        # due righe per partita: gol casa (attacco casa vs difesa trasferta) e gol trasferta
        X = np.zeros((2 * n, 2 * k + 1))
        rows = np.arange(n)
        h = g.home_team.map(ix).values
        a = g.away_team.map(ix).values
        X[rows, h] = 1
        X[rows, k + a] = 1
        X[rows, 2 * k] = 1  # fattore campo
        X[n + rows, a] = 1
        X[n + rows, k + h] = 1
        y = np.concatenate([g[hcol].values, g[acol].values]).astype(float)
        age = (cutoff - g.date).dt.days.values
        w = np.tile(0.5 ** (age / self.half_life_days), 2)
        reg = PoissonRegressor(alpha=self.alpha, max_iter=1000).fit(X, y, sample_weight=w)
        att = dict(zip(teams, reg.coef_[:k]))
        dfn = dict(zip(teams, reg.coef_[k:2 * k]))
        # squadre nuove (neopromosse): media delle 3 squadre peggiori per attacco/difesa
        new_att = np.mean(sorted(att.values())[:3])
        new_def = np.mean(sorted(dfn.values())[-3:])
        # con i tiri in porta: gol attesi = tiri in porta attesi x conversione media del campionato
        goals = (g.home_goals + g.away_goals).values
        conv = 1.0 if self.target == "goals" else np.sum(w[:n] * goals) / np.sum(w[:n] * (g[hcol] + g[acol]).values)
        return {"att": att, "def": dfn, "home": reg.coef_[2 * k], "icpt": reg.intercept_,
                "new_att": new_att, "new_def": new_def, "conv": conv}

    def rates(self, league, home, away):
        p = self.params[league]
        ah, aa = p["att"].get(home, p["new_att"]), p["att"].get(away, p["new_att"])
        dh, da = p["def"].get(home, p["new_def"]), p["def"].get(away, p["new_def"])
        return (p["conv"] * np.exp(p["icpt"] + ah + da + p["home"]),
                p["conv"] * np.exp(p["icpt"] + aa + dh))


def market_probs_from_rates(lh, la):
    s = market_summary(score_matrix(lh, la))
    return {"1": s["1"], "X": s["X"], "2": s["2"], "1X": s["1"] + s["X"], "X2": s["2"] + s["X"],
            "O25": s["O25"], "U25": 1 - s["O25"], "GG": s["GG"], "NG": 1 - s["GG"]}


def market_implied(row):
    """Probabilità implicite (senza margine) nelle quote medie pre-partita."""
    p = 1 / np.array([row.odds_1, row.odds_x, row.odds_2])
    p = p / p.sum()
    out = {"1": p[0], "X": p[1], "2": p[2], "1X": p[0] + p[1], "X2": p[2] + p[1]}
    if np.isfinite(row.odds_over25) and np.isfinite(row.odds_under25):
        q = 1 / np.array([row.odds_over25, row.odds_under25])
        out["O25"], out["U25"] = q / q.sum()
    return out

"""Strategie trovate online e nella letteratura, provate sulle 105.000 partite di predictor.scan (20 campionati,
2012-oggi) alla quota media di mercato (proxy Snai). Le strategie senza parametri sono giudicate su tutto il periodo
e sul 2020-oggi; quelle con parametri (GAP) li scelgono sul 2014-19 e si verificano sul 2020-oggi.

Esegui: python -m predictor.online
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .scan import SCAN_DIR, SPLIT, fair, load_all

FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]


def load():
    path = SCAN_DIR / "online_matches.pkl"
    if path.exists():
        return pd.read_pickle(path)
    df = load_all()
    df = df.dropna(subset=["avg_1", "avg_X", "avg_2"]).reset_index(drop=True)
    p = fair(df[["avg_1", "avg_X", "avg_2"]])
    df["p1"], df["pX"], df["p2"] = p.T
    pc = fair(df[["pinc_1", "pinc_X", "pinc_2"]])
    df["c1"], df["cX"], df["c2"] = pc.T
    po, pco = fair(df[["avg_O", "avg_U"]]), fair(df[["pinc_O", "pinc_U"]])
    df["pO"], df["cO"], df["cU"] = po[:, 0], pco[:, 0], pco[:, 1]
    df["res"] = np.select([df.home_goals > df.away_goals, df.home_goals == df.away_goals], ["1", "X"], "2")
    df = df.join(standings(df)).join(previous_results(df))
    df.to_pickle(path)
    return df


def standings(df):
    """Prima di ogni partita: punti, posizione, partite giocate e squadre del campionato (solo risultati passati)."""
    out = []
    for (div, season), g in df.groupby(["div", "season"]):
        teams = sorted(set(g.home_team) | set(g.away_team))
        pts, played = dict.fromkeys(teams, 0), dict.fromkeys(teams, 0)
        rows = []
        for day, dg in g.groupby(g.date.dt.normalize()):
            table = sorted(teams, key=lambda t: -pts[t])
            pos = {t: i + 1 for i, t in enumerate(table)}
            ranked = [pts[t] for t in table]
            for r in dg.itertuples():
                rows.append({"idx": r.Index, "n_teams": len(teams), "h_pts": pts[r.home_team],
                             "a_pts": pts[r.away_team], "h_pos": pos[r.home_team], "a_pos": pos[r.away_team],
                             "h_played": played[r.home_team], "a_played": played[r.away_team],
                             "safe_pts": ranked[len(teams) - 4], "fourth_pts": ranked[3], "top_pts": ranked[0]})
            for r in dg.itertuples():
                hp = 3 if r.home_goals > r.away_goals else 1 if r.home_goals == r.away_goals else 0
                ap = 3 if r.away_goals > r.home_goals else 1 if r.home_goals == r.away_goals else 0
                pts[r.home_team] += hp
                pts[r.away_team] += ap
                played[r.home_team] += 1
                played[r.away_team] += 1
        out.append(pd.DataFrame(rows))
    return pd.concat(out).set_index("idx")


def previous_results(df):
    """Differenza reti dell'ultima partita, serie di vittorie e se la squadra era nel campionato la stagione prima."""
    long = pd.concat([
        df[["date", "div", "season", "home_team", "home_goals", "away_goals"]].set_axis(
            ["date", "div", "season", "team", "gf", "ga"], axis=1).assign(side="h", idx=df.index),
        df[["date", "div", "season", "away_team", "away_goals", "home_goals"]].set_axis(
            ["date", "div", "season", "team", "gf", "ga"], axis=1).assign(side="a", idx=df.index),
    ]).sort_values(["team", "date"])
    long["gd"] = long.gf - long.ga
    g = long.groupby("team")
    long["last_gd"] = g.gd.shift(1)
    win = (long.gd > 0).astype(int)
    streak = win.groupby((win != win.shift()).cumsum()).cumsum() * win  # vittorie di fila fino a questa inclusa
    long["win_streak"] = streak.groupby(long.team).shift(1)
    in_div = set(zip(long["div"], long.season, long.team))
    long["promoted"] = [(d, s - 1, t) not in in_div for d, s, t in zip(long["div"], long.season, long.team)]
    first = long.groupby("div").season.transform("min")  # la prima stagione del dataset non ha storia
    long.loc[long.season == first, "promoted"] = False
    cols = ["last_gd", "win_streak", "promoted"]
    h = long[long.side == "h"].set_index("idx")[cols].add_prefix("h_")
    a = long[long.side == "a"].set_index("idx")[cols].add_prefix("a_")
    return h.join(a)


def bets(df, mask, outcome, stake=1.0):
    """Giocate 1X2 alla quota media: outcome è '1', 'X', '2' o un array per riga."""
    d = df[mask]
    o = pd.Series(outcome, index=df.index)[mask] if not isinstance(outcome, str) else pd.Series(outcome, index=d.index)
    odds = np.choose(o.map({"1": 0, "X": 1, "2": 2}).values, [d.avg_1.values, d.avg_X.values, d.avg_2.values])
    close = np.choose(o.map({"1": 0, "X": 1, "2": 2}).values, [d.c1.values, d.cX.values, d.c2.values])
    return pd.DataFrame({"season": d.season.values, "odds": odds, "won": (d.res.values == o.values).astype(float),
                         "p_close": close, "stake": stake if np.isscalar(stake) else np.asarray(stake)[mask.values]})


def summary(b, name, source):
    pnl = b.stake * np.where(b.won == 1, b.odds - 1, -1.0)
    roi = pnl.sum() / b.stake.sum()
    late = b.season >= SPLIT
    roi_late = pnl[late].sum() / b.stake[late].sum() if late.any() else np.nan
    by = pd.Series(pnl).groupby(b.season.values).sum()
    flat = np.where(b.won == 1, b.odds - 1, -1.0)
    clv = (b.odds * b.p_close - 1).dropna()
    return {"strategia": name, "fonte": source, "giocate": len(b), "ROI": roi,
            "ROI_se": flat.std() / np.sqrt(len(b)), "ROI 2020+": roi_late, "CLV": clv.mean() if len(clv) else np.nan,
            "stagioni+": f"{(by > 0).sum()}/{len(by)}", "puntata max": b.stake.max()}


def fibonacci_draws(df, teams_per_league=3):
    """Pareggio della stessa squadra a ogni partita, puntata Fibonacci che riparte dopo un pareggio (Rigoni).
    Squadre: le 3 con più pareggi la stagione prima nello stesso campionato."""
    long = pd.concat([df.assign(team=df.home_team), df.assign(team=df.away_team)])
    draws = long.assign(d=long.res == "X").groupby(["div", "season", "team"]).d.mean()
    out = []
    for (div, season), g in df.groupby(["div", "season"]):
        if (div, season - 1) not in draws.index.droplevel(2):
            continue
        prev = draws.loc[(div, season - 1)]
        teams = [t for t in prev.sort_values(ascending=False).index if t in set(g.home_team) | set(g.away_team)]
        for team in teams[:teams_per_league]:
            k = 0
            for idx in g.index[(g.home_team == team) | (g.away_team == team)]:
                r = df.loc[idx]
                out.append({"season": season, "odds": r.avg_X, "won": float(r.res == "X"), "p_close": r.cX,
                            "stake": FIB[min(k, len(FIB) - 1)]})
                k = 0 if r.res == "X" else k + 1
    return pd.DataFrame(out)


def gap_ratings(df, stat=("shots", "corners"), lam=0.1, phi=0.5):
    """GAP ratings (Wheatcroft 2020) sul totale tiri + angoli: previsione di quanto 'attacca' la partita."""
    x_h = sum(df[f"home_{s}"] for s in stat)
    x_a = sum(df[f"away_{s}"] for s in stat)
    R = {}
    pred = np.full(len(df), np.nan)
    for i, (div, ht, at, sh, sa) in enumerate(zip(df["div"], df.home_team, df.away_team, x_h, x_a)):
        h, a = R.setdefault((div, ht), [np.nan] * 4), R.setdefault((div, at), [np.nan] * 4)  # HA HD AA AD
        ph, pa = (h[0] + a[3]) / 2, (a[2] + h[1]) / 2
        pred[i] = ph + pa
        if np.isnan(sh) or np.isnan(sa):
            continue
        for rating, idx, obs in ((h, 0, sh), (h, 1, sa), (a, 2, sa), (a, 3, sh)):
            if np.isnan(rating[idx]):
                rating[idx] = obs
        ph, pa = (h[0] + a[3]) / 2, (a[2] + h[1]) / 2
        eh, ea = sh - ph, sa - pa
        h[0] += lam * eh; a[3] += lam * eh; h[2] += lam * phi * eh; a[1] += lam * phi * eh  # noqa: E702
        a[2] += lam * ea; h[1] += lam * ea; a[0] += lam * phi * ea; h[3] += lam * phi * ea  # noqa: E702
        for rating in (h, a):  # le valutazioni dell'altro campo partono dalla prima osservazione
            for j in range(4):
                if np.isnan(rating[j]):
                    rating[j] = rating[j ^ 2] if not np.isnan(rating[j ^ 2]) else rating[j]
    return pred


def gap_over_under(df):
    """Logistica walk-forward: P(over) da logit quota media + GAP. Soglia scelta sul 2014-19."""
    d = df.assign(gap=gap_ratings(df), over=(df.home_goals + df.away_goals > 2.5).astype(float))
    d = d.dropna(subset=["gap", "pO", "avg_O", "avg_U"])
    d["lo"] = np.log(d.pO / (1 - d.pO))
    d["gapz"] = (d.gap - d.groupby(["div", "season"]).gap.transform(lambda s: s.expanding().mean().shift(1)))
    d = d.dropna(subset=["gapz"])
    preds = []
    for s in range(2014, d.season.max() + 1):
        tr, te = d[d.season < s], d[d.season == s]
        m = LogisticRegression(C=1.0).fit(tr[["lo", "gapz"]], tr.over)
        preds.append(pd.Series(m.predict_proba(te[["lo", "gapz"]])[:, 1], index=te.index))
    d = d.loc[pd.concat(preds).index].assign(p=pd.concat(preds))
    rows = []
    for side, p, odds, close, won in (("O", d.p, d.avg_O, d.cO, d.over), ("U", 1 - d.p, d.avg_U, d.cU, 1 - d.over)):
        rows.append(pd.DataFrame({"season": d.season, "odds": odds, "won": won, "p_close": close, "stake": 1.0,
                                  "ev": p * odds - 1}))
    b = pd.concat(rows)
    dev = b[b.season < SPLIT]
    best = max((0.0, 0.02, 0.04, 0.06, 0.08), key=lambda t: summary(dev[dev.ev > t], "", "")["ROI"]
               if (dev.ev > t).sum() >= 300 else -1)
    return b[b.ev > best], best


def main():
    warnings.filterwarnings("ignore")
    df = load()
    print(f"partite: {len(df)}")
    fav = np.where(df.p1 >= df.p2, "1", "2")
    p_fav = np.maximum(df.p1, df.p2)
    late = (df.h_played >= 2 * (df.n_teams - 1) - 6) & (df.a_played >= 2 * (df.n_teams - 1) - 6)
    danger = lambda pts: (pts - df.safe_pts).between(-6, 3)  # noqa: E731
    calm = lambda pts: ((pts - df.safe_pts) > 9) & ((df.fourth_pts - pts) > 9)  # noqa: E731
    early = (df.h_played < 8) & (df.a_played < 8)
    S = []
    add = lambda b, n, s: S.append(summary(b, n, s))  # noqa: E731
    add(bets(df, df.index >= 0, "1"), "Sempre la squadra di casa", "tesi Maynooth 2012")
    add(bets(df, df.index >= 0, "X"), "Sempre il pareggio", "Lezana 2026 (Premier)")
    add(bets(df, df.index >= 0, "2"), "Sempre la trasferta", "Lezana 2026 (Premier)")
    add(bets(df, (df.p1 - df.p2).abs() < 0.10, "X"), "Pareggio fra squadre equilibrate", "consiglio comune online")
    add(bets(df, (df.avg_X >= 3.0) & (df.avg_X <= 3.4), "X"), "Pareggio a quota 3.0-3.4", "consiglio comune online")
    add(bets(df, p_fav >= 0.70, pd.Series(fav, index=df.index)), "Favoriti forti (prob. >= 70%)",
        "bias favorito/sfavorito")
    add(bets(df, (p_fav >= 0.55) & (p_fav < 0.70), pd.Series(fav, index=df.index)), "Favoriti medi (55-70%)",
        "bias favorito/sfavorito")
    add(bets(df, df.p1 < df.p2, "1"), "Sfavorito in casa", "Borghesi 2007, Kennesaw")
    add(bets(df, (df.p1 < 0.30), "1"), "Sfavorito netto in casa (< 30%)", "Borghesi 2007")
    add(fibonacci_draws(df), "Pareggi con progressione Fibonacci (3 squadre/lega)", "Rigoni; Archontakis-Osborne")
    add(bets(df, late & danger(df.h_pts) & calm(df.a_pts), "1"),
        "Fine stagione: casa che lotta per salvarsi vs ospite senza obiettivi", "motivazione (NFL: Borghesi)")
    add(bets(df, late & danger(df.a_pts) & calm(df.h_pts), "2"),
        "Fine stagione: ospite che lotta per salvarsi vs casa senza obiettivi", "motivazione")
    add(bets(df, df.h_last_gd <= -3, "1"), "Casa dopo una sconfitta di 3+ gol", "reazione eccessiva del mercato")
    add(bets(df, df.a_last_gd <= -3, "2"), "Trasferta dopo una sconfitta di 3+ gol", "reazione eccessiva del mercato")
    add(bets(df, df.h_win_streak >= 5, "2"), "Contro chi ha vinto 5 di fila (casa)", "regressione dopo le serie")
    add(bets(df, df.a_win_streak >= 5, "1"), "Contro chi ha vinto 5 di fila (trasferta)", "regressione dopo le serie")
    add(bets(df, early & df.h_promoted.astype(bool), "1"), "Neopromossa in casa, prime 8 giornate",
        "entusiasmo delle neopromosse")
    add(bets(df, early & df.a_promoted.astype(bool), "1"), "Contro la neopromossa fuori casa, prime 8",
        "neopromosse sottovalutate/sopravvalutate")
    gb, t = gap_over_under(df)
    add(gb, f"GAP ratings tiri+angoli su Over/Under 2.5 (soglia {t:.0%} scelta sul 2014-19)",
        "Wheatcroft 2020 (LSE)")
    out = pd.DataFrame(S)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(out.round(4).to_string(index=False))
    out.to_csv(SCAN_DIR / "online_results.csv", index=False)


if __name__ == "__main__":
    main()

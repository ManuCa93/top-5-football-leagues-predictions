"""Grid search: quali informazioni sulle squadre migliorano le probabilità del mercato, con quali parametri,
e quali regole di gioco rendono di più.

Tutto walk-forward: ogni mese il modello si addestra solo sulle partite giocate prima del primo del mese
e prevede quelle del mese. Ogni feature di una partita usa solo dati precedenti al calcio d'inizio.
Il modello parte dalle quote di mercato (logit di 1 e 2 rispetto a X) e impara solo le correzioni.

DEV (scelta di feature, parametri e soglie): 2020/21 - 2024/25. HOLDOUT (una volta sola): 2025/26 - 2026/27.
"""
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import matches_with_market
from .config import DATA_DIR, season_of
from .data import load_matches
from .features import elo_before
from .odds import implied_goal_rates, market_summary, score_matrix
from .walkforward import DEV_SEASONS, HOLDOUT_SEASONS

RESEARCH_DIR = DATA_DIR / "research"
FIRST_SEASON = 2005  # prima stagione con quote medie (Betbrain) su football-data.co.uk
TRAIN_FIRST_SEASON = 2006  # la prima serve solo come "stagione precedente"
LEVEL_STATS = ("pts", "gd", "sotd", "shd", "cod", "gf", "ga", "sotf", "sota", "uxgd", "unpxgd", "deepd", "uppda",
               "xgf", "xga", "npxgf", "npxga")  # livello della squadra
RESIDUAL_STATS = ("res", "gres", "scres", "cnres", "totres")  # rendimento rispetto a quanto si aspettava il mercato


# ---------------------------------------------------------------- dati

def _fair_power(o1, ox, o2):
    q = 1 / np.array([o1, ox, o2])
    if not np.all(np.isfinite(q)):
        return [np.nan] * 3
    k = brentq(lambda k: (q ** k).sum() - 1, 0.3, 3.0)
    return list(q ** k)


def load_base():
    """Tutte le partite dal 2005/06 con probabilità di mercato (p_*), quote Snai stimate (q_*),
    probabilità di chiusura senza margine (c_*) ed esito y (0=1, 1=X, 2=2)."""
    m = load_matches(seasons=range(FIRST_SEASON, season_of(datetime.now()) + 1), refresh=False)
    base = matches_with_market(m)  # p_*, q_*, lambda_* in cache per partita
    close = np.array([_fair_power(*r) for r in base[["close_1", "close_x", "close_2"]].values])
    base["c_1"], base["c_X"], base["c_2"] = close.T
    base["c_1X"], base["c_X2"] = base.c_1 + base.c_X, base.c_2 + base.c_X
    base = base.merge(close_gg(base), on="match_id", how="left")
    base["c_NG"] = 1 - base.c_GG
    base["y"] = np.select([base.home_goals > base.away_goals, base.home_goals == base.away_goals], [0, 1], 2)
    return base.sort_values("date").reset_index(drop=True)


def close_gg(base):
    """P(GG) senza margine dalle quote di chiusura (1X2 + Over/Under 2.5), in cache per partita."""
    path = RESEARCH_DIR / "close_gg.pkl"
    cached = pd.read_pickle(path) if path.exists() else pd.DataFrame(columns=["match_id", "c_GG"])
    todo = base[~base.match_id.isin(cached.match_id)].dropna(subset=["close_1", "close_x", "close_2"])
    if len(todo):
        rows = []
        for r in todo.itertuples():
            lh, la = implied_goal_rates(r.close_1, r.close_x, r.close_2, r.close_over25, r.close_under25)
            rows.append({"match_id": r.match_id, "c_GG": market_summary(score_matrix(lh, la))["GG"]})
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
        cached.to_pickle(path)
    return cached


def team_long(base, with_advanced=False):
    """Una riga per (partita, squadra) con statistiche della partita e attese del mercato."""
    cols = ["match_id", "league", "season", "date"]
    h = base[cols].assign(team=base.home_team, is_home=1, gf=base.home_goals, ga=base.away_goals,
                          sotf=base.home_sot, sota=base.away_sot, shf=base.home_shots, sha=base.away_shots,
                          cof=base.home_corners, coa=base.away_corners, xpts=3 * base.p_1 + base.p_X,
                          xgd=base.lambda_home - base.lambda_away, lf=base.lambda_home, la=base.lambda_away)
    a = base[cols].assign(team=base.away_team, is_home=0, gf=base.away_goals, ga=base.home_goals,
                          sotf=base.away_sot, sota=base.home_sot, shf=base.away_shots, sha=base.home_shots,
                          cof=base.away_corners, coa=base.home_corners, xpts=3 * base.p_2 + base.p_X,
                          xgd=base.lambda_away - base.lambda_home, lf=base.lambda_away, la=base.lambda_home)
    long = pd.concat([h, a], ignore_index=True)
    if with_advanced:
        from .understat import advanced_stats
        long = long.merge(advanced_stats(base), on=["match_id", "is_home"], how="left")
        long["uxgd"] = long.xgf - long.xga
        long["unpxgd"] = long.npxgf - long.npxga
        long["deepd"] = long.deepf - long.deepa
        long["uppda"] = long.ppda_allowed - long.ppda  # positivo = pressa più dell'avversario
        long["uxres"] = long.uxgd - (long.lf - long.la)  # xG rispetto ai gol attesi dal mercato
        long["uxptsres"] = long.xpts_us - long.xpts
    long = long.sort_values(["team", "date"]).reset_index(drop=True)
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga], [3.0, 1.0], 0.0)
    long["gd"] = long.gf - long.ga
    long["sotd"] = long.sotf - long.sota
    long["shd"] = long.shf - long.sha
    long["cod"] = long.cof - long.coa
    long["res"] = long.pts - long.xpts
    long["gres"] = long.gd - long.xgd
    # per il GG: ha segnato / ha subito rispetto alla probabilità che dava il mercato, gol totali vs attesi
    long["scres"] = (long.gf > 0) - (1 - np.exp(-long.lf))
    long["cnres"] = (long.ga > 0) - (1 - np.exp(-long.la))
    long["totres"] = long.gf + long.ga - long.lf - long.la
    long.loc[long.gf.isna(), ["pts", "res", "scres", "cnres"]] = np.nan  # partite non ancora giocate
    return long


# ---------------------------------------------------------------- famiglie di feature

def _to_match(long, values, name):
    """Da valori per (partita, squadra) a colonne per partita: casa, trasferta e differenza."""
    s = long[["match_id", "is_home"]].assign(v=values.values)
    h = s[s.is_home == 1].set_index("match_id").v
    a = s[s.is_home == 0].set_index("match_id").v
    return pd.DataFrame({f"{name}_h": h, f"{name}_a": a, f"{name}_diff": h - a})


def season_weighted(long, stat, m1, m2, venue=False):
    """Media della stagione in corso fino alla partita precedente, con le due stagioni prima che pesano
    come m1 e m2 partite. Esempio m1=10, m2=3 a metà stagione (19 partite): 59% questa, 31% la scorsa,
    9% due stagioni fa. Le neopromosse ricevono il livello delle squadre peggiori della lega."""
    keys = ["team", "is_home"] if venue else ["team"]
    df = long[keys + ["league", "season", stat]].copy()
    g = df.groupby(keys + ["season"])[stat]
    val = df[stat].fillna(0)
    cum = val.groupby([df[k] for k in keys + ["season"]]).cumsum() - val
    n = df[stat].notna().astype(int).groupby([df[k] for k in keys + ["season"]]).cumsum() - df[stat].notna()
    season_mean = g.mean()
    q = 0.2 if stat in LEVEL_STATS else 0.5
    league_prior = (season_mean.reset_index().merge(df[keys + ["season", "league"]].drop_duplicates(keys + ["season"]))
                    .groupby(["league", "season"] + (["is_home"] if venue else []))[stat].quantile(q))
    rating_num, rating_den = cum.astype(float), n.astype(float)
    for back, weight in ((1, m1), (2, m2)):
        if weight == 0:
            continue
        idx = [df[k] for k in keys] + [df.season - back]
        prev = season_mean.reindex(pd.MultiIndex.from_arrays(idx)).values
        pidx = [df.league, df.season - back] + ([df.is_home] if venue else [])
        prior = league_prior.reindex(pd.MultiIndex.from_arrays(pidx)).values
        prev = np.where(np.isnan(prev), prior, prev)
        ok = ~np.isnan(prev)
        rating_num += np.where(ok, weight * np.nan_to_num(prev), 0)
        rating_den += np.where(ok, weight, 0)
    rating = rating_num / rating_den.replace(0, np.nan)
    if venue:  # la squadra di casa usa il rendimento in casa, quella in trasferta fuori
        return _to_match(long, rating, f"swv_{stat}_{m1}_{m2}")
    return _to_match(long, rating, f"sw_{stat}_{m1}_{m2}")


def ewm_strength(long, stat, halflife):
    """Alternativa ai pesi per stagione: media con decadimento esponenziale (mezza vita in partite)."""
    v = long.groupby("team")[stat].transform(lambda s: s.ewm(halflife=halflife, ignore_na=True).mean().shift(1))
    return _to_match(long, v, f"ewm_{stat}_{halflife}")


def form_last_n(long, stat, n):
    v = long.groupby("team")[stat].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    return _to_match(long, v, f"last{n}_{stat}")


def form_last_days(long, stat, days):
    """Media e numero di partite di campionato negli ultimi `days` giorni (partita corrente esclusa)."""
    out = []
    for _, g in long.groupby("team"):
        r = g.set_index("date")[stat].rolling(f"{days}D", closed="left")
        out.append(pd.DataFrame({"mean": r.mean().values, "n": r.count().values}, index=g.index))
    o = pd.concat(out).reindex(long.index)
    return pd.concat([_to_match(long, o["mean"], f"days{days}_{stat}"),
                      _to_match(long, o["n"], f"days{days}_n")], axis=1)


def rest_league(long):
    v = long.groupby("team").date.diff().dt.total_seconds() / 86400
    return _to_match(long, v.clip(upper=14), "rest_lg")


def schedule_all(long, euro):
    """Riposo e carico contando campionato + coppe europee (openfootball, dal 2011/12; Europa League dal 2020/21).
    euro_prev / euro_next: partita europea nei 4 giorni prima / dopo (il calendario è noto in anticipo)."""
    lg_days = long.date.dt.normalize()
    out = pd.DataFrame(index=long.index, columns=["rest_all", "n14_all", "euro_prev", "euro_next"], dtype=float)
    eu = {t: np.sort(g.date.values.astype("datetime64[D]")) for t, g in euro.groupby("team")}
    for team, idx in long.groupby("team").groups.items():
        lg = np.sort(lg_days[idx].values.astype("datetime64[D]"))
        e = eu.get(team, np.array([], dtype="datetime64[D]"))
        allm = np.sort(np.concatenate([lg, e]))
        d = lg_days[idx].values.astype("datetime64[D]")
        prev_i = np.searchsorted(allm, d, side="left") - 1
        last = np.where(prev_i >= 0, allm[np.clip(prev_i, 0, None)], np.datetime64("NaT"))
        out.loc[idx, "rest_all"] = np.clip((d - last) / np.timedelta64(1, "D"), None, 14)
        out.loc[idx, "n14_all"] = np.searchsorted(allm, d, side="left") - np.searchsorted(allm, d - 14, side="left")
        out.loc[idx, "euro_prev"] = (np.searchsorted(e, d, side="left") - np.searchsorted(e, d - 4, side="left")) > 0
        out.loc[idx, "euro_next"] = (np.searchsorted(e, d + 4, side="right") - np.searchsorted(e, d, side="right")) > 0
    out = out.astype(float)
    return pd.concat([_to_match(long, out[c], c) for c in out.columns], axis=1)


def elo(base, k, home, carry):
    e, _ = elo_before(base, k=k, home=home, carry=carry)
    e = e.set_index("match_id")
    return pd.DataFrame({f"elo_{k}_{home}_{carry}_diff": e.elo_home - e.elo_away})


# ---------------------------------------------------------------- modello e metriche

def month_starts(seasons):
    first = datetime(min(seasons), 7, 1)
    last = datetime(max(seasons) + 1, 7, 1)
    return list(pd.date_range(first, last, freq="MS"))


TARGETS = {"1x2": ("1", "X", "2"), "gg": ("NG", "GG")}


def design(base, feats, cols, train_from=TRAIN_FIRST_SEASON, target="1x2"):
    """Righe utilizzabili e matrice X: logit di mercato (1 e 2 rispetto a X, oppure GG), più le feature scelte."""
    df = base[(base.season >= train_from) & base.p_1.notna() & base.p_GG.notna()]
    if target == "1x2":
        X = np.column_stack([np.log(df.p_1 / df.p_X), np.log(df.p_2 / df.p_X)])
        y = df.y.values
    else:
        X = np.log(df.p_GG / df.p_NG).values[:, None]
        y = ((df.home_goals > 0) & (df.away_goals > 0)).astype(int).values
    if cols:
        X = np.column_stack([X, feats.reindex(df.match_id)[list(cols)].values])
    meta = df[["match_id", "date", "season", "league"]].reset_index(drop=True).assign(y=y)
    meta.attrs["target"] = target
    return meta, X


def fit_predict(meta, X, C=1.0, seasons=DEV_SEASONS, halflife_years=None, refit_months=1, target=None):
    """Ogni `refit_months` mesi riaddestra sulle partite precedenti e prevede le successive.
    halflife_years: le partite vecchie pesano meno (il calcio cambia), peso dimezzato ogni N anni."""
    y = meta.y.values
    dates = meta.date.values
    classes = TARGETS[target or meta.attrs.get("target", "1x2")]
    out = np.full((len(meta), len(classes)), np.nan)
    months = month_starts(seasons)[::refit_months] + [month_starts(seasons)[-1]]
    in_seasons = meta.season.isin(seasons).values
    for ms, me in zip(months[:-1], months[1:]):
        ms, me = np.datetime64(ms), np.datetime64(me)
        test = (dates >= ms) & (dates < me) & in_seasons
        if not test.any():
            continue
        train = dates < ms
        w = None
        if halflife_years:
            age = (ms - dates[train]) / np.timedelta64(1, "D") / 365.25
            w = 0.5 ** (age / halflife_years)
        model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=C, max_iter=1000))
        model.fit(X[train], y[train], logisticregression__sample_weight=w)
        out[test] = model.predict_proba(X[test])
    keep = ~np.isnan(out[:, 0])
    res = meta[keep].copy()
    for i, c in enumerate(classes):
        res[f"m_{c}"] = out[keep, i]
    return res


def walk_forward(base, feats, cols, C=1.0, seasons=DEV_SEASONS, train_from=TRAIN_FIRST_SEASON,
                 halflife_years=None, refit_months=1, target="1x2"):
    """Probabilità previste walk-forward. cols=[] = solo quote (ricalibrate)."""
    meta, X = design(base, feats, cols, train_from, target)
    return fit_predict(meta, X, C, seasons, halflife_years, refit_months, target)


def logloss_delta(pred, base):
    """Log loss del modello e del mercato sulle stesse partite, con errore standard della differenza."""
    b = base.set_index("match_id").loc[pred.match_id]
    pm = pred[["m_1", "m_X", "m_2"]].values[np.arange(len(pred)), pred.y.values]
    pk = b[["p_1", "p_X", "p_2"]].values[np.arange(len(pred)), pred.y.values]
    d = -np.log(pm) + np.log(pk)
    return {"ll_model": float(-np.log(pm).mean()), "ll_market": float(-np.log(pk).mean()),
            "delta": float(d.mean()), "se": float(d.std() / np.sqrt(len(d))), "n": len(d)}


VALUE_MARKETS = ("1", "X", "2", "1X", "X2")


def value_bets(pred, base, threshold=0.0, markets=VALUE_MARKETS, max_odds=10.0):
    """Scommesse singole dove p_modello x quota Snai > 1 + soglia. Restituisce una riga per giocata con
    l'esito vero e il valore atteso secondo le quote di chiusura (molto meno rumoroso del P&L)."""
    b = base.set_index("match_id").loc[pred.match_id].reset_index()
    hg, ag = b.home_goals.values, b.away_goals.values
    if "m_GG" in pred:
        p = {"GG": pred.m_GG.values, "NG": pred.m_NG.values}
        won = {"GG": (hg > 0) & (ag > 0), "NG": (hg == 0) | (ag == 0)}
    else:
        p = {"1": pred.m_1.values, "X": pred.m_X.values, "2": pred.m_2.values}
        p["1X"], p["X2"] = p["1"] + p["X"], p["2"] + p["X"]
        won = {"1": hg > ag, "X": hg == ag, "2": hg < ag, "1X": hg >= ag, "X2": hg <= ag}
    rows = []
    for k in markets:
        q = b[f"q_{k}"].values
        sel = (p[k] * q > 1 + threshold) & (q <= max_odds)
        rows.append(pd.DataFrame({"match_id": b.match_id.values[sel], "date": b.date.values[sel], "market": k,
                                  "p": p[k][sel], "q": q[sel], "won": won[k][sel],
                                  "ev_close": b[f"c_{k}"].values[sel] * q[sel] - 1}))
    bets = pd.concat(rows, ignore_index=True)
    bets["pnl"] = np.where(bets.won, bets.q - 1, -1.0)
    return bets


def bets_summary(bets):
    if bets.empty:
        return {"bets": 0, "roi": np.nan, "roi_se": np.nan, "ev_close": np.nan}
    return {"bets": len(bets), "roi": bets.pnl.mean(), "roi_se": bets.pnl.std() / np.sqrt(len(bets)),
            "ev_close": bets.ev_close.mean(), "ev_model": (bets.p * bets.q - 1).mean()}


def timer():
    t0 = time.time()
    return lambda: f"{time.time() - t0:.0f}s"


# ---------------------------------------------------------------- banca delle feature

SW_GRID = [(m1, m2) for m1 in (0, 5, 10, 19, 38) for m2 in (0, 3, 8)]
EWM_GRID = (5, 10, 20, 40)
LASTN_GRID = (3, 5, 8, 12)
DAYS_GRID = (14, 21, 30)
ELO_GRID = [(k, h, c) for k in (10, 20, 30) for h in (40, 70) for c in (0.7, 0.9)]
STRENGTH_STATS = ("pts", "gd", "sotd", "shd", "cod", "res", "gres")
GG_STATS = ("gf", "ga", "sotf", "sota", "shf", "sha", "scres", "cnres", "totres")
# Understat (dal 2014/15)
ADV_STRENGTH_STATS = ("uxgd", "unpxgd", "deepd", "uppda", "uxres", "uxptsres")
ADV_GG_STATS = ("xgf", "xga", "npxgf", "npxga")
SW_GRID_GG = [(m1, m2) for m1 in (0, 10, 38) for m2 in (0, 5)]


def feature_bank(base, sofa=True):
    """Tutte le feature candidate con tutti i parametri della grid (in cache)."""
    path = RESEARCH_DIR / f"bank{'_sofa' if sofa else ''}.pkl"
    if path.exists() and path.stat().st_mtime > max(p.stat().st_mtime for p in RESEARCH_DIR.parent.glob("raw/*.csv")):
        return pd.read_pickle(path)
    t = timer()
    long = team_long(base, with_advanced=True)
    parts = []
    for stat in STRENGTH_STATS + ADV_STRENGTH_STATS:
        for m1, m2 in SW_GRID:
            parts.append(season_weighted(long, stat, m1, m2))
            parts.append(season_weighted(long, stat, m1, m2, venue=True))
        for h in EWM_GRID:
            parts.append(ewm_strength(long, stat, h))
        for n in LASTN_GRID:
            parts.append(form_last_n(long, stat, n))
    for stat in GG_STATS + ADV_GG_STATS:
        for m1, m2 in SW_GRID_GG:
            parts.append(season_weighted(long, stat, m1, m2))
            parts.append(season_weighted(long, stat, m1, m2, venue=True))
        for h in EWM_GRID:
            parts.append(ewm_strength(long, stat, h))
        for n in (5, 10):
            parts.append(form_last_n(long, stat, n))
    for d in DAYS_GRID:
        parts.append(form_last_days(long, "pts", d))
        parts.append(form_last_days(long, "res", d))
    parts.append(rest_league(long))
    from .europe import european_dates
    parts.append(schedule_all(long, european_dates(base)))
    print(f"  squadre: {t()}", flush=True)
    for k, h, c in ELO_GRID:
        parts.append(elo(base, k, h, c))
    print(f"  elo: {t()}", flush=True)
    if sofa:
        from .sofa_features import build
        parts.append(build(base))
        print(f"  sofascore: {t()}", flush=True)
    bank = pd.concat([p.loc[:, ~p.columns.duplicated()] for p in parts], axis=1)
    bank = bank.loc[:, ~bank.columns.duplicated()].astype(np.float32)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    bank.to_pickle(path)
    return bank

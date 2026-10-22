"""Caccia alle inefficienze: tutti i campionati di football-data.co.uk con le quote Pinnacle (dal 2012/13),
mercati 1X2 e Over/Under 2.5. Tre famiglie di regole, scelte sul 2012-2019 e verificate sul 2020-oggi:

A. valore contro Pinnacle: quota media (proxy Snai) x probabilità equa di Pinnacle > 1 + t
B. distorsioni fisse: (campionato, esito, fascia di quota) in attivo sul passato
C. gradient boosting su quote, struttura del mercato e storia delle squadre (Elo, forma, riposo, gol)

Metriche: ROI alla quota media e CLV contro la chiusura Pinnacle senza margine (più stabile del ROI).
Esegui: python -m predictor.scan
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import DATA_DIR, RAW_DIR
from .data import BASE_URL, _read_csv_bytes, season_code
from .features import elo_before

DIVS = {
    "E0": "Premier League", "E1": "Championship", "E2": "League One", "E3": "League Two", "EC": "National League",
    "SC0": "Scozia", "SC1": "Scozia 2", "D1": "Bundesliga", "D2": "2. Bundesliga", "I1": "Serie A", "I2": "Serie B",
    "SP1": "La Liga", "SP2": "Segunda", "F1": "Ligue 1", "F2": "Ligue 2", "N1": "Eredivisie", "B1": "Belgio",
    "P1": "Portogallo", "T1": "Turchia", "G1": "Grecia",
}
FIRST, SPLIT, LAST = 2012, 2020, 2026  # regole scelte su FIRST..SPLIT-1, verificate su SPLIT..LAST
OUTCOMES = ("1", "X", "2", "O", "U")
COLS = {  # nome interno -> colonne possibili nel CSV, in ordine di preferenza
    "pin_1": ("PSH",), "pin_X": ("PSD",), "pin_2": ("PSA",),
    "pinc_1": ("PSCH",), "pinc_X": ("PSCD",), "pinc_2": ("PSCA",),
    "avg_1": ("AvgH", "BbAvH"), "avg_X": ("AvgD", "BbAvD"), "avg_2": ("AvgA", "BbAvA"),
    "max_1": ("MaxH", "BbMxH"), "max_X": ("MaxD", "BbMxD"), "max_2": ("MaxA", "BbMxA"),
    "b365_1": ("B365H",), "b365_X": ("B365D",), "b365_2": ("B365A",),
    "pin_O": ("P>2.5",), "pin_U": ("P<2.5",), "pinc_O": ("PC>2.5",), "pinc_U": ("PC<2.5",),
    "avg_O": ("Avg>2.5", "BbAv>2.5"), "avg_U": ("Avg<2.5", "BbAv<2.5"),
    "max_O": ("Max>2.5", "BbMx>2.5"), "max_U": ("Max<2.5", "BbMx<2.5"),
}
STATS = {"home_shots": "HS", "away_shots": "AS", "home_sot": "HST", "away_sot": "AST", "home_corners": "HC",
         "away_corners": "AC"}
SCAN_DIR = DATA_DIR / "research"


def _raw(div, season):
    path = RAW_DIR / f"{div}_{season_code(season)}.csv"
    if not path.exists() or season == LAST:
        try:
            time.sleep(0.5)
            r = requests.get(BASE_URL.format(code=season_code(season), div=div), timeout=30)
            r.raise_for_status()
            path.write_bytes(r.content)
        except requests.RequestException as e:
            if not path.exists():
                print(f"[WARN] {div} {season}: {e}")
                return None
    return _read_csv_bytes(path.read_bytes())


def load_all():
    frames = []
    for div in DIVS:
        for season in range(FIRST, LAST + 1):
            raw = _raw(div, season)
            if raw is None or "HomeTeam" not in raw:
                continue
            raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            d = pd.DataFrame({"div": div, "season": season, "home_team": raw.HomeTeam, "away_team": raw.AwayTeam,
                              "home_goals": raw.FTHG.astype(float), "away_goals": raw.FTAG.astype(float)})
            d["date"] = pd.to_datetime(raw.Date, dayfirst=True, format="mixed")
            for name, options in COLS.items():
                col = next((c for c in options if c in raw), None)
                d[name] = pd.to_numeric(raw[col], errors="coerce") if col else np.nan
            for name, col in STATS.items():
                d[name] = pd.to_numeric(raw[col], errors="coerce") if col in raw else np.nan
            frames.append(d)
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "div"]).reset_index(drop=True)
    df["match_id"] = np.arange(len(df))
    return df


def fair(odds):
    """Power de-vig vettoriale: righe di quote -> probabilità senza margine (NaN se manca una quota)."""
    q = 1 / np.asarray(odds, dtype=float)
    ok = np.isfinite(q).all(1) & (q > 0).all(1)
    lo, hi = np.full(len(q), 0.3), np.full(len(q), 3.0)
    qq = np.where(ok[:, None], q, 0.5)
    for _ in range(50):
        k = (lo + hi) / 2
        s = (qq ** k[:, None]).sum(1)
        lo, hi = np.where(s > 1, k, lo), np.where(s > 1, hi, k)
    p = qq ** ((lo + hi) / 2)[:, None]
    p[~ok] = np.nan
    return p


def team_features(df):
    """Elo, punti nelle ultime 5, gol fatti/subiti nelle ultime 10 e riposo: solo partite precedenti."""
    country = df["div"].str.extract(r"^([A-Z]+)")[0]
    m = df.assign(home_team=country + ":" + df.home_team, away_team=country + ":" + df.away_team)
    elo, _ = elo_before(m)
    long = pd.concat([
        m[["match_id", "date", "home_team", "home_goals", "away_goals"]].set_axis(
            ["match_id", "date", "team", "gf", "ga"], axis=1).assign(is_home=1),
        m[["match_id", "date", "away_team", "away_goals", "home_goals"]].set_axis(
            ["match_id", "date", "team", "gf", "ga"], axis=1).assign(is_home=0),
    ]).sort_values(["team", "date"])
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga], [3, 1], 0)
    g = long.groupby("team")
    long["pts5"] = g.pts.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    long["gf10"] = g.gf.transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    long["ga10"] = g.ga.transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    long["rest"] = g.date.diff().dt.days.clip(upper=30)
    cols = ["pts5", "gf10", "ga10", "rest"]
    h = long[long.is_home == 1].set_index("match_id")[cols].add_prefix("h_")
    a = long[long.is_home == 0].set_index("match_id")[cols].add_prefix("a_")
    out = elo.set_index("match_id").join(h).join(a)
    out["elo_diff"] = out.elo_home - out.elo_away
    return out.drop(columns=["elo_home", "elo_away"])


def candidates(df):
    """Una riga per (partita, esito) con quota media, probabilità eque e risultato."""
    p_pin = {"1X2": fair(df[["pin_1", "pin_X", "pin_2"]]), "OU": fair(df[["pin_O", "pin_U"]])}
    p_pinc = {"1X2": fair(df[["pinc_1", "pinc_X", "pinc_2"]]), "OU": fair(df[["pinc_O", "pinc_U"]])}
    p_avg = {"1X2": fair(df[["avg_1", "avg_X", "avg_2"]]), "OU": fair(df[["avg_O", "avg_U"]])}
    margin = {"1X2": (1 / df[["avg_1", "avg_X", "avg_2"]]).sum(1).values - 1,
              "OU": (1 / df[["avg_O", "avg_U"]]).sum(1).values - 1}
    hg, ag = df.home_goals.values, df.away_goals.values
    won = {"1": hg > ag, "X": hg == ag, "2": hg < ag, "O": hg + ag > 2.5, "U": hg + ag < 2.5}
    feats = team_features(df).loc[df.match_id].reset_index(drop=True)
    rows = []
    for o in OUTCOMES:
        mk, i = ("1X2", "1X2".index(o)) if o in "1X2" else ("OU", "OU".index(o))
        r = pd.DataFrame({"match_id": df.match_id, "div": df["div"], "season": df.season, "date": df.date,
                          "outcome": o, "market": mk, "odds": df[f"avg_{o}"].values,
                          "p_pin": p_pin[mk][:, i], "p_avg": p_avg[mk][:, i], "p_close": p_pinc[mk][:, i],
                          "margin": margin[mk], "max_ratio": (df[f"max_{o}"] / df[f"avg_{o}"]).values,
                          "b365_ratio": (df.get(f"b365_{o}", np.nan) / df[f"avg_{o}"]).values
                          if f"b365_{o}" in df else np.nan,
                          "won": won[o].astype(float)})
        rows.append(pd.concat([r, feats], axis=1))
    c = pd.concat(rows, ignore_index=True)
    c = c[np.isfinite(c.odds) & (c.odds > 1)]
    c["pin_gap"] = c.p_pin * c.odds - 1  # >0: la media paga più del prezzo equo Pinnacle
    c["month"] = c.date.dt.month
    return c.reset_index(drop=True)


def summary(b, label):
    if len(b) == 0:
        return {"regola": label, "giocate": 0}
    pnl = np.where(b.won == 1, b.odds - 1, -1.0)
    by = pd.Series(pnl).groupby(b.season.values).mean()
    clv = (b.odds * b.p_close - 1).dropna()
    return {"regola": label, "giocate": len(b), "ROI": pnl.mean(), "ROI_se": pnl.std() / np.sqrt(len(b)),
            "CLV": clv.mean() if len(clv) else np.nan, "stagioni+": f"{(by > 0).sum()}/{len(by)}"}


def rule_pin_value(c, t, max_odds):
    return c[(c.pin_gap > t) & (c.odds <= max_odds)]


BUCKETS = [1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 100.0]


def rule_bias_cells(c, train, min_n=300, min_roi=0.03):
    c = c.assign(bucket=pd.cut(c.odds, BUCKETS))
    tr = c[c.season.isin(train)]
    roi = tr.assign(pnl=np.where(tr.won == 1, tr.odds - 1, -1.0)).groupby(
        ["div", "outcome", "bucket"], observed=True).pnl.agg(["mean", "size"])
    good = roi[(roi["size"] >= min_n) & (roi["mean"] > min_roi)].index
    key = pd.MultiIndex.from_frame(c[["div", "outcome", "bucket"]])
    return c[key.isin(good)], len(good)


ML_FEATURES = ["p_pin", "p_avg", "odds", "margin", "max_ratio", "b365_ratio", "pin_gap", "month", "elo_diff",
               "h_pts5", "a_pts5", "h_gf10", "a_gf10", "h_ga10", "a_ga10", "h_rest", "a_rest"]


def ml_walk_forward(c, seasons):
    """Per ogni stagione, un modello addestrato solo sulle stagioni precedenti predice P(esito vinto)."""
    c = c.assign(div_code=c["div"].astype("category").cat.codes, out_code=c.outcome.map(
        {o: i for i, o in enumerate(OUTCOMES)}))
    feats = ML_FEATURES + ["div_code", "out_code"]
    preds = []
    for s in seasons:
        tr, te = c[c.season < s], c[c.season == s]
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                               min_samples_leaf=200, l2_regularization=1.0,
                                               categorical_features=[feats.index("div_code"), feats.index("out_code")],
                                               random_state=0)
        model.fit(tr[feats], tr.won)
        preds.append(pd.Series(model.predict_proba(te[feats])[:, 1], index=te.index))
        print(f"  ML stagione {s}: addestrato su {len(tr)} righe", flush=True)
    return pd.concat(preds)


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def main():
    warnings.filterwarnings("ignore")
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    cache = SCAN_DIR / "scan_candidates.pkl"
    if cache.exists() and "--refresh" not in sys.argv:
        c = pd.read_pickle(cache)
    else:
        df = load_all()
        print(f"partite: {len(df)} in {df['div'].nunique()} campionati")
        c = candidates(df)
        c.to_pickle(cache)
    c = c[c.season >= FIRST]
    train_seasons, test_seasons = range(FIRST, SPLIT), range(SPLIT, LAST + 1)
    tr, te = c[c.season < SPLIT], c[c.season >= SPLIT]
    res = []

    # A. valore contro Pinnacle: scelgo soglia e quota massima sul train
    grid = [(t, m) for t in (0.0, 0.01, 0.02, 0.03, 0.05) for m in (2.5, 4.0, 10.0)]
    best = max(grid, key=lambda g: summary(rule_pin_value(tr.dropna(subset=["p_close"]), *g), "")
               .get("CLV", -1) if len(rule_pin_value(tr, *g)) >= 300 else -1)
    for name, part in (("train", tr), ("TEST", te)):
        res.append(summary(rule_pin_value(part, *best), f"A valore vs Pinnacle t={best[0]} q<={best[1]} [{name}]"))

    # B. distorsioni fisse per campionato/esito/fascia di quota
    for min_roi in (0.03, 0.06):
        sel, n_cells = rule_bias_cells(c, train_seasons, min_roi=min_roi)
        res.append(summary(sel[sel.season < SPLIT], f"B celle ROI>{min_roi:.0%} ({n_cells}) [train]"))
        res.append(summary(sel[sel.season >= SPLIT], f"B celle ROI>{min_roi:.0%} ({n_cells}) [TEST]"))

    # C. gradient boosting walk-forward; soglia scelta sul 2016-2019, verificata dopo
    ml_path = SCAN_DIR / "scan_ml.pkl"
    if ml_path.exists() and "--refresh" not in sys.argv:
        pred = pd.read_pickle(ml_path)
    else:
        pred = ml_walk_forward(c, range(2016, LAST + 1))
        pred.to_pickle(ml_path)
    m = c.loc[pred.index].assign(p_ml=pred)
    ok = m.p_pin.notna()
    for name, part in (("2016-19", m[ok & (m.season < SPLIT)]), ("2020-oggi", m[ok & (m.season >= SPLIT)])):
        print(f"log loss {name}: Pinnacle {logloss(part.p_pin, part.won):.5f}  ML {logloss(part.p_ml, part.won):.5f}"
              f"  media {logloss(part.p_avg, part.won):.5f}")
    dev, test = m[m.season < SPLIT], m[m.season >= SPLIT]
    grid = [(t, mo) for t in (0.0, 0.02, 0.04, 0.07, 0.10) for mo in (2.5, 4.0, 10.0)]
    pick = lambda d, t, mo: d[(d.p_ml * d.odds - 1 > t) & (d.odds <= mo)]  # noqa: E731
    best = max(grid, key=lambda g: summary(pick(dev, *g), "").get("ROI", -1) if len(pick(dev, *g)) >= 300 else -1)
    res.append(summary(pick(dev, *best), f"C gradient boosting t={best[0]} q<={best[1]} [2016-19]"))
    res.append(summary(pick(test, *best), f"C gradient boosting t={best[0]} q<={best[1]} [TEST]"))
    for mk in ("1X2", "OU"):
        res.append(summary(pick(test[test.market == mk], *best), f"C ... solo {mk} [TEST]"))

    # riferimenti
    res.append(summary(te[te.market == "1X2"].sample(frac=1, random_state=0).groupby("match_id").head(1),
                       "riferimento: un esito 1X2 a caso [TEST]"))
    out = pd.DataFrame(res)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print(out.round(4).to_string(index=False))
    out.to_csv(SCAN_DIR / "scan_results.csv", index=False)


if __name__ == "__main__":
    main()

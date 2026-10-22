"""Simulazione "2 doppie a giornata" su tutti e 5 i campionati, dal 2010/11 a oggi.

Ogni giornata (finestra venerdì 12:00 -> martedì 12:00 per il weekend, martedì 12:00 -> venerdì 12:00 per i turni
infrasettimanali) si scelgono 2 schedine da 2 gambe su 4 partite diverse, con le sole quote pre-partita.
Puntata fissa per schedina, quote Snai stimate (1X2 = media di mercato, doppia chance e GG/NG ricostruite).

  python -m predictor.doubles
"""
import json
import warnings

import numpy as np
import pandas as pd

from .config import REPORTS_DIR
from .research import RESEARCH_DIR, design, feature_bank, fit_predict, load_base, timer

STAKE = 5.0
FIRST_SEASON = 2010
MARKETS = ("1", "X", "2", "1X", "X2", "GG", "NG")
MIN_LEG_ODDS = 1.20
WINNERS = {
    "1": lambda h, a: h > a, "X": lambda h, a: h == a, "2": lambda h, a: h < a,
    "1X": lambda h, a: h >= a, "X2": lambda h, a: h <= a,
    "GG": lambda h, a: (h > 0) & (a > 0), "NG": lambda h, a: (h == 0) | (a == 0),
}


def model_probs(base, seasons):
    """Probabilità 1X2 del modello scelto dalla grid (quote + punti pesati per stagione), walk-forward."""
    cfg = json.loads((RESEARCH_DIR / "best_config_1x2_all.json").read_text())
    meta, X = design(base, feature_bank(base, sofa=False), cfg["columns"], cfg["train_from"])
    pred = fit_predict(meta, X, C=cfg["C"], seasons=seasons, halflife_years=cfg["halflife_years"])
    return pred.set_index("match_id")[["m_1", "m_X", "m_2"]]


def legs_table(base, model):
    """Una riga per (partita, mercato) con p di mercato, p del modello, quota Snai, p di chiusura, esito."""
    b = base.join(model, on="match_id")
    b["m_1X"], b["m_X2"] = b.m_1 + b.m_X, b.m_2 + b.m_X
    rows = []
    for k in MARKETS:
        rows.append(pd.DataFrame({
            "match_id": b.match_id, "date": b.date, "season": b.season, "market": k,
            "p": b[f"p_{k}"], "pm": b.get(f"m_{k}", b[f"p_{k}"]), "q": b[f"q_{k}"], "c": b.get(f"c_{k}"),
            "won": WINNERS[k](b.home_goals, b.away_goals)}))
    legs = pd.concat(rows, ignore_index=True).dropna(subset=["p", "q"])
    legs = legs[legs.q >= MIN_LEG_ODDS]
    # giornata: weekend (ven 12:00 - mar 12:00) o turno infrasettimanale (mar 12:00 - ven 12:00)
    t = legs.date - pd.Timedelta(hours=12)
    wd = t.dt.weekday
    start = t.dt.normalize() - pd.to_timedelta(np.where(wd >= 4, wd - 4, np.where(wd <= 0, wd + 3, wd - 1)), unit="D")
    legs["round"] = start + pd.Timedelta(hours=12)
    return legs


def _pairs(cands, odds_range=None, key="p"):
    """Due doppie su 4 partite diverse: la coppia migliore per `key` (con quota totale nell'intervallo), poi la
    migliore fra le partite rimaste."""
    cands = cands.sort_values(key, ascending=False).drop_duplicates("match_id").head(30)
    recs = cands.to_dict("records")
    used, slips = set(), []
    for _ in range(2):
        best = None
        for i in range(len(recs)):
            if recs[i]["match_id"] in used:
                continue
            for j in range(i + 1, len(recs)):
                if recs[j]["match_id"] in used:
                    continue
                q = recs[i]["q"] * recs[j]["q"]
                if odds_range and not odds_range[0] <= q <= odds_range[1]:
                    continue
                score = recs[i][key] * recs[j][key]
                if best is None or score > best[0]:
                    best = (score, recs[i], recs[j])
            if best is not None and not odds_range:
                break  # senza vincoli la coppia migliore è con il primo
        if best is None:
            break
        slips.append([best[1], best[2]])
        used.update({best[1]["match_id"], best[2]["match_id"]})
    return slips


STRATEGIES = {
    "Favoriti più probabili (1/2)": lambda w, rng: _pairs(w[w.market.isin(["1", "2"])]),
    "Doppie chance più probabili": lambda w, rng: _pairs(w[w.market.isin(["1X", "X2"])]),
    "Meno costosa, quota 1.6-2.6": lambda w, rng: _pairs(w.assign(ev=w.p * w.q), (1.6, 2.6), key="ev"),
    "Meno costosa, quota 2.5-4": lambda w, rng: _pairs(w.assign(ev=w.p * w.q), (2.5, 4.0), key="ev"),
    "Modello (pesi stagioni), valore": lambda w, rng: _pairs(
        w[w.market.isin(["1", "X", "2", "1X", "X2"])].assign(ev=lambda d: d.pm * d.q), (1.6, 3.0), key="ev"),
    "GG/NG più probabili": lambda w, rng: _pairs(w[w.market.isin(["GG", "NG"])]),
    "Casuale (quote 1.3-2.0)": lambda w, rng: _pairs(
        w[w.q.between(1.3, 2.0)].assign(r=rng.random(len(w[w.q.between(1.3, 2.0)]))), key="r"),
}


def simulate(legs):
    rng = np.random.default_rng(0)
    out = []
    for rnd, w in legs.groupby("round"):
        for name, pick in STRATEGIES.items():
            for slip in pick(w, rng):
                q = slip[0]["q"] * slip[1]["q"]
                won = bool(slip[0]["won"] and slip[1]["won"])
                c = slip[0]["c"] * slip[1]["c"] if pd.notna(slip[0]["c"]) and pd.notna(slip[1]["c"]) else np.nan
                out.append({"strategia": name, "round": rnd, "season": slip[0]["season"], "quota": q,
                            "prob": slip[0]["p"] * slip[1]["p"], "won": won,
                            "pnl": STAKE * (q - 1) if won else -STAKE, "ev_close": c * q - 1,
                            "gambe": f"{slip[0]['market']}+{slip[1]['market']}"})
    return pd.DataFrame(out)


def summarize(sim):
    rows = []
    rng = np.random.default_rng(1)
    for name, g in sim.groupby("strategia", sort=False):
        per_round = g.groupby("round").pnl.sum()
        staked = len(g) * STAKE
        boot = [per_round.sample(len(per_round), replace=True, random_state=rng.integers(1 << 31)).sum() / staked
                for _ in range(2000)]
        lo, hi = np.percentile(boot, [5, 95])
        curve = per_round.cumsum()
        by_season = g.groupby("season").pnl.agg(["sum", "size"])
        seasons = by_season[by_season["size"] >= 0.5 * by_season["size"].median()]["sum"]  # esclude la stagione in corso
        # probabilità di essere in attivo dopo N stagioni pescando stagioni a caso dalla storia
        p_profit = {n: float((rng.choice(seasons.values, (5000, n)).sum(axis=1) > 0).mean()) for n in (1, 3, 5)}
        streak = (~g.won).astype(int).groupby(g.won.cumsum()).sum().max()
        rows.append({"strategia": name, "schedine": len(g), "vinte": g.won.mean(), "prob_media": g.prob.mean(),
                     "quota_media": g.quota.mean(), "puntato": staked, "pnl": g.pnl.sum(),
                     "roi": g.pnl.sum() / staked, "roi_90_lo": lo, "roi_90_hi": hi,
                     "clv": g.ev_close.mean(), "stagioni_in_attivo": f"{(seasons > 0).sum()}/{len(seasons)}",
                     "max_drawdown": float((curve.cummax() - curve).max()), "serie_perse_max": int(streak),
                     **{f"p_attivo_{n}st": v for n, v in p_profit.items()}})
    return pd.DataFrame(rows)


def main():
    warnings.filterwarnings("ignore")
    t = timer()
    base = load_base()
    seasons = range(FIRST_SEASON, int(base.season.max()) + 1)
    base = base[base.season.isin(seasons)]
    legs = legs_table(base, model_probs(load_base(), seasons))
    print(f"gambe pronte: {len(legs)} ({t()})", flush=True)
    sim = simulate(legs)
    sim.to_pickle(RESEARCH_DIR / "doubles_sim.pkl")
    res = summarize(sim)
    res.to_csv(RESEARCH_DIR / "doubles_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print("\nROI per stagione:")
    print(sim.pivot_table(index="season", columns="strategia", values="pnl", aggfunc="sum").div(
        sim.pivot_table(index="season", columns="strategia", values="pnl", aggfunc="size") * STAKE).round(3).to_string())
    out = REPORTS_DIR / "simulazione-due-doppie.html"
    render_html(sim, res, out)
    print(f"report: {out} ({t()})")


if __name__ == "__main__":
    main()


COLORS = ["#2563eb", "#0891b2", "#16a34a", "#ca8a04", "#9333ea", "#dc2626", "#6b7280"]


def render_html(sim, res, path):
    """Report con la curva del P&L cumulato di ogni strategia e la tabella riassuntiva."""
    rounds = sorted(sim["round"].unique())
    x_of = {r: i for i, r in enumerate(rounds)}
    curves = {n: g.groupby("round").pnl.sum().cumsum() for n, g in sim.groupby("strategia", sort=False)}
    lo = min(c.min() for c in curves.values())
    hi = max(max(c.max() for c in curves.values()), 0)
    W, H, P = 900, 380, 40
    sx = lambda i: P + i * (W - 2 * P) / max(len(rounds) - 1, 1)
    sy = lambda v: P + (hi - v) * (H - 2 * P) / (hi - lo)
    lines = []
    for (name, c), color in zip(curves.items(), COLORS):
        pts = " ".join(f"{sx(x_of[r]):.1f},{sy(v):.1f}" for r, v in c.items())
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{pts}"><title>{name}</title></polyline>')
    years = []
    for season in sorted(sim.season.unique()):
        first = sim[sim.season == season]["round"].min()
        years.append(f'<line x1="{sx(x_of[first]):.1f}" x2="{sx(x_of[first]):.1f}" y1="{P}" y2="{H - P}" class="grid"/>'
                     f'<text x="{sx(x_of[first]) + 2:.1f}" y="{H - P + 14}" class="lab">{str(season)[2:]}/{str(season + 1)[2:]}</text>')
    ticks = [f'<line x1="{P}" x2="{W - P}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{"zero" if v == 0 else "grid"}"/>'
             f'<text x="4" y="{sy(v) + 4:.1f}" class="lab">{v:+.0f}€</text>'
             for v in np.linspace(np.floor(lo / 200) * 200, np.ceil(hi / 200) * 200, 7).round()]
    legend = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in zip(curves, COLORS))
    rows = "".join(
        f"<tr><td>{r.strategia}</td><td>{r.schedine}</td><td>{r.vinte:.0%}</td><td>{r.quota_media:.2f}</td>"
        f"<td class='{'pos' if r.pnl > 0 else 'neg'}'>{r.pnl:+.0f}€</td><td>{r.roi:+.1%}</td>"
        f"<td>{r.roi_90_lo:+.1%} / {r.roi_90_hi:+.1%}</td><td>{r.clv:+.1%}</td><td>{r.stagioni_in_attivo}</td>"
        f"<td>{r.max_drawdown:.0f}€</td><td>{r.serie_perse_max}</td><td>{r.p_attivo_5st:.0%}</td></tr>"
        for r in res.itertuples())
    html = f"""<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Due doppie a giornata</title><style>
:root{{--bg:#fafafa;--fg:#111;--mut:#6b7280;--line:#e5e7eb;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111318;--fg:#e5e7eb;--mut:#9ca3af;--line:#2a2f3a;--card:#181b22}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:960px;margin:auto}} h1{{font-size:22px;margin:0 0 4px}} p{{color:var(--mut);margin:4px 0 16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:16px;overflow-x:auto}}
svg{{width:100%;height:auto}} .grid{{stroke:var(--line)}} .zero{{stroke:var(--mut);stroke-dasharray:4 3}}
.lab{{fill:var(--mut);font-size:11px}} .legend span{{display:inline-flex;align-items:center;margin:2px 12px 2px 0;font-size:13px}}
.legend i{{width:12px;height:3px;margin-right:6px;display:inline-block}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
td:first-child,th:first-child{{text-align:left}} .pos{{color:#16a34a}} .neg{{color:#dc2626}}
</style></head><body><main>
<h1>Due doppie a giornata, 5 campionati, {str(sim.season.min())}/{str(sim.season.min() + 1)[2:]} – oggi</h1>
<p>5 € a schedina, 2 schedine per giornata (weekend e turni infrasettimanali), quote Snai stimate, solo quote pre-partita.
CLV = valore atteso secondo le quote di chiusura (dal 2019/20). "In attivo dopo 5 stagioni" = quota di simulazioni,
pescando stagioni a caso dalla storia, che finiscono sopra zero.</p>
<div class="card"><div class="legend">{legend}</div>
<svg viewBox="0 0 {W} {H}">{''.join(ticks)}{''.join(years)}{''.join(lines)}</svg></div>
<div class="card"><table><tr><th>Strategia</th><th>Schedine</th><th>Vinte</th><th>Quota media</th><th>P&amp;L</th><th>ROI</th>
<th>ROI 90%</th><th>CLV</th><th>Stagioni +</th><th>Max calo</th><th>Perse di fila</th><th>In attivo dopo 5 st.</th></tr>{rows}</table></div>
<p>"Modello (pesi stagioni)" usa una feature scelta guardando le stagioni 2014–2024: lì fa +3.1%, nelle stagioni non usate
per sceglierla −2.0% (2010–13) e −3.2% (2025–26).</p>
</main></body></html>"""
    path.parent.mkdir(exist_ok=True)
    path.write_text(html, encoding="utf-8")

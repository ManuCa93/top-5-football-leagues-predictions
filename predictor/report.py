"""Report settimanale in HTML (reports/AAAA-MM-GG.html)."""
import html
import json

import numpy as np
import pandas as pd

from .config import LEAGUES

MARKET_LABEL = {"1": "1", "X": "X", "2": "2", "1X": "1X", "X2": "X2", "GG": "Goal", "NG": "No Goal"}

CSS = """
:root { --bg:#f7f7f5; --card:#fff; --text:#1d1d1f; --muted:#6b6b70; --line:#e3e3e0; --accent:#0b6e4f;
        --neg:#b3261e; --pos:#0b6e4f; --chip:#eef3f0; }
@media (prefers-color-scheme: dark) { :root { --bg:#121314; --card:#1b1c1e; --text:#ececec; --muted:#9a9aa0;
        --line:#2c2d30; --accent:#4fc39a; --neg:#ff7a70; --pos:#4fc39a; --chip:#1f2a25; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:26px; margin:0 0 4px; } h2 { font-size:19px; margin:36px 0 12px; }
.muted { color:var(--muted); } .neg { color:var(--neg); } .pos { color:var(--pos); }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-top:16px; }
.kpi, .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
.kpi .v { font-size:22px; font-weight:650; font-variant-numeric:tabular-nums; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px; }
.card h3 { margin:0 0 8px; font-size:16px; }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { font-size:12px; text-transform:uppercase; letter-spacing:.03em; color:var(--muted); font-weight:600; }
.scroll { overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:12px; }
.chip { display:inline-block; background:var(--chip); border-radius:6px; padding:0 6px; font-weight:600; }
.note { background:var(--card); border-left:4px solid var(--accent); border-radius:8px; padding:12px 16px; }
.flag { color:var(--neg); font-weight:600; }
small { color:var(--muted); }
"""


GIORNI = ["lun", "mar", "mer", "gio", "ven", "sab", "dom"]


def when(d):
    """Data in ora italiana (i CSV di football-data.co.uk sono in ora UK, sempre un'ora indietro)."""
    d = d + pd.Timedelta(hours=1)
    return f"{GIORNI[d.weekday()]} {d:%d/%m %H:%M}"


def e(x):
    return html.escape(str(x))


def euro(x, sign=False):
    return f"{x:+.2f} €" if sign else f"{x:.2f} €"


def pct(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}%"


def num(x, fmt="{:.1f}"):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def bankroll_svg(points, width=640, height=140):
    if len(points) < 2:
        return ""
    xs = np.linspace(10, width - 10, len(points))
    lo, hi = min(points), max(points)
    span = (hi - lo) or 1
    ys = [height - 15 - (p - lo) / span * (height - 30) for p in points]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Andamento bankroll">'
            f'<polyline fill="none" stroke="var(--accent)" stroke-width="2.5" points="{path}"/>'
            f'<text x="10" y="12" font-size="11" fill="var(--muted)">max {hi:.0f} €</text>'
            f'<text x="10" y="{height - 2}" font-size="11" fill="var(--muted)">min {lo:.0f} €</text></svg>')


def slip_card(s):
    rows = "".join(
        f"<tr><td>{e(l['home_team'])} – {e(l['away_team'])}<br><small>{e(LEAGUES[l['league']]['name'])}, "
        f"{when(l['date'])}</small></td><td><span class='chip'>{MARKET_LABEL[l['market']]}</span></td>"
        f"<td>{l['q']:.2f}</td><td>{pct(l['p'])}</td><td>{1 / l['p']:.2f}</td></tr>"
        for l in s["legs"])
    fair = 1 / s["prob"]
    return f"""<div class="card"><h3>{e(s['type'])}</h3>
<table><tr><th>Partita</th><th>Esito</th><th>Quota Snai*</th><th>Prob.</th><th>Quota equa</th></tr>{rows}</table>
<p>Quota totale <b>{s['odds']:.2f}</b> · probabilità di vincere <b>{pct(s['prob'])}</b><br>
Puntata <b>{euro(s['stake'])}</b> → vincita {euro(s['stake'] * s['odds'])} · costo atteso
<span class="neg">{euro(s['stake'] * s['ev'], sign=True)}</span> ({s['ev']:+.1%})<br>
<small>Su Snai: se la quota totale è sotto {s['odds']:.2f} costa più del previsto; da {fair:.2f} in su è a valore positivo.</small></p></div>"""


def fixtures_table(fx):
    head = ("<tr><th>Partita</th><th>Prob. 1 / X / 2</th><th>Quota equa 1 / X / 2</th><th>Snai* 1 / X / 2</th>"
            "<th>Goal</th><th>Modello storico</th><th>Classifica</th><th>Forma</th><th>Gol fatti/subiti (ult. 5)</th>"
            "<th>xG stagione</th><th>Riposo</th><th>Precedenti</th></tr>")
    rows = []
    for r in fx.sort_values(["league", "date"]).itertuples():
        mod = [getattr(r, f"mod_{k}") for k in ("1", "X", "2")]
        mkt = [getattr(r, f"p_{k}") for k in ("1", "X", "2")]
        gap = max(abs(a - b) for a, b in zip(mod, mkt)) if not np.isnan(mod[0]) else 0
        mod_txt = " / ".join(pct(x) for x in mod) + (" <span class='flag'>≠ quote</span>" if gap > 0.08 else "")
        rows.append(
            f"<tr><td><b>{e(r.home_team)} – {e(r.away_team)}</b><br><small>{e(LEAGUES[r.league]['name'])}, "
            f"{when(r.date)}</small></td>"
            f"<td>{' / '.join(pct(x) for x in mkt)}</td>"
            f"<td>{' / '.join(f'{1 / x:.2f}' for x in mkt)}</td>"
            f"<td>{r.q_1:.2f} / {r.q_X:.2f} / {r.q_2:.2f}</td>"
            f"<td>{pct(r.p_GG)}<br><small>quota equa {1 / r.p_GG:.2f}</small></td>"
            f"<td>{mod_txt}</td>"
            f"<td>{num(r.h_pos, '{:.0f}')}° ({num(r.h_pt, '{:.0f}')} pt)<br>{num(r.a_pos, '{:.0f}')}° ({num(r.a_pt, '{:.0f}')} pt)</td>"
            f"<td>{e(r.h_form)}<br>{e(r.a_form)}</td>"
            f"<td>{num(r.h_gf5)} / {num(r.h_ga5)}<br>{num(r.a_gf5)} / {num(r.a_ga5)}</td>"
            f"<td>{num(r.h_xgf, '{:.2f}')} / {num(r.h_xga, '{:.2f}')}<br>{num(r.a_xgf, '{:.2f}')} / {num(r.a_xga, '{:.2f}')}</td>"
            f"<td>{num(r.h_rest, '{:.0f}')} gg<br>{num(r.a_rest, '{:.0f}')} gg</td>"
            f"<td><small>{e(r.h2h) or '–'}</small></td></tr>")
    return f"<div class='scroll'><table>{head}{''.join(rows)}</table></div>"


def closed_table(ledger):
    done = ledger[ledger.stato != "aperta"].tail(15).iloc[::-1]
    if done.empty:
        return "<p class='muted'>Nessuna schedina chiusa per ora.</p>"
    rows = []
    for r in done.itertuples():
        legs = json.loads(r.gambe)
        desc = "<br>".join(f"{e(l['home'])} – {e(l['away'])} <span class='chip'>{MARKET_LABEL[l['market']]}</span>" for l in legs)
        cls = "pos" if r.pnl > 0 else "neg"
        played = "" if r.giocata == 1 else " <small>(non giocata)</small>"
        rows.append(f"<tr><td>{e(r.creata)}</td><td>{e(r.tipo)}{played}</td><td>{desc}</td><td>{r.quota:.2f}</td>"
                    f"<td>{euro(r.puntata)}</td><td>{e(r.stato)}</td><td class='{cls}'>{euro(r.pnl, sign=True)}</td></tr>")
    return ("<div class='scroll'><table><tr><th>Data</th><th>Tipo</th><th>Gambe</th><th>Quota</th><th>Puntata</th>"
            f"<th>Esito</th><th>P&L</th></tr>{''.join(rows)}</table></div>")


def render(run_date, fixtures, slips, ledger, bankroll, initial, weekly_budget, demo=False):
    played = ledger[(ledger.stato != "aperta") & (ledger.giocata == 1)].sort_values("creata")
    curve = [initial] + list(initial + played.groupby("creata").pnl.sum().cumsum())
    staked = played.puntata.sum()
    roi = played.pnl.sum() / staked if staked else 0.0
    weekly_cost = sum(s["stake"] * s["ev"] for s in slips)
    no_odds = "" if len(fixtures) else (
        "<p class='note'>Le quote per le prossime partite non sono ancora pubblicate. "
        "football-data.co.uk le aggiorna di solito il venerdì per il weekend e il martedì per il turno infrasettimanale: rilancia allora.</p>")
    return f"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Schedine {run_date:%d/%m/%Y}</title>
<style>{CSS}</style></head><body><main>
<h1>Schedine della settimana{' (prova su dati passati)' if demo else ''}</h1>
<div class="muted">Generato il {GIORNI[run_date.weekday()]} {run_date:%d/%m/%Y %H:%M} · probabilità dal mercato senza margine · quote Snai stimate</div>
<div class="kpis">
 <div class="kpi"><div class="muted">Bankroll</div><div class="v">{euro(bankroll)}</div><small>partito da {euro(initial)}</small></div>
 <div class="kpi"><div class="muted">Budget questa settimana</div><div class="v">{euro(sum(s['stake'] for s in slips))}</div><small>tetto {euro(weekly_budget)}</small></div>
 <div class="kpi"><div class="muted">Costo atteso settimana</div><div class="v neg">{euro(weekly_cost, sign=True)}</div><small>media nel lungo periodo</small></div>
 <div class="kpi"><div class="muted">Risultato finora</div><div class="v {'pos' if roi >= 0 else 'neg'}">{roi:+.1%}</div><small>su {euro(staked)} giocati</small></div>
</div>
{bankroll_svg(curve)}
{no_odds}
<h2>Schedine proposte</h2>
<div class="cards">{''.join(slip_card(s) for s in slips) or "<p class='muted'>Nessuna schedina: mancano partite con quote.</p>"}</div>
<p><small>* Quota Snai stimata dalla media dei bookmaker (1X2 ≈ Snai; doppia chance e Goal/No Goal ricostruite, errore tipico 2–3%).
Controlla sempre la quota vera su Snai prima di giocare.</small></p>
<div class="note"><b>Come leggere i numeri.</b> La probabilità è quella delle quote senza margine: nei test su 9.000 partite nessun modello
(storia delle squadre, gol, tiri) l'ha migliorata. Ogni schedina ha un costo atteso negativo, cioè il margine Snai: le doppie sui favoriti
costano meno, le quote alte di più. <b>Quote maggiorate:</b> se Snai offre su un esito una quota più alta della "quota equa" in tabella,
quella giocata è a valore positivo: è l'unico caso in cui il vantaggio è dalla tua parte.</div>
<h2>Schedine chiuse</h2>{closed_table(ledger)}
<h2>Partite e storia delle squadre</h2>
<p class="muted">Per ogni riga: sopra la squadra di casa, sotto quella in trasferta. "≠ quote" = il modello storico si discosta di oltre 8 punti dal mercato
(di solito ha ragione il mercato).</p>
{fixtures_table(fixtures) if len(fixtures) else ''}
</main></body></html>"""

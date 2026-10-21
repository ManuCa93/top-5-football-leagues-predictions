"""Feature da Sofascore, calcolate per ogni partita di football-data usando solo ciò che precede il calcio d'inizio:
- carico: riposo e partite giocate in tutte le competizioni, Europa prima e dopo, forma in tutte le competizioni;
- assenti: infortunati nella partita di campionato precedente (già noti il venerdì), pesati per i minuti che
  quei giocatori avevano giocato. Gli assenti "al calcio d'inizio" dicono di più ma includono notizie
  arrivate dopo le quote del venerdì: servono solo come confronto (colonne *_now) e non si usano per decidere.
"""
import json
from collections import deque

import numpy as np
import pandas as pd

from .sofascore import INJURY_REASON, LINEUPS_DIR, UNIQUE_TOURNAMENT, load_league_events, load_team_events

EUROPE = {7, 679, 17015, 465}  # Champions, Europa League, Conference League, Supercoppa europea
LEAGUE_IDS = set(UNIQUE_TOURNAMENT.values())
SUSPENSION_REASON = 3
HOUR = pd.Timedelta(hours=1)


def team_names(base, events):
    """id Sofascore -> nome football-data: le coppie che giocano lo stesso giorno nella stessa lega."""
    ev = events.assign(day=events.date.dt.normalize())
    b = base[["league", "date", "home_team", "away_team"]].assign(day=base.date.dt.normalize())
    j = ev.merge(b, on=["league", "day"])
    pairs = pd.concat([j[["home_id", "home_team"]].set_axis(["sid", "name"], axis=1),
                       j[["away_id", "away_team"]].set_axis(["sid", "name"], axis=1)])
    counts = pairs.value_counts().reset_index()
    best = counts.sort_values("count", ascending=False).drop_duplicates("sid")
    return dict(zip(best.sid, best.name))


def link_events(base, events, names):
    """Evento Sofascore di campionato -> match_id (stesse squadre, data entro 3 giorni)."""
    ev = events.assign(home_team=events.home_id.map(names), away_team=events.away_id.map(names))
    j = ev.merge(base[["match_id", "league", "season", "home_team", "away_team", "date"]],
                 on=["league", "season", "home_team", "away_team"], suffixes=("_sofa", ""))
    j = j[(j.date_sofa - j.date).abs() <= pd.Timedelta(days=3)]
    return j[["id", "match_id", "home_id", "away_id"]].drop_duplicates("match_id")


def _team_side(base, link):
    """Una riga per (partita, squadra) con id Sofascore e orario."""
    m = base[["match_id", "date"]].merge(link, on="match_id")
    h = m.assign(sid=m.home_id, is_home=1)
    a = m.assign(sid=m.away_id, is_home=0)
    return pd.concat([h, a], ignore_index=True)[["match_id", "id", "date", "sid", "is_home"]]


def _to_match(side, values, name):
    s = side[["match_id", "is_home"]].assign(v=np.asarray(values, dtype=float))
    h = s[s.is_home == 1].set_index("match_id").v
    a = s[s.is_home == 0].set_index("match_id").v
    return pd.DataFrame({f"{name}_h": h, f"{name}_a": a, f"{name}_diff": h - a})


def schedule_features(side, team_events, form_days=(14, 21, 30)):
    """Riposo, partite negli ultimi 7/14 giorni, Europa nei 4 giorni prima/dopo, punti per partita
    in tutte le competizioni negli ultimi N giorni."""
    te = team_events[team_events.status.isin(["finished", "notstarted"])]
    rows = []
    for sid, g in side.groupby("sid"):
        mine = te[(te.home_id == sid) | (te.away_id == sid)].sort_values("date")
        dates = mine.date.values
        done = (mine.status == "finished").values
        europe = mine.tournament.isin(EUROPE).values
        cup = (~mine.tournament.isin(LEAGUE_IDS | EUROPE)).values
        home = (mine.home_id == sid).values
        gf = np.where(home, mine.home_score, mine.away_score).astype(float)
        ga = np.where(home, mine.away_score, mine.home_score).astype(float)
        pts = np.select([gf > ga, gf == ga], [3.0, 1.0], 0.0)
        for r in g.itertuples():
            k = r.date.to_datetime64()
            before = (dates < k - 2 * HOUR) & done
            after = (dates > k + 2 * HOUR)
            out = {"idx": r.Index}
            last = dates[before].max() if before.any() else None
            out["rest_all"] = min((k - last) / np.timedelta64(1, "D"), 14) if last is not None else np.nan
            for d in (7, 14):
                out[f"n{d}_all"] = int((before & (dates >= k - np.timedelta64(d, "D"))).sum())
            win_prev = before & (dates >= k - np.timedelta64(4, "D"))
            win_next = after & (dates <= k + np.timedelta64(4, "D"))
            out["euro_prev"] = float((win_prev & europe).any())
            out["euro_next"] = float((win_next & europe).any())
            out["cup_next"] = float((win_next & cup).any())
            for d in form_days:
                w = before & (dates >= k - np.timedelta64(d, "D"))
                out[f"form{d}_all"] = pts[w].mean() if w.any() else np.nan
            rows.append(out)
    f = pd.DataFrame(rows).set_index("idx").reindex(side.index)
    return pd.concat([_to_match(side, f[c], c) for c in f.columns], axis=1)


def _lineup_table(link_all):
    """Per ogni evento di campionato con formazione: minuti per giocatore e assenti, lato casa e trasferta."""
    rows = []
    for r in link_all.itertuples():
        path = LINEUPS_DIR / f"{r.id}.json"
        if not path.exists():
            continue
        lu = json.loads(path.read_text())
        for side, sid in (("home", r.home_id), ("away", r.away_id)):
            s = lu[side]
            rows.append({"id": r.id, "sid": sid, "date": r.date,
                         "minutes": {p["id"]: p["min"] or 0 for p in s["players"]},
                         "injured": [m["id"] for m in s["missing"] if m["type"] == "missing" and m["reason"] == INJURY_REASON],
                         "suspended": [m["id"] for m in s["missing"] if m["reason"] == SUSPENSION_REASON],
                         "doubtful": [m["id"] for m in s["missing"] if m["type"] == "doubtful"]})
    return pd.DataFrame(rows)


def absence_features(side, lineups, lookbacks=(5, 10, 20)):
    """Quota di minuti persa per assenti: share = minuti del giocatore nelle ultime N partite di campionato
    / (90 x N). Somma sugli assenti. inj_prev = infortunati nella partita precedente (noti il venerdì),
    susp = squalificati (noti dopo la partita precedente), *_now = assenti al calcio d'inizio (solo confronto)."""
    out = {}
    for sid, g in lineups.sort_values("date").groupby("sid"):
        history = deque(maxlen=max(lookbacks))
        prev_injured = []
        for r in g.itertuples():
            feats = {}
            for lb in lookbacks:
                recent = list(history)[-lb:]
                tot = {}
                for mins in recent:
                    for p, m in mins.items():
                        tot[p] = tot.get(p, 0) + m
                denom = 90.0 * max(len(recent), 1)
                share = lambda ids: sum(tot.get(p, 0) for p in ids) / denom if recent else np.nan
                feats[f"inj_prev{lb}"] = share(prev_injured)
                feats[f"susp{lb}"] = share(r.suspended)
                feats[f"inj_now{lb}"] = share(r.injured)
                feats[f"dbt_now{lb}"] = share(r.doubtful)
            out[(r.id, sid)] = feats
            history.append(r.minutes)
            prev_injured = r.injured
    f = pd.DataFrame([out.get((r.id, r.sid), {}) for r in side.itertuples()], index=side.index)
    return pd.concat([_to_match(side, f[c], c) for c in f.columns], axis=1)


def build(base):
    """Tutte le feature Sofascore per le partite di base (indice: match_id). NaN dove mancano i dati."""
    events = load_league_events()
    names = team_names(base, events)
    link = link_events(base, events, names)
    side = _team_side(base, link)
    sched = schedule_features(side, load_team_events())
    lineups = _lineup_table(base[["match_id", "date"]].merge(link, on="match_id"))
    absn = absence_features(side, lineups)
    feats = pd.concat([sched, absn], axis=1)
    feats.attrs["coverage"] = len(link) / base[base.season >= 2019].shape[0]
    return feats

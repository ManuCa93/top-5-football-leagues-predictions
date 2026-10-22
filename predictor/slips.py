"""Costruzione delle schedine.

Ogni gamba ha probabilità vera p (dal mercato senza margine) e quota Snai q.
Il valore atteso di una schedina con gambe su partite diverse è prod(p*q) - 1: con quote Snai
è quasi sempre negativo, quindi "schedina decente" = la combinazione che perde meno in media,
dentro i vincoli di probabilità e di quota del suo tipo.
"""
import itertools

import numpy as np

MARKETS = ("1", "X", "2", "1X", "X2", "GG", "NG")

# nome, gambe, prob. minima di ogni gamba, intervallo quota totale, quota del budget settimanale
SLIP_TYPES = [
    {"name": "Doppia solida", "legs": 2, "min_leg_prob": 0.62, "odds": (1.6, 2.6), "budget_share": 0.5},
    {"name": "Tripla", "legs": 3, "min_leg_prob": 0.50, "odds": (3.0, 6.0), "budget_share": 0.3},
    {"name": "Quota alta", "legs": 4, "min_leg_prob": 0.40, "odds": (8.0, 25.0), "budget_share": 0.2},
]
# "2 doppie a giornata": la strategia che perde meno nella simulazione 2010-2026 (predictor/doubles.py)
DUE_DOPPIE = [
    {"name": "Doppia 1", "legs": 2, "min_leg_prob": 0.0, "odds": (1.6, 2.6), "budget_share": 0.5},
    {"name": "Doppia 2", "legs": 2, "min_leg_prob": 0.0, "odds": (1.6, 2.6), "budget_share": 0.5},
]
MIN_LEG_ODDS = 1.20  # sotto questa quota la gamba aggiunge rischio quasi senza pagare
TOP_CANDIDATES = 30


def candidate_legs(matches):
    """matches: iterabile di dict con match_id, label, p_<mercato>, q_<mercato>."""
    legs = []
    for m in matches:
        for k in MARKETS:
            p, q = m.get(f"p_{k}"), m.get(f"q_{k}")
            if p is None or q is None or not np.isfinite(p) or not np.isfinite(q) or q < MIN_LEG_ODDS:
                continue
            legs.append({"match_id": m["match_id"], "label": m["label"], "market": k,
                         "p": float(p), "q": float(q), "ev": float(p * q),
                         **{f: m[f] for f in ("league", "home_team", "away_team", "date") if f in m}})
    return legs


def best_slip(legs, spec, used):
    eligible = sorted((l for l in legs if l["p"] >= spec["min_leg_prob"] and l["match_id"] not in used),
                      key=lambda l: l["ev"], reverse=True)
    # per ogni partita tengo solo le due gambe migliori, poi le migliori in assoluto
    per_match, pool = {}, []
    for l in eligible:
        per_match[l["match_id"]] = per_match.get(l["match_id"], 0) + 1
        if per_match[l["match_id"]] <= 2:
            pool.append(l)
    pool = pool[:TOP_CANDIDATES]
    lo, hi = spec["odds"]
    best = None
    for combo in itertools.combinations(pool, spec["legs"]):
        if len({l["match_id"] for l in combo}) < spec["legs"]:
            continue
        q = float(np.prod([l["q"] for l in combo]))
        if not lo <= q <= hi:
            continue
        p = float(np.prod([l["p"] for l in combo]))
        key = (p * q, p)
        if best is None or key > best[0]:
            best = (key, combo, p, q)
    if best is None:
        return None
    _, combo, p, q = best
    return {"type": spec["name"], "legs": list(combo), "prob": p, "odds": q, "ev": p * q - 1}


def build_slips(legs, weekly_budget, min_stake=1.0, slip_types=SLIP_TYPES):
    used, slips = set(), []
    for spec in slip_types:
        slip = best_slip(legs, spec, used)
        if slip is None:
            continue
        stake = max(min_stake, round(weekly_budget * spec["budget_share"] * 2) / 2)
        slip["stake"] = stake
        slip["expected_return"] = stake * (slip["ev"])
        used.update(l["match_id"] for l in slip["legs"])
        slips.append(slip)
    return slips

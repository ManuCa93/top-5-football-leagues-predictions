"""Test anti-leakage: le feature di una partita non devono cambiare se tolgo i dati futuri.

Esegui con: python -m pytest tests  (oppure python tests/test_leakage.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from predictor.data import load_matches  # noqa: E402
from predictor.features import match_features  # noqa: E402
from predictor.model import GoalModel  # noqa: E402

CUTOFF = pd.Timestamp("2025-01-10 12:00")


def _matches():
    m = load_matches(seasons=range(2022, 2026), refresh=False)
    return m


def test_features_do_not_see_future():
    m = _matches()
    full = match_features(m).set_index("match_id")
    cut = match_features(m[m.date < CUTOFF]).set_index("match_id")
    common = full.loc[cut.index, cut.columns]
    pd.testing.assert_frame_equal(common.sort_index(), cut.sort_index(), check_exact=False, rtol=1e-9)


def test_features_do_not_see_own_result():
    m = _matches()
    flipped = m.copy()
    last = flipped.date.max()
    mask = flipped.date == last
    flipped.loc[mask, ["home_goals", "away_goals"]] = flipped.loc[mask, ["away_goals", "home_goals"]].values + 5
    a = match_features(m).set_index("match_id")
    b = match_features(flipped).set_index("match_id")
    ids = m.loc[mask, "match_id"]
    pd.testing.assert_frame_equal(a.loc[ids], b.loc[ids])


def test_goal_model_rejects_future_rows():
    m = _matches()
    try:
        GoalModel().fit(m, CUTOFF)
    except AssertionError:
        return
    raise AssertionError("GoalModel ha accettato partite successive al cutoff")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)

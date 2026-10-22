"""Le feature della grid non devono cambiare se si tolgono le partite future.
Esegui: python tests/test_research_leakage.py"""
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from predictor import research as R  # noqa: E402

CUTOFF = datetime(2020, 2, 1)


def _families(base):
    long = R.team_long(base, with_advanced=True)
    return pd.concat([
        R.season_weighted(long, "uxgd", 10, 3), R.ewm_strength(long, "xgf", 10), R.form_last_n(long, "uppda", 5),
        R.season_weighted(long, "pts", 10, 3), R.season_weighted(long, "res", 5, 0, venue=True),
        R.ewm_strength(long, "sotd", 10), R.form_last_n(long, "gd", 5), R.form_last_days(long, "pts", 21),
        R.rest_league(long), R.elo(base, 20, 70, 0.9),
    ], axis=1)


def test_features_ignore_future_matches():
    warnings.filterwarnings("ignore")
    base = R.load_base()
    base = base[(base.season >= 2017) & (base.season <= 2020)].reset_index(drop=True)
    full = _families(base)
    cut = _families(base[base.date < CUTOFF].reset_index(drop=True))
    ids = base.match_id[base.date < CUTOFF]
    a, b = full.loc[ids].sort_index(), cut.loc[ids].sort_index()
    assert np.allclose(a.values, b.values, equal_nan=True), (a - b).abs().max().sort_values().tail()


def test_features_ignore_own_result():
    warnings.filterwarnings("ignore")
    base = R.load_base()
    base = base[(base.season >= 2018) & (base.season <= 2019)].reset_index(drop=True)
    target = base.index[base.date >= CUTOFF][0]
    flipped = base.copy()
    for c in ("home_goals", "away_goals", "home_sot", "away_sot", "home_shots", "away_shots", "home_corners",
              "away_corners"):
        flipped.loc[target, c] = flipped.loc[target, c] + 5
    mid = base.match_id[target]
    assert np.allclose(_families(base).loc[mid].values, _families(flipped).loc[mid].values, equal_nan=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)

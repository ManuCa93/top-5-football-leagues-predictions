"""Estrae dalla storia git tutte le quote Snai inserite a mano in get_odds_mapping()
e le salva in data/snai_history.csv (servono a tarare le quote GG/NG e doppia chance)."""
import ast
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from predictor.config import DATA_DIR  # noqa: E402


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def odds_dicts(source):
    """Tutti i dict letterali restituiti da get_odds_mapping() in un sorgente."""
    start = source.find("def get_odds_mapping")
    if start < 0:
        return None
    body = source[start:]
    ret = body.find("return {")
    if ret < 0:
        return None
    body = body[ret + len("return "):]
    depth = 0
    for i, c in enumerate(body):
        depth += c == "{"
        depth -= c == "}"
        if depth == 0:
            break
    try:
        return ast.literal_eval(body[: i + 1])
    except (ValueError, SyntaxError):
        return None


def main():
    rows = []
    for line in git("log", "--format=%h %cs").splitlines():
        commit, date = line.split()
        for path in git("ls-tree", "-r", "--name-only", commit).splitlines():
            if not path.endswith(".py"):
                continue
            mapping = odds_dicts(git("show", f"{commit}:{path}"))
            if not mapping:
                continue
            for league, matches in mapping.items():
                for m in matches:
                    if "home" in m and "away" in m:
                        rows.append({"commit_date": date, "league": league, **m})
    df = pd.DataFrame(rows)
    df = df.sort_values("commit_date").drop_duplicates(["league", "home", "away", "1", "X", "2"])
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "snai_history.csv", index=False)
    print(f"{len(df)} righe, {df[['league', 'home', 'away']].drop_duplicates().shape[0]} partite")


if __name__ == "__main__":
    main()

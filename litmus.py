"""Litmus test: hide a player's N most-played games, recommend from the rest of the library,
and see whether the hidden favourites come back.

  python litmus.py <steamid or profile URL>          # fetches once, then reuses libraries/<id>.json
  python litmus.py --library libraries/<id>.json --holdout 10

Reports, for each hidden game, its rank among all recommendable games (or why it can't be
recommended), hits in the top 15/50/100, and a "most popular games" baseline for comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from features import load_features
from recommend import LIBRARIES, RecConfig, exclusion_reasons, get_library, recommend


def popularity_ranking(f, owned: set[int], cfg: RecConfig) -> pd.DataFrame:
    """Baseline: rank eligible games by review volume x review score, ignoring taste."""
    ok = exclusion_reasons(f, owned, cfg).eq("").to_numpy()
    g = f.games[ok]
    pop = g.assign(score=g.log_reviews * g.review_score).sort_values("score", ascending=False)
    return pop.assign(rank=range(1, len(pop) + 1))[["appid", "rank"]]


def run(f, library: dict[int, int], holdout: int, cfg: RecConfig, label: str) -> pd.DataFrame:
    by_time = sorted(library.items(), key=lambda kv: kv[1], reverse=True)
    hidden = dict(by_time[:holdout])
    visible = dict(by_time[holdout:])          # hidden games are neither owned nor used for taste

    ranking = recommend(f, visible, cfg, full=True)   # final list order, family cap applied
    reasons = exclusion_reasons(f, set(visible), cfg)
    pop = popularity_ranking(f, set(visible), cfg)

    rows = []
    for appid, minutes in hidden.items():
        row = {"game": None, "hours": round(minutes / 60), "rank": None,
               "popularity_rank": None, "note": ""}
        if appid not in f.row_of:
            row.update(game=f"appid {appid}", note="not in catalog")
            rows.append(row)
            continue
        r = f.row_of[appid]
        row["game"] = f.games.name.iat[r]
        if reasons.iat[r]:
            row["note"] = f"excluded: {reasons.iat[r]}"
        else:
            # Its own slot, or its family's slot if another edition/sequel took the family's place.
            fam = ranking[ranking.family_id == f.games.family_id.iat[r]]
            row["rank"] = int(fam["rank"].iat[0])
            if fam.appid.iat[0] != appid:
                row["note"] = f"as {fam['name'].iat[0]}"
            row["popularity_rank"] = int(pop.loc[pop.appid == appid, "rank"].iat[0])
        rows.append(row)
    res = pd.DataFrame(rows)

    testable = res["rank"].notna()
    print(f"\n=== {label} ===")
    with pd.option_context("display.width", 200, "display.max_colwidth", 50):
        print(res.to_string(index=False))
    print(f"\n{testable.sum()} of {holdout} hidden games can be recommended at all.")
    for k in (15, 50, 100):
        model = int(res.loc[res["rank"] <= k, "rank"].nunique())   # one family slot = one hit
        base = int((res.popularity_rank <= k).sum())
        print(f"  top {k:>3}: model finds {model}, popularity baseline finds {base}")
    print(f"  ({len(ranking):,} games are eligible in total)")
    top = recommend(f, visible, cfg)
    print("\nThe 15 recommendations from the visible library:")
    for i, x in top.iterrows():
        mark = "  <-- hidden favourite" if x.appid in hidden else ""
        print(f"  {i + 1:>2}. {x['name']}  (because you played {x.because}){mark}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("user", nargs="?")
    ap.add_argument("--library", type=Path)
    ap.add_argument("--holdout", type=int, default=10)
    args = ap.parse_args()
    if not args.library and args.user and args.user.isdigit():
        cached = LIBRARIES / f"{args.user}.json"
        args.library = cached if cached.exists() else None
    if not (args.user or args.library):
        ap.error("give a Steam user or --library")
    f = load_features()
    library = get_library(args.user, args.library)
    print(f"Library: {len(library)} games, {sum(m > 0 for m in library.values())} played.")
    base = RecConfig()
    for method in ("knn", "centroid"):
        run(f, library, args.holdout, replace(base, method=method), f"method = {method}")


if __name__ == "__main__":
    main()

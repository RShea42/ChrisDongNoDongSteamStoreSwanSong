"""Recommend games from the catalog for one Steam library.

  python recommend.py <steamid or profile URL>
  python recommend.py --library libraries/<steamid>.json      # a saved {appid: minutes} file

Taste comes from the most-played games ("liked") and the least-played games that were still
launched ("disliked"; unplayed games are ignored). Every catalog game is scored by tag similarity
to the liked games, minus similarity to the disliked ones, plus a little review quality.
All knobs are in RecConfig.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from features import Features, is_same_game, load_features

HERE = Path(__file__).parent
LIBRARIES = HERE / "libraries"


@dataclass
class RecConfig:
    n: int = 15
    # Taste profile
    liked_count: int = 15            # most-played games that define what the user likes
    disliked_count: int = 10         # least-played (but launched) games
    min_disliked_minutes: int = 5    # launched for less than this = probably a crash or a misclick
    max_disliked_minutes: int = 120  # played more than this = not really a dislike, even at the bottom
    playtime_measure: str = "raw"    # "raw" minutes, or "relative" to the game's median playtime
    # Scoring
    method: str = "knn"              # "knn": match any one liked game; "centroid": match the average
    knn_k: int = 3                   # knn: average of the k best matches among liked games
    dislike_weight: float = 0.5
    # Share of the score from quality = half review score, half review volume (log scale). Proven
    # games are a safer bet than near-identical obscure clones with the same tags.
    quality_weight: float = 0.3
    # Variety: picks take turns between the liked games they resemble, this many per liked game per
    # round, so one dense genre (e.g. deckbuilders) can't fill the whole list.
    max_per_anchor: int = 2
    # Filters
    min_reviews: int = 50
    allow_adult: bool = False
    allow_sequels_of_owned: bool = True   # own Hades -> Hades II may be recommended
    max_franchise_picks: int = 3          # ...but at most this many from series the user owns
    max_per_family: int = 1               # editions/sequels of one game in the final list


def taste(f: Features, library: dict[int, int], cfg: RecConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick the liked and disliked games from a library ({appid: minutes})."""
    played = pd.DataFrame({"appid": list(library), "minutes": list(library.values())})
    played = played[played.minutes > 0]
    played = played[played.appid.isin(f.row_of.keys())]
    played["row"] = played.appid.map(f.row_of)
    g = f.games.iloc[played.row]
    played = played[g.has_features.to_numpy()].copy()
    played["name"] = f.games.name.iloc[played.row].to_numpy()

    if cfg.playtime_measure == "relative":
        median = f.games.median_playtime_hours.iloc[played.row].to_numpy() * 60
        typical = np.nanmedian(f.games.median_playtime_hours) * 60
        played["measure"] = played.minutes / np.where(np.isnan(median), typical, np.maximum(median, 30))
    else:
        played["measure"] = played.minutes.astype(float)

    liked = played.nlargest(cfg.liked_count, "measure")
    rest = played.drop(liked.index)
    rest = rest[rest.minutes.between(cfg.min_disliked_minutes, cfg.max_disliked_minutes)]
    disliked = rest.nsmallest(cfg.disliked_count, "measure")
    for t in (liked, disliked):
        t["weight"] = np.log1p(t.minutes / 60) + 0.1
    return liked, disliked


def similarity(f: Features, rows: np.ndarray, ref: pd.DataFrame, cfg: RecConfig):
    """Similarity of each candidate row to a set of reference games. Returns (score, best match)."""
    if ref.empty:
        return np.zeros(len(rows)), np.full(len(rows), -1)
    S = (f.X[rows] @ f.X[ref.row.to_numpy()].T).toarray()          # candidates x reference games
    w = ref.weight.to_numpy() / ref.weight.max()
    best = S.argmax(axis=1)
    if cfg.method == "centroid":
        profile = (f.X[ref.row.to_numpy()].T @ w).ravel()
        profile /= np.linalg.norm(profile) or 1
        return np.asarray(f.X[rows] @ profile).ravel(), best
    Sw = S * w
    k = min(cfg.knn_k, Sw.shape[1])
    return np.sort(Sw, axis=1)[:, -k:].mean(axis=1), best


def exclusion_reasons(f: Features, owned: set[int], cfg: RecConfig) -> pd.Series:
    """Why each catalog game can't be recommended ('' = it can)."""
    g = f.games
    reason = pd.Series("", index=g.index, dtype=object)
    owned_rows = g.appid.isin(owned)
    owned_families = set(g.family_id[owned_rows])
    in_family = g.family_id.isin(owned_families) & ~owned_rows
    same_game = pd.Series(False, index=g.index)
    for i in g.index[in_family]:
        same_game.at[i] = any(is_same_game(g.loc[i], g.loc[j])
                              for j in g.index[owned_rows & (g.family_id == g.family_id.at[i])])
    sequel = in_family & ~same_game
    checks = [  # first matching reason wins
        (~g.has_features, "no tags or genres"),
        (owned_rows, "already owned"),
        (same_game, "edition/expansion of an owned game"),
        (sequel & (not cfg.allow_sequels_of_owned), "sequel of an owned game"),
        (~g.is_released, "not released"),
        (g.is_software, "software, not a game"),
        (g.is_non_game, "demo/soundtrack/playtest/benchmark"),
        (g.is_duplicate_listing, "duplicate listing (usually a DLC)"),
        (g.is_adult & (not cfg.allow_adult), "adult content"),
        (g.total_reviews < cfg.min_reviews, f"fewer than {cfg.min_reviews} reviews"),
    ]
    for mask, why in reversed(checks):
        reason[mask.to_numpy()] = why
    return reason


def recommend(f: Features, library: dict[int, int], cfg: RecConfig | None = None,
              full: bool = False) -> pd.DataFrame:
    """Top cfg.n games, or with full=True every recommendable game in final list order."""
    cfg = cfg or RecConfig()
    liked, disliked = taste(f, library, cfg)
    if liked.empty:
        raise ValueError("None of this library's played games are in the catalog.")
    reasons = exclusion_reasons(f, set(library), cfg)
    rows = np.flatnonzero(reasons.eq("").to_numpy())

    like, best = similarity(f, rows, liked, cfg)
    dislike, _ = similarity(f, rows, disliked, cfg)
    g = f.games.iloc[rows]
    out = pd.DataFrame({
        "appid": g.appid.to_numpy(), "name": g.name.to_numpy(),
        "like": like, "dislike": dislike, "review_score": g.review_score.to_numpy(),
        "total_reviews": g.total_reviews.to_numpy(), "family_id": g.family_id.to_numpy(),
        "because": liked.name.to_numpy()[best],
    })
    volume = g.log_reviews.to_numpy() / max(g.log_reviews.max(), 1)
    out["quality"] = 0.5 * out.review_score + 0.5 * volume
    sim = out.like - cfg.dislike_weight * out.dislike
    out["score"] = (1 - cfg.quality_weight) * sim + cfg.quality_weight * out.quality

    # Expansions are never recommended themselves; the base game inherits the expansion's score
    # if that is higher (expansions without a recommendable base are simply dropped).
    is_exp = g.is_expansion.to_numpy()
    if is_exp.any():
        to_base = out[is_exp].assign(base=g.base_appid.to_numpy()[is_exp]).groupby("base").score.max()
        out = out[~is_exp]
        out["score"] = np.fmax(out.score, out.appid.map(to_base).to_numpy())
    out = out.sort_values("score", ascending=False, ignore_index=True)

    # Family cap: keep the best-scoring editions/sequels of each game.
    out = out[out.groupby("family_id").cumcount() < cfg.max_per_family]
    # Variety: round-robin across the liked games each pick resembles.
    out = out.assign(_round=out.groupby("because").cumcount() // cfg.max_per_anchor)
    out = out.sort_values(["_round", "score"], ascending=[True, False], ignore_index=True)
    # Franchise cap: past the first few picks from series the user already owns, the rest are moved
    # out of the top n (to just after it, in score order, so the litmus top-50/100 stay meaningful).
    owned_families = set(f.games.family_id[f.games.appid.isin(library)])
    franchise = out.family_id.isin(owned_families)
    over = franchise & (franchise.cumsum() > cfg.max_franchise_picks)
    kept = out[~over]
    out = pd.concat([kept.iloc[:cfg.n], pd.concat([kept.iloc[cfg.n:], out[over]]).sort_index()],
                    ignore_index=True)
    out = out.drop(columns="_round").assign(rank=np.arange(1, len(out) + 1))
    return out if full else out.head(cfg.n)


def get_library(user: str | None, path: Path | None, save: bool = True) -> dict[int, int]:
    """Load a saved library, or fetch it from Steam and save a copy to libraries/."""
    if path:
        return {int(k): int(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    from steam_api import owned_games, resolve_steamid
    steamid = resolve_steamid(user)
    lib = owned_games(steamid)
    if save:
        LIBRARIES.mkdir(exist_ok=True)
        (LIBRARIES / f"{steamid}.json").write_text(json.dumps(lib), encoding="utf-8")
    return lib


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("user", nargs="?", help="SteamID64, profile URL or custom URL name")
    ap.add_argument("--library", type=Path, help="saved {appid: minutes} JSON instead of fetching")
    ap.add_argument("--method", choices=["knn", "centroid"], default="knn")
    args = ap.parse_args()
    if not (args.user or args.library):
        ap.error("give a Steam user or --library")
    f = load_features()
    recs = recommend(f, get_library(args.user, args.library), RecConfig(method=args.method))
    with pd.option_context("display.width", 200, "display.max_colwidth", 45):
        print(recs[["name", "score", "like", "dislike", "review_score", "because"]].round(3))


if __name__ == "__main__":
    sys.exit(main())

"""Steam Games EDA dashboard over the games_cleaned table in Supabase.

Run: streamlit run app.py   (SUPABASE_DB_URL in .streamlit/secrets.toml)
"""
from collections import Counter
from itertools import combinations

import altair as alt
import numpy as np
import pandas as pd
import psycopg2
import streamlit as st

st.set_page_config(page_title="Steam Games EDA", page_icon="🎮", layout="wide")

BLUE = "#3987e5"  # single series color; reads on light and dark backgrounds
COLUMNS = """appid, name, release_year, price, price_type, positive, negative, total_reviews,
             positive_review_pct, user_score, score_rank, metacritic_score, genres, tags,
             genre_count, tag_count, windows, mac, linux, peak_ccu, achievements, recommendations,
             average_playtime_forever, median_playtime_forever, dlc_count, required_age"""
CORR_COLUMNS = {
    "price": "Price", "total_reviews": "Total reviews", "positive_review_pct": "Positive review %",
    "recommendations": "Recommendations", "peak_ccu": "Peak CCU", "metacritic_score": "Metacritic score",
    "average_playtime_forever": "Avg playtime", "median_playtime_forever": "Median playtime",
    "achievements": "Achievements", "dlc_count": "DLC count", "required_age": "Required age",
    "genre_count": "Genre count", "tag_count": "Tag count", "release_year": "Release year",
}


@st.cache_data(ttl="1h", show_spinner="Loading games from Supabase...")
def load() -> pd.DataFrame:
    with psycopg2.connect(st.secrets["SUPABASE_DB_URL"]) as conn, conn.cursor() as cur:
        cur.execute(f"select {COLUMNS} from public.games_cleaned")
        return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])


def split_counts(s: pd.Series) -> pd.Series:
    """Counts of each comma-separated value ('Action,Indie' -> Action, Indie)."""
    items = s.dropna().str.split(",").explode().str.strip()
    return items[items.ne("")].value_counts()


def pair_counts(s: pd.Series, n: int = 20) -> pd.DataFrame:
    pairs = Counter()
    for text in s.dropna():
        pairs.update(combinations(sorted({x.strip() for x in text.split(",") if x.strip()}), 2))
    return pd.DataFrame([{"Pair": f"{a} + {b}", "Games": c} for (a, b), c in pairs.most_common(n)])


def hbar(counts: pd.Series, label: str, height: int = 420) -> alt.Chart:
    data = counts.rename_axis(label).reset_index(name="Games")
    return alt.Chart(data).mark_bar(color=BLUE, cornerRadiusEnd=4, height={"band": 0.8}).encode(
        x=alt.X("Games:Q", title="Number of games"),
        y=alt.Y(f"{label}:N", sort="-x", title=None),
        tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip("Games:Q", format=",")],
    ).properties(height=height)


def histogram(data: pd.DataFrame, field: str, title: str, step: float | None = None,
              maxbins: int = 50) -> alt.Chart:
    b = alt.Bin(step=step) if step else alt.Bin(maxbins=maxbins)
    return alt.Chart(data).mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X(f"{field}:Q", bin=b, title=title),
        y=alt.Y("count():Q", title="Number of games"),
        tooltip=[alt.Tooltip(f"{field}:Q", bin=b, title=title), alt.Tooltip("count():Q", title="Games", format=",")],
    ).properties(height=320)


df = load()

# ---- Filters ---------------------------------------------------------------
st.sidebar.header("Filters")
y0, y1 = int(df.release_year.min()), int(df.release_year.max())
years = st.sidebar.slider("Release year", y0, y1, (y0, y1))
price_types = st.sidebar.multiselect("Price", ["Free", "Paid"], default=["Free", "Paid"])
all_genres = split_counts(df.genres).index.tolist()
genre_pick = st.sidebar.multiselect("Genres (any of)", all_genres)
min_reviews = st.sidebar.number_input("Minimum total reviews", 0, step=10)

f = df[df.release_year.between(*years) & df.price_type.isin(price_types) & (df.total_reviews >= min_reviews)]
if genre_pick:
    f = f[f.genres.fillna("").str.split(",").apply(lambda g: any(x.strip() in genre_pick for x in g))]
st.sidebar.caption(f"{len(f):,} of {len(df):,} games match.")

st.title("🎮 Steam Games EDA")
st.caption("Exploratory analysis of ~126k Steam games, read live from the `games_cleaned` table in Supabase.")
if f.empty:
    st.warning("No games match these filters.")
    st.stop()

overview, reviews, genres_tags, correlations, quality, explore = st.tabs(
    ["Overview", "Reviews", "Genres & Tags", "Correlations", "Data quality", "Explore"])

# ---- Overview --------------------------------------------------------------
with overview:
    c = st.columns(4)
    c[0].metric("Games", f"{len(f):,}")
    c[1].metric("Free to play", f"{(f.price_type == 'Free').mean():.0%}")
    paid = f.loc[f.price_type == "Paid", "price"]
    c[2].metric("Median paid price", f"${paid.median():.2f}" if len(paid) else "-")
    c[3].metric("Have 10+ reviews", f"{(f.total_reviews >= 10).mean():.0%}")

    st.subheader("Games released per year")
    per_year = f.groupby("release_year").size().reset_index(name="Games")
    st.altair_chart(alt.Chart(per_year).mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("release_year:O", title="Release year"),
        y=alt.Y("Games:Q", title="Number of games"),
        tooltip=[alt.Tooltip("release_year:O", title="Year"), alt.Tooltip("Games:Q", format=",")],
    ).properties(height=320), width="stretch")

    st.subheader("Platform support")
    plat = pd.Series({"Windows": f.windows.mean(), "Mac": f.mac.mean(), "Linux": f.linux.mean()})
    st.altair_chart(alt.Chart(plat.rename_axis("Platform").reset_index(name="Share")).mark_bar(
        color=BLUE, cornerRadiusEnd=4, height={"band": 0.6}).encode(
        x=alt.X("Share:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="Share of games"),
        y=alt.Y("Platform:N", sort="-x", title=None),
        tooltip=["Platform", alt.Tooltip("Share:Q", format=".1%")],
    ).properties(height=160), width="stretch")

# ---- Reviews ---------------------------------------------------------------
with reviews:
    st.subheader("Review coverage")
    st.caption("Most Steam games have very few reviews, which limits review-based analysis.")
    cov = pd.DataFrame([{"Threshold": f"> {t:,} reviews", "Games": int((f.total_reviews > t).sum()),
                         "Share": (f.total_reviews > t).mean()} for t in [0, 10, 50, 100, 500, 1000]])
    st.dataframe(cov, hide_index=True, column_config={
        "Games": st.column_config.NumberColumn(format="localized"),
        "Share": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})

    left, right = st.columns(2)
    with left:
        st.markdown("**Distribution of review counts** (log scale)")
        logs = pd.DataFrame({"log(1 + total reviews)": np.log1p(f.total_reviews)})
        st.altair_chart(histogram(logs, "log(1 + total reviews)", "log(1 + total reviews)"),
                        width="stretch")
    with right:
        floor = st.slider("Only games with at least this many reviews", 1, 1000, 50, key="pct_floor")
        st.markdown(f"**Positive review %** (games with {floor:,}+ reviews)")
        pos = f.loc[f.total_reviews >= floor, ["positive_review_pct"]].dropna()
        st.altair_chart(histogram(pos, "positive_review_pct", "Positive review %", step=5),
                        width="stretch")
        st.caption(f"{len(pos):,} games · median {pos.positive_review_pct.median():.0f}% positive")

# ---- Genres & tags ---------------------------------------------------------
with genres_tags:
    left, right = st.columns(2)
    with left:
        st.subheader("15 most common genres")
        st.altair_chart(hbar(split_counts(f.genres).head(15), "Genre"), width="stretch")
    with right:
        st.subheader("20 most common tags")
        st.altair_chart(hbar(split_counts(f.tags).head(20), "Tag"), width="stretch")

    st.subheader("Genres vs. tags per game")
    st.caption("Games usually list 1-3 genres but many tags, so tags carry more detail for similarity.")
    left, right = st.columns(2)
    left.altair_chart(histogram(f, "genre_count", "Genres per game", step=1), width="stretch")
    right.altair_chart(histogram(f, "tag_count", "Tags per game", step=1), width="stretch")

    st.subheader("Most common pairings")
    left, right = st.columns(2)
    with left:
        st.markdown("**Genre pairs**")
        st.altair_chart(hbar(pair_counts(f.genres).set_index("Pair").Games, "Pair"), width="stretch")
    with right:
        st.markdown("**Tag pairs**")
        st.altair_chart(hbar(pair_counts(f.tags).set_index("Pair").Games, "Pair"), width="stretch")

# ---- Correlations ----------------------------------------------------------
with correlations:
    st.subheader("Correlation matrix")
    method = st.radio("Method", ["Spearman", "Pearson"], horizontal=True,
                      help="Spearman uses ranks, so the heavy skew in reviews, players and playtime doesn't "
                           "let a few huge games dominate. Pearson measures straight-line relationships.")
    st.caption("+1: both rise together · 0: no relationship · -1: one rises as the other falls. "
               "Zeros in Metacritic score are treated as missing.")
    num = f[list(CORR_COLUMNS)].astype(float).rename(columns=CORR_COLUMNS)
    num["Metacritic score"] = num["Metacritic score"].replace(0, np.nan)
    corr = num.corr(method=method.lower())
    order = list(corr.columns)
    long = corr.rename_axis("Variable 1").reset_index().melt("Variable 1", var_name="Variable 2", value_name="r")
    base = alt.Chart(long).encode(
        x=alt.X("Variable 2:N", sort=order, title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("Variable 1:N", sort=order, title=None),
    )
    cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color("r:Q", title="r", scale=alt.Scale(domain=[-1, 0, 1], range=["#e34948", "#b8b7b0", "#2a78d6"])),
        tooltip=["Variable 1", "Variable 2", alt.Tooltip("r:Q", format=".2f")],
    )
    labels = base.mark_text(fontSize=11).encode(
        text=alt.Text("r:Q", format=".2f"),
        color=alt.condition("abs(datum.r) > 0.55", alt.value("white"), alt.value("#0b0b0b")),
    )
    st.altair_chart((cells + labels).properties(height=560), width="stretch")

    pairs = long[long["Variable 1"] < long["Variable 2"]].dropna().copy()
    pairs["|r|"] = pairs.r.abs()
    st.markdown("**Strongest relationships**")
    st.dataframe(pairs.sort_values("|r|", ascending=False).head(10)[["Variable 1", "Variable 2", "r"]].round(2),
                 hide_index=True, width="stretch")

# ---- Data quality ----------------------------------------------------------
with quality:
    st.subheader("Missing genres and tags")
    g_miss, t_miss = f.genres.isna(), f.tags.isna()
    c = st.columns(4)
    c[0].metric("Missing genres", f"{g_miss.sum():,}")
    c[1].metric("Missing tags", f"{t_miss.sum():,}")
    c[2].metric("Missing both", f"{(g_miss & t_miss).sum():,}")
    c[3].metric("At least one", f"{(~(g_miss & t_miss)).sum():,}")

    st.subheader("Sparse columns")
    st.caption("Share of games where the value is zero or missing.")
    cols = ["user_score", "score_rank", "metacritic_score", "positive_review_pct", "price", "total_reviews"]
    sparse = pd.Series({col: (f[col].isna() | (f[col] == 0)).mean() for col in cols}).sort_values(ascending=False)
    st.altair_chart(alt.Chart(sparse.rename_axis("Column").reset_index(name="Share")).mark_bar(
        color=BLUE, cornerRadiusEnd=4, height={"band": 0.7}).encode(
        x=alt.X("Share:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="Zero or missing"),
        y=alt.Y("Column:N", sort="-x", title=None),
        tooltip=["Column", alt.Tooltip("Share:Q", format=".1%")],
    ).properties(height=240), width="stretch")

    st.subheader("User score vs. calculated positive %")
    scored = f[f.user_score > 0].copy()
    if scored.empty:
        st.info("No games with a user score match the current filters.")
    else:
        scored["calculated_pct"] = scored.positive / (scored.positive + scored.negative) * 100
        scored["difference"] = scored.user_score - scored.calculated_pct
        st.caption(f"Only {len(scored):,} games have a nonzero user score; it differs from the calculated "
                   f"positive % by {scored.difference.abs().mean():.1f} points on average, so the dashboard "
                   "uses positive % instead.")
        st.dataframe(scored[["name", "user_score", "calculated_pct", "difference", "score_rank"]].round(2),
                     hide_index=True, width="stretch")

# ---- Explore ---------------------------------------------------------------
with explore:
    search = st.text_input("Search by name")
    shown = f[f.name.str.contains(search, case=False, na=False, regex=False)] if search else f
    st.caption(f"{len(shown):,} games · sorted by total reviews")
    st.dataframe(
        shown.sort_values("total_reviews", ascending=False).head(1000)[
            ["name", "release_year", "price", "total_reviews", "positive_review_pct", "genres", "tags"]],
        hide_index=True, width="stretch", height=520,
        column_config={
            "name": "Name", "release_year": st.column_config.NumberColumn("Year", format="%d"),
            "price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "total_reviews": st.column_config.NumberColumn("Reviews", format="localized"),
            "positive_review_pct": st.column_config.NumberColumn("Positive %", format="%.0f%%"),
            "genres": "Genres", "tags": "Tags"})

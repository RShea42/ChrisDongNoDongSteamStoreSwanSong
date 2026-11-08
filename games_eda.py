import pandas as pd

df = pd.read_csv("games.csv")

print("Shape:", df.shape)
display(df.head())
display(df.sample(10, random_state=42))
df.info()
Code cell 2
Cell 2 of 31
for col in df.columns:
    print(col, df[col].dtype)
Code cell 3
Cell 3 of 31
missing = pd.DataFrame({
    "Missing": df.isna().sum(),
    "Missing %": df.isna().mean() * 100
})

display(missing.sort_values("Missing %", ascending=False))

print("Duplicate rows:", df.duplicated().sum())
print("Duplicate AppIDs:", df["AppID"].duplicated().sum())
Code cell 4
Cell 4 of 31
import csv

with open("games.csv", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    first_row = next(reader)

print("Header fields:", len(header))
print("First row fields:", len(first_row))

for i, (column, value) in enumerate(zip(header, first_row)):
    print(i, column, ":", str(value)[:60])
Code cell 5
Cell 5 of 31
display(df.iloc[:10, -6:])
Code cell 6
Cell 6 of 31
print(df.columns[-10:].tolist())

print(df["Movies"].head(10).tolist())
print(df["Screenshots"].head(10).tolist())
Code cell 7
Cell 7 of 31
import csv

with open("games.csv", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)

    header = next(reader)
    first_row = next(reader)

print("Header fields:", len(header))
print("First row fields:", len(first_row))

for i in range(max(0, len(header) - 6), len(header)):
    print(
        f"Position {i}:",
        f"Header = {header[i]!r}",
        f"Value = {first_row[i] if i < len(first_row) else 'MISSING'}"
    )
Code cell 8
Cell 8 of 31
import csv

with open("games.csv", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    row = next(reader)

for i in range(len(row)):
    column = header[i] if i < len(header) else "NO HEADER"
    value = str(row[i])[:65]

    print(f"{i:2} | {column:28} | {value}")
    
Code cell 9
Cell 9 of 31
import csv
import pandas as pd

with open("games.csv", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    headers = next(reader)

# Repair the merged column header
headers[7:8] = ["Discount", "DLC count"]

# Reload with the corrected 40-column schema
df = pd.read_csv(
    "games.csv",
    header=0,
    names=headers
)

print("Dataset shape:", df.shape)
print("Number of columns:", len(df.columns))

display(df.head())
Code cell 10
Cell 10 of 31
print(df[[
    "AppID",
    "Name",
    "Price",
    "Discount",
    "DLC count",
    "Screenshots",
    "Movies"
]].head())

print("\nMovies missing percentage:")
print(df["Movies"].isna().mean() * 100)

print("\nAppID is a regular column:")
print("AppID" in df.columns)
Code cell 11
Cell 11 of 31
import csv

with open("games.csv", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)

    total = 0
    populated = 0
    field_counts = {}

    for row in reader:
        total += 1
        field_counts[len(row)] = field_counts.get(len(row), 0) + 1

        if len(row) == 40 and row[39].strip():
            populated += 1

print("Total rows:", total)
print("Rows with populated Movies:", populated)
print("Field counts:", field_counts)
Code cell 12
Cell 12 of 31
df = df.drop(columns=["Movies"])

quality_report = pd.DataFrame({
    "Data Type": df.dtypes,
    "Missing Count": df.isna().sum(),
    "Missing %": (df.isna().mean() * 100).round(2),
    "Unique Values": df.nunique()
})

display(quality_report.sort_values("Missing %", ascending=False))
Code cell 13
Cell 13 of 31
quality_report = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.values,
    "Missing Count": df.isna().sum().values,
    "Missing %": (df.isna().mean() * 100).round(2).values,
    "Unique Values": df.nunique().values
})

quality_report = quality_report.sort_values(
    "Missing %",
    ascending=False
).reset_index(drop=True)

display(quality_report)
Code cell 14
Cell 14 of 31
numeric_cols = df.select_dtypes(include="number").columns

zero_report = pd.DataFrame({
    "Column": numeric_cols,
    "Zero Count": [(df[col] == 0).sum() for col in numeric_cols],
    "Zero %": [
        round((df[col] == 0).mean() * 100, 2)
        for col in numeric_cols
    ]
})

display(zero_report.sort_values("Zero %", ascending=False))
Code cell 15
Cell 15 of 31
# Inspect games with nonzero user scores
display(
    df.loc[
        df["User score"] > 0,
        [
            "AppID",
            "Name",
            "Metacritic score",
            "User score",
            "Score rank",
            "Positive",
            "Negative"
        ]
    ].head(20)
)

# Check whether Score rank and User score
# are populated for the same games
print(
    pd.crosstab(
        df["User score"] > 0,
        df["Score rank"].notna(),
        rownames=["Nonzero user score"],
        colnames=["Score rank present"]
    )
)
Code cell 16
Cell 16 of 31
scored_games = df.loc[df["User score"] > 0].copy()

scored_games["Calculated positive %"] = (
    scored_games["Positive"]
    / (scored_games["Positive"] + scored_games["Negative"])
    * 100
)

scored_games["Difference"] = (
    scored_games["User score"] - scored_games["Calculated positive %"]
)

display(
    scored_games[
        [
            "Name",
            "User score",
            "Calculated positive %",
            "Difference",
            "Score rank"
        ]
    ].round(2)
)

print(
    "Mean absolute difference:",
    scored_games["Difference"].abs().mean().round(2)
)
Code cell 17
Cell 17 of 31
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("AppID is a column:", "AppID" in df.columns)
print("Discount exists:", "Discount" in df.columns)
print("DLC count exists:", "DLC count" in df.columns)
print("Old merged column exists:", "DiscountDLC count" in df.columns)
print("Movies exists:", "Movies" in df.columns)
print("Duplicate AppIDs:", df["AppID"].duplicated().sum())
Code cell 18
Cell 18 of 31
# It works well enough. Let's actually start E'ing this D.
df["Total reviews"] = df["Positive"] + df["Negative"]

df["Positive review %"] = (
    df["Positive"] / df["Total reviews"] * 100
).where(df["Total reviews"] > 0)

# Measure review coverage
thresholds = [0, 10, 50, 100, 500, 1000]

for threshold in thresholds:
    count = (df["Total reviews"] > threshold).sum()
    percentage = count / len(df) * 100

    print(
        f"Games with > {threshold:,} reviews: "
        f"{count:,} ({percentage:.2f}%)"
    )
Code cell 19
Cell 19 of 31
#Charts are dope. Let's make some charts.
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 5))

plt.hist(
    np.log1p(df["Total reviews"]),
    bins=50,
    edgecolor="black"
)

plt.title("Distribution of Steam Game Review Counts")
plt.xlabel("log(1 + Total Reviews)")
plt.ylabel("Number of Games")

plt.show()
Code cell 20
Cell 20 of 31
# Who doesn't like a good top 15?
genres = (
    df["Genres"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)

genre_counts = genres.value_counts()

display(genre_counts.head(20))

genre_counts.head(15).sort_values().plot(
    kind="barh",
    figsize=(10, 6),
    title="15 Most Common Steam Game Genres"
)

plt.xlabel("Number of Games")
plt.ylabel("Genre")
plt.show()
Code cell 21
Cell 21 of 31
missing_genres = df["Genres"].isna()
missing_tags = df["Tags"].isna()

print("Missing genres:", missing_genres.sum())
print("Missing tags:", missing_tags.sum())

print(
    "Missing both:",
    (missing_genres & missing_tags).sum()
)

print(
    "At least one available:",
    (~(missing_genres & missing_tags)).sum()
)
Code cell 22
Cell 22 of 31
genre_available = df["Genres"].fillna("").str.strip().ne("")
tag_available = df["Tags"].fillna("").str.strip().ne("")
description_available = df["About the game"].fillna("").str.strip().ne("")

coverage = pd.DataFrame({
    "Genres available": genre_available,
    "Tags available": tag_available,
    "Description available": description_available
})

display(
    coverage.value_counts()
    .rename("Number of games")
    .reset_index()
)
Code cell 23
Cell 23 of 31
# Count genres per game
df["Genre count"] = (
    df["Genres"]
    .fillna("")
    .apply(lambda x: len([g for g in x.split(",") if g.strip()]))
)

# Count tags per game
df["Tag count"] = (
    df["Tags"]
    .fillna("")
    .apply(lambda x: len([t for t in x.split(",") if t.strip()]))
)

print("Genre counts:")
display(df["Genre count"].describe())

print("\nTag counts:")
display(df["Tag count"].describe())
Code cell 24
Cell 24 of 31
display(
    df.loc[
        df["Genre count"] >= 3,
        ["Name", "Genres", "Tags", "Genre count", "Tag count"]
    ].head(15)
)

display(
    df[["Genre count", "Tag count"]].describe()
)
Code cell 25
Cell 25 of 31
#Can we use genres AND tags? or is one better than the other? The peaks and valleys are looking pretty defining.

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df["Genre count"].hist(
    bins=range(0, int(df["Genre count"].max()) + 2),
    ax=axes[0]
)
axes[0].set_title("Genres per Game")
axes[0].set_xlabel("Number of Genres")
axes[0].set_ylabel("Number of Games")

df["Tag count"].hist(
    bins=range(0, int(df["Tag count"].max()) + 2),
    ax=axes[1]
)
axes[1].set_title("Tags per Game")
axes[1].set_xlabel("Number of Tags")

plt.tight_layout()
plt.show()
Code cell 26
Cell 26 of 31
tags = (
    df["Tags"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)

tag_counts = tags[tags.ne("")].value_counts()

display(tag_counts.head(25))
Code cell 27
Cell 27 of 31
tag_counts.head(20).sort_values().plot(
    kind="barh",
    figsize=(10, 7),
    title="20 Most Common Steam Tags"
)

plt.xlabel("Number of Games")
plt.ylabel("Tag")
plt.show()
Code cell 28
Cell 28 of 31
missing_all = df.loc[
    ~genre_available &
    ~tag_available &
    ~description_available
]

print("Games missing all three features:", len(missing_all))

display(
    missing_all[
        ["AppID", "Name", "Release date", "Positive", "Negative"]
    ].head(20)
)
Code cell 29
Cell 29 of 31
from collections import Counter
from itertools import combinations

genre_pairs = Counter()

for genre_string in df["Genres"].dropna():
    game_genres = sorted(set(
        genre.strip()
        for genre in genre_string.split(",")
        if genre.strip()
    ))

    genre_pairs.update(combinations(game_genres, 2))

genre_pair_counts = pd.DataFrame(
    [
        {"Genre 1": pair[0], "Genre 2": pair[1], "Count": count}
        for pair, count in genre_pairs.most_common(20)
    ]
)

display(genre_pair_counts)
Code cell 30
Cell 30 of 31
tag_pairs = Counter()

for tag_string in df["Tags"].dropna():
    game_tags = sorted(set(
        tag.strip()
        for tag in tag_string.split(",")
        if tag.strip()
    ))

    tag_pairs.update(combinations(game_tags, 2))

tag_pair_counts = pd.DataFrame(
    [
        {"Tag 1": pair[0], "Tag 2": pair[1], "Count": count}
        for pair, count in tag_pairs.most_common(20)
    ]
)

display(tag_pair_counts)
Code cell 31
Cell 31 of 31
tag_pair_counts["Pair"] = (
    tag_pair_counts["Tag 1"] + " + " + tag_pair_counts["Tag 2"]
)

tag_pair_counts.sort_values("Count").plot(
    x="Pair",
    y="Count",
    kind="barh",
    figsize=(12, 8),
    legend=False,
    title="20 Most Common Steam Tag Combinations"
)

plt.xlabel("Number of Games")
plt.ylabel("Tag Combination")
plt.show()
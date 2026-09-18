"""The tables and figures of the data section.

Reads the frozen corpus and the raw Letterboxd tables and writes the five
figures the report shows. Needs pandas and matplotlib only.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import regions

BLUE = "#0072B2"
GREY = "#cccccc"
# Okabe-Ito, one hue per region, and the same hues cycled for the language pie.
REGION_COLOR = {
    "Europe": "#0072B2", "North America": "#D55E00", "Asia": "#009E73",
    "South America": "#CC79A7", "Oceania": "#56B4E9", "Africa": "#E69F00",
    "Antarctica": "#999999",
}
WHEEL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00",
         "#8C6D31", "#7570B3", "#66A61E", "#A6761D", "#3B7EA1", "#B07AA1"]


def _shade(color: str, factor: float) -> tuple:
    """Lighten a hex colour towards white by factor in [0, 1]."""
    rgb = tuple(int(color[k:k + 2], 16) / 255 for k in (1, 3, 5))
    return tuple(c + (1 - c) * factor for c in rgb)


def _style(size: float):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": size,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "figure.dpi": 200,
    })


def _write(fig, name: str) -> str:
    os.makedirs(config.FIG_DIR, exist_ok=True)
    path = os.path.join(config.FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _hbar(labels, values, xlabel, name, figsize, size=13, label_size=11):
    _style(size)
    fig, ax = plt.subplots(figsize=figsize)
    y = range(len(labels))
    ax.barh(list(y), values, color=BLUE, height=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(values) * 1.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color=GREY, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for position, value in zip(y, values):
        ax.text(value + max(values) * 0.015, position, f"{value:.1f}",
                va="center", fontsize=label_size)
    fig.tight_layout()
    return _write(fig, name)


def _vbar(labels, values, xlabel, ylabel, name, figsize, size=13, label_size=11,
          tick_rotation=0, value_rotation=0, headroom=1.16):
    _style(size)
    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(labels))
    ax.bar(list(x), values, color=BLUE, width=0.62 if tick_rotation == 0 else 0.72)
    ax.set_xticks(list(x))
    if tick_rotation:
        ax.set_xticklabels(labels, rotation=tick_rotation, ha="right",
                           rotation_mode="anchor")
    else:
        ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * headroom)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color=GREY, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for position, value in zip(x, values):
        ax.text(position, value + max(values) * 0.02, f"{value:.1f}",
                ha="center", va="bottom", fontsize=label_size,
                rotation=value_rotation)
    fig.tight_layout()
    return _write(fig, name)


def corpus_ids() -> set:
    """Movie ids of the frozen corpus."""
    return set(pd.read_csv(config.ENCODED_GENRES_CSV, usecols=[config.MOVIE_ID])
               [config.MOVIE_ID])


def poster_size_table() -> pd.DataFrame:
    """Poster shapes by share of the corpus. Shapes under ten movies are pooled."""
    sizes = pd.read_csv(config.POSTER_SIZES_CSV, index_col=0)
    counts = sizes.groupby(config.SIZE).count()
    total = counts.sum().values
    counts["size_group"] = counts.index
    counts.loc[counts[config.MOVIE_ID] < 10, "size_group"] = "others"
    counts = counts.groupby("size_group").sum()
    counts["relative_percentage"] = counts[config.MOVIE_ID] / total * 100
    return counts.sort_values(config.MOVIE_ID, ascending=False)


def genre_shares() -> pd.Series:
    """Share of the corpus carrying each genre. A movie may carry several."""
    encoded = pd.read_csv(config.ENCODED_GENRES_CSV, index_col=0)
    one_hot = encoded.drop(columns=[config.GENRE_COUNT])
    shares = one_hot.sum() / len(encoded) * 100
    shares.index = [c[len(config.GENRE_PREFIX):] for c in shares.index]
    return shares.sort_values(ascending=False)


def genre_count_shares() -> pd.Series:
    """Share of the corpus by number of genres carried."""
    encoded = pd.read_csv(config.ENCODED_GENRES_CSV,
                          usecols=[config.GENRE_COUNT])
    return encoded[config.GENRE_COUNT].value_counts(normalize=True).sort_index() * 100


def region_country_shares(min_country_share: float = 1.0):
    """Shares of country entries by region, and by country within each region.

    Countries below min_country_share per cent of the corpus are pooled into an
    unlabelled remainder inside their own region, so the outer ring stays legible.
    """
    countries = pd.read_csv(config.COUNTRIES_CSV)
    countries = countries[countries[config.ID].isin(corpus_ids())].copy()
    countries["region"] = countries[config.COUNTRY].map(regions.REGION)
    countries = countries.dropna(subset=["region"])
    total = len(countries)

    by_region = countries["region"].value_counts() / total * 100
    by_region = by_region[by_region >= 0.1]
    inner, outer = [], []
    for region in by_region.index:
        share = countries[countries["region"] == region][config.COUNTRY] \
            .value_counts() / total * 100
        named = share[share >= min_country_share]
        rest = share.sum() - named.sum()
        inner.append((region, by_region[region]))
        outer.extend((region, name, value) for name, value in named.items())
        if rest > 0:
            outer.append((region, "", rest))
    return pd.Series(dict(inner)), pd.DataFrame(
        outer, columns=["region", "country", "share"])


def language_shares(min_share: float = 1.0) -> pd.Series:
    """Share of language entries by language, over the corpus.

    Both row types are pooled, as the original analysis pooled them, so a movie
    contributes one entry per language it lists. Languages below min_share per
    cent are collected into Others.
    """
    languages = pd.read_csv(config.LANGUAGES_CSV)
    languages = languages[languages[config.ID].isin(corpus_ids())
                          & languages["type"].isin(config.LANGUAGE_TYPES)]
    share = languages[config.LANGUAGE].value_counts(normalize=True) * 100
    named = share[share >= min_share]
    others = share.sum() - named.sum()
    if others > 0:
        named = pd.concat([named, pd.Series({"Others": others})])
    return named


def release_decades() -> pd.Series:
    """Movies of the corpus by decade of first release."""
    movies = pd.read_csv(config.MOVIES_CSV, usecols=[config.ID, config.DATE])
    movies = movies[movies[config.ID].isin(corpus_ids())]
    years = pd.to_numeric(movies[config.DATE], errors="coerce").dropna()
    return (years // 10 * 10).astype(int).value_counts().sort_index()


def _sunburst(inner: pd.Series, outer: pd.DataFrame, name: str,
              label_floor: float = 2.0) -> str:
    """Region disc inside, country ring outside, one hue per region.

    Regions below label_floor are merged into one unlabeled gray wedge so that
    the shares printed stay the corpus shares and nothing is labeled outside
    the circle.
    """
    _style(11)
    fig, ax = plt.subplots(figsize=(3.6, 3.6))

    kept = inner[inner >= label_floor]
    rest = float(inner.sum() - kept.sum())
    order = list(kept.index)
    inner_values = list(kept.values) + ([rest] if rest > 0 else [])
    inner_colors = [REGION_COLOR.get(r, GREY) for r in order]
    if rest > 0:
        inner_colors.append("#BBBBBB")

    outer = outer[outer["region"].isin(order)].set_index("region").loc[order] \
        .reset_index()
    outer_colors, ranks = [], {}
    for region in outer["region"]:
        ranks[region] = ranks.get(region, -1) + 1
        outer_colors.append(_shade(REGION_COLOR.get(region, GREY),
                                   min(0.25 + 0.11 * ranks[region], 0.82)))
    outer_values = list(outer["share"].values)
    outer_labels = list(outer["country"].values)
    if rest > 0:
        outer_values.append(rest)
        outer_labels.append("")
        outer_colors.append("#D5D5D5")

    inner_wedges, *_ = ax.pie(
        inner_values, radius=0.66, colors=inner_colors, startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 0.8})
    outer_wedges, *_ = ax.pie(
        outer_values, radius=1.0, colors=outer_colors, startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 0.8})

    for wedge, label, value in zip(inner_wedges, order, kept.values):
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        wide = value >= 10
        radius = 0.40 if wide else 0.50
        text = label if wide else label.replace(" ", "\n")
        ax.text(radius * np.cos(angle), radius * np.sin(angle),
                f"{text}\n{value:.1f}%", ha="center", va="center",
                fontsize=10 if wide else 7.5, color="white",
                linespacing=1.0)

    for wedge, label, value in zip(outer_wedges, outer_labels, outer_values):
        if not label or value < 3.0:
            continue
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        ax.text(0.83 * np.cos(angle), 0.83 * np.sin(angle), label, ha="center",
                va="center", fontsize=9)
    ax.set(aspect="equal")
    fig.tight_layout(pad=0.2)
    return _write(fig, name)


def _pie(share: pd.Series, name: str) -> str:
    """One slice per language, percentages printed outside the wedge."""
    _style(11)
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    colors = [GREY if label == "Others" else WHEEL[i % len(WHEEL)]
              for i, label in enumerate(share.index)]
    wedges, *_ = ax.pie(share.values, radius=1.0, colors=colors, startangle=90,
                        counterclock=False,
                        wedgeprops={"edgecolor": "white", "linewidth": 0.8})
    small = 0
    for wedge, label, value in zip(wedges, share.index, share.values):
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x, y = np.cos(angle), np.sin(angle)
        offset = 0.0
        if value < 2.0:
            offset = 0.10 + 0.24 * small
            small += 1
        text = f"{label} {value:.1f}%" if value >= 2.0 else label
        ax.annotate(text, xy=(x, y),
                    xytext=(1.18 * x, 1.10 * y + offset),
                    ha="center" if abs(x) < 0.4 else ("left" if x > 0 else "right"),
                    va="center", fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": "#777777",
                                "linewidth": 0.6, "shrinkA": 0, "shrinkB": 2})
    ax.set(aspect="equal")
    fig.tight_layout()
    return _write(fig, name)


def figures() -> list:
    """Write the five figures of the data section and return their paths."""
    shares = genre_shares()
    counts = genre_count_shares()
    inner, outer = region_country_shares()
    languages = language_shares()
    decades = release_decades()

    paths = [
        _vbar(shares.index, shares.values, "", "Share of movies (%)",
              "genre_distribution.png", (7.2, 4.0), size=12, label_size=8.5,
              tick_rotation=45, value_rotation=90, headroom=1.26),
        _vbar([str(i) for i in counts.index], counts.values, "Genres per movie",
              "Share of movies (%)", "genre_count_distribution.png", (3.4, 4.4),
              size=12, label_size=10),
        _sunburst(inner, outer, "region_distribution.png"),
        _pie(languages, "language_distribution.png"),
    ]

    _style(12)
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.bar(decades.index, decades.values / 1000, width=8, color=BLUE)
    ax.set_xlabel("Decade of first release")
    ax.set_ylabel("Movies (thousands)")
    ax.set_xticks(list(decades.index))
    ax.set_xticklabels([f"{d}s" for d in decades.index], fontsize=9.5,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(0, 176, 25))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color=GREY, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    paths.append(_write(fig, "release_decades.png"))
    return paths

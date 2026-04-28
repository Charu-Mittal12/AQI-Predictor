import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
print(BASE_DIR)
DATA_DIR     = BASE_DIR / "data" / "raw" / "cpcb"
BASELINE_DIR = BASE_DIR / "drift_baseline"
PLOTS_DIR    = BASE_DIR / "eda_plots"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STATION_HOUR_CSV = DATA_DIR / "station_hour.csv"
STATIONS_CSV     = DATA_DIR / "stations.csv"
BASELINE_FILE    = BASELINE_DIR / "baseline_stats.json"

POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]
TARGET_COL = "AQI"

AQI_BINS   = [0, 50, 100, 150, 200, 300, 500]
AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
AQI_COLORS = ["#00b050", "#92d050", "#ffff00", "#ff7c00", "#ff0000", "#7030a0"]

# ── Matplotlib style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#2e3248",
    "axes.labelcolor":  "#c8cde4",
    "axes.titlecolor":  "#e8ecff",
    "xtick.color":      "#8890b0",
    "ytick.color":      "#8890b0",
    "grid.color":       "#2a2e42",
    "grid.alpha":       0.6,
    "text.color":       "#c8cde4",
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "figure.dpi":       130,
})

ACCENT    = "#4f8ef7"
ACCENT2   = "#f7834f"
ACCENT3   = "#4ff7b8"
PLOT_BG   = "#0f1117"


# ── 1. Load & clean ────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(STATION_HOUR_CSV, parse_dates=["Datetime"])
    df.columns = df.columns.str.strip()

    present_cols = [c for c in POLLUTANT_COLS if c in df.columns]
    keep = ["StationId", "Datetime"] + present_cols + (
        [TARGET_COL] if TARGET_COL in df.columns else []
    ) + (["AQI_Bucket"] if "AQI_Bucket" in df.columns else [])
    df = df[[c for c in keep if c in df.columns]].copy()

    df.dropna(subset=present_cols, how="all", inplace=True)

    for col in present_cols + ([TARGET_COL] if TARGET_COL in df.columns else []):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[load] {len(df):,} rows | {df['StationId'].nunique()} stations")
    return df, present_cols


def load_stations():
    if not STATIONS_CSV.exists():
        print(f"[warn] stations.csv not found at {STATIONS_CSV}")
        return pd.DataFrame()
    stations = pd.read_csv(STATIONS_CSV)
    stations.columns = stations.columns.str.strip()
    print(f"[load] stations.csv → {len(stations)} records")
    return stations


# ── 2. Per-feature stats ───────────────────────────────────────────────────────
def compute_stats(df: pd.DataFrame, feature_cols: list) -> dict:
    stats = {}
    for col in feature_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        stats[col] = {
            "count":    int(s.count()),
            "mean":     round(float(s.mean()), 4),
            "std":      round(float(s.std()), 4),
            "min":      round(float(s.min()), 4),
            "p25":      round(float(s.quantile(0.25)), 4),
            "p50":      round(float(s.median()), 4),
            "p75":      round(float(s.quantile(0.75)), 4),
            "p95":      round(float(s.quantile(0.95)), 4),
            "max":      round(float(s.max()), 4),
            "null_pct": round(float(df[col].isna().mean() * 100), 2),
            "skewness": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurt()), 4),
            "histogram": _histogram(s),
        }
    return stats


def _histogram(series: pd.Series, bins: int = 50) -> dict:
    counts, edges = np.histogram(series.dropna(), bins=bins)
    return {
        "counts": counts.tolist(),
        "edges":  [round(float(e), 6) for e in edges.tolist()],
    }


# ── 3. Geographic / station overview ──────────────────────────────────────────
def analyze_stations(df: pd.DataFrame, stations: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  STATION & GEOGRAPHIC OVERVIEW")
    print("=" * 60)

    station_ids_in_data = set(df["StationId"].unique())
    n_stations = len(station_ids_in_data)
    print(f"\n  Total unique StationIds in hourly data : {n_stations}")

    if stations.empty:
        print("  (stations.csv unavailable – skipping geo breakdown)")
        return

    # Merge to get metadata only for stations present in hourly data
    meta = stations[stations["StationId"].isin(station_ids_in_data)].copy()
    missing_meta = station_ids_in_data - set(meta["StationId"])
    if missing_meta:
        print(f"  Stations in hourly data with no metadata : {len(missing_meta)}")

    if "State" in meta.columns:
        n_states = meta["State"].nunique()
        print(f"  States covered   : {n_states}")
        state_counts = meta["State"].value_counts()
        print(f"\n  Stations per State (top 15):")
        for state, cnt in state_counts.head(15).items():
            bar = "█" * cnt
            print(f"    {str(state):<30} {cnt:>3}  {bar}")

    if "City" in meta.columns:
        n_cities = meta["City"].nunique()
        print(f"\n  Cities covered   : {n_cities}")
        city_counts = meta["City"].value_counts()
        print(f"\n  Stations per City (top 20):")
        for city, cnt in city_counts.head(20).items():
            bar = "█" * cnt
            print(f"    {str(city):<28} {cnt:>3}  {bar}")

    if "Status" in meta.columns:
        print(f"\n  Station Status breakdown:")
        for status, cnt in meta["Status"].value_counts().items():
            print(f"    {str(status):<20} {cnt:>4}")

    # Records per station
    recs = df.groupby("StationId").size().rename("n_records")
    recs_df = recs.reset_index().merge(
        meta[["StationId"] + [c for c in ["StationName", "City", "State"] if c in meta.columns]],
        on="StationId", how="left"
    ).sort_values("n_records", ascending=False)

    print(f"\n  Top 10 stations by record count:")
    print(f"    {'StationId':<20} {'City':<20} {'State':<20} {'Records':>8}")
    print(f"    {'-'*68}")
    for _, row in recs_df.head(10).iterrows():
        city  = str(row.get("City", ""))[:18]
        state = str(row.get("State", ""))[:18]
        print(f"    {str(row['StationId']):<20} {city:<20} {state:<20} {row['n_records']:>8,}")

    print(f"\n  Bottom 10 stations by record count (data sparsity):")
    print(f"    {'StationId':<20} {'City':<20} {'State':<20} {'Records':>8}")
    print(f"    {'-'*68}")
    for _, row in recs_df.tail(10).iterrows():
        city  = str(row.get("City", ""))[:18]
        state = str(row.get("State", ""))[:18]
        print(f"    {str(row['StationId']):<20} {city:<20} {state:<20} {row['n_records']:>8,}")

    print("=" * 60)
    return meta, recs_df


# ── 4. Temporal overview ───────────────────────────────────────────────────────
def analyze_temporal(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  TEMPORAL OVERVIEW")
    print("=" * 60)
    print(f"  Date range  : {df['Datetime'].min()} → {df['Datetime'].max()}")
    total_hours = int((df['Datetime'].max() - df['Datetime'].min()).total_seconds() / 3600)
    print(f"  Span        : ~{total_hours:,} hours  (~{total_hours//24:,} days)")

    df["_year"]  = df["Datetime"].dt.year
    df["_month"] = df["Datetime"].dt.month
    df["_hour"]  = df["Datetime"].dt.hour

    print(f"\n  Records per Year:")
    for yr, cnt in df["_year"].value_counts().sort_index().items():
        bar = "█" * (cnt // max(df["_year"].value_counts().max() // 40, 1))
        print(f"    {yr}  {cnt:>8,}  {bar}")

    # Active stations per year
    active = df.groupby("_year")["StationId"].nunique()
    print(f"\n  Active Stations per Year:")
    for yr, cnt in active.items():
        print(f"    {yr}  {cnt:>4} stations")

    df.drop(columns=["_year", "_month", "_hour"], inplace=True)
    print("=" * 60)


# ── 5. Nullity / completeness ──────────────────────────────────────────────────
def analyze_nullity(df: pd.DataFrame, feature_cols: list):
    print("\n" + "=" * 60)
    print("  DATA COMPLETENESS")
    print("=" * 60)
    total = len(df)
    print(f"\n  {'Feature':<14} {'Present':>10} {'Missing':>10} {'Null%':>7}")
    print(f"  {'-'*46}")
    for col in feature_cols + ([TARGET_COL] if TARGET_COL in df.columns else []):
        if col not in df.columns:
            continue
        present = df[col].notna().sum()
        missing = total - present
        pct     = missing / total * 100
        bar     = "▓" * int(pct / 2)
        print(f"  {col:<14} {present:>10,} {missing:>10,} {pct:>6.1f}%  {bar}")
    print("=" * 60)


# ── 6. Print EDA summary ───────────────────────────────────────────────────────
def print_eda_summary(df: pd.DataFrame, stats: dict, feature_cols: list):
    print("\n" + "=" * 60)
    print("  POLLUTANT STATISTICS SUMMARY")
    print("=" * 60)

    print(f"\n{'Feature':<12} {'Mean':>8} {'Std':>8} {'P50':>8} {'P95':>8} {'Null%':>7} {'Skew':>7}")
    print("-" * 60)
    for col, v in stats.items():
        print(
            f"  {col:<12} {v['mean']:>8.2f} {v['std']:>8.2f} "
            f"{v['p50']:>8.2f} {v['p95']:>8.2f} "
            f"{v['null_pct']:>6.1f}% {v['skewness']:>7.2f}"
        )

    if TARGET_COL in df.columns:
        print(f"\n  AQI distribution:")
        df["aqi_cat"] = pd.cut(df[TARGET_COL], bins=AQI_BINS, labels=AQI_LABELS, right=True)
        dist = df["aqi_cat"].value_counts().sort_index()
        for cat, count in dist.items():
            pct = count / len(df.dropna(subset=[TARGET_COL])) * 100
            print(f"    {str(cat):<14} {count:>6,}  ({pct:.1f}%)")

    print("\n  Top correlations with AQI:")
    if TARGET_COL in df.columns:
        corr = df[feature_cols + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
        for col, val in corr.abs().sort_values(ascending=False).head(6).items():
            direction = "+" if corr[col] > 0 else "-"
            print(f"    {col:<12}  {direction}{abs(val):.3f}")

    print("=" * 60)


# ── 7. Save baseline ───────────────────────────────────────────────────────────
def save_baseline(stats: dict):
    payload = {"source": "training", "features": stats}
    with open(BASELINE_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[baseline] Saved → {BASELINE_FILE}")
    print("[baseline] Run drift_detection.py to compare new data against this baseline.")


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, name: str):
    path = PLOTS_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor=PLOT_BG)
    plt.close(fig)
    print(f"  [plot] → {path}")


# ── V1: Stations per State bar chart ──────────────────────────────────────────
def plot_stations_per_state(meta: pd.DataFrame):
    if "State" not in meta.columns:
        return
    counts = meta["State"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(5, len(counts) * 0.35)), facecolor=PLOT_BG)
    colors = plt.cm.cool(np.linspace(0.2, 0.9, len(counts)))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="none", height=0.7)
    for bar, val in zip(bars, counts.values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left", fontsize=9, color="#c8cde4")
    ax.set_xlabel("Number of Monitoring Stations")
    ax.set_title("Monitoring Stations by State", fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    _save(fig, "v1_stations_per_state.png")


# ── V2: Stations per City (top 20) ────────────────────────────────────────────
def plot_stations_per_city(meta: pd.DataFrame):
    if "City" not in meta.columns:
        return
    counts = meta["City"].value_counts().head(20).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=PLOT_BG)
    colors = plt.cm.plasma(np.linspace(0.25, 0.85, len(counts)))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="none", height=0.7)
    for bar, val in zip(bars, counts.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left", fontsize=9, color="#c8cde4")
    ax.set_xlabel("Number of Monitoring Stations")
    ax.set_title("Top 20 Cities by Station Count", fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    _save(fig, "v2_stations_per_city.png")


# ── V3: Records per Year ───────────────────────────────────────────────────────
def plot_records_per_year(df: pd.DataFrame):
    yr_counts = df["Datetime"].dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=PLOT_BG)
    ax.bar(yr_counts.index.astype(str), yr_counts.values,
           color=ACCENT, edgecolor="#0f1117", linewidth=0.5)
    for x, y in zip(yr_counts.index, yr_counts.values):
        ax.text(str(x), y + yr_counts.max() * 0.01, f"{y:,}",
                ha="center", va="bottom", fontsize=9, color="#c8cde4")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Records")
    ax.set_title("Hourly Records per Year", fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    _save(fig, "v3_records_per_year.png")


# ── V4: Active stations per year ──────────────────────────────────────────────
def plot_active_stations_per_year(df: pd.DataFrame):
    active = df.groupby(df["Datetime"].dt.year)["StationId"].nunique()
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=PLOT_BG)
    ax.plot(active.index.astype(str), active.values,
            marker="o", color=ACCENT2, linewidth=2.5, markersize=8)
    ax.fill_between(range(len(active)), active.values, alpha=0.15, color=ACCENT2)
    for x, (yr, y) in enumerate(active.items()):
        ax.text(x, y + active.max() * 0.02, str(y),
                ha="center", va="bottom", fontsize=9, color="#c8cde4")
    ax.set_xticks(range(len(active)))
    ax.set_xticklabels(active.index.astype(str), rotation=45)
    ax.set_xlabel("Year")
    ax.set_ylabel("Active Stations")
    ax.set_title("Active Monitoring Stations per Year", fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    _save(fig, "v4_active_stations_per_year.png")


# ── V5: AQI bucket distribution ───────────────────────────────────────────────
def plot_aqi_distribution(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        return
    df2 = df.copy()
    df2["aqi_cat"] = pd.cut(df2[TARGET_COL], bins=AQI_BINS, labels=AQI_LABELS, right=True)
    dist = df2["aqi_cat"].value_counts().reindex(AQI_LABELS).fillna(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=PLOT_BG)

    # Bar
    bars = ax1.bar(dist.index, dist.values, color=AQI_COLORS, edgecolor="#0f1117", linewidth=0.5)
    for bar, val in zip(bars, dist.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + dist.max() * 0.01,
                 f"{int(val):,}", ha="center", va="bottom", fontsize=8.5, color="#c8cde4")
    ax1.set_xlabel("AQI Category")
    ax1.set_ylabel("Number of Readings")
    ax1.set_title("AQI Category Distribution (Count)", fontweight="bold", pad=12)
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(axis="y", alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.spines[:].set_visible(False)

    # Pie
    wedge_props = {"linewidth": 1.5, "edgecolor": PLOT_BG}
    ax2.pie(dist.values, labels=dist.index, colors=AQI_COLORS,
            autopct="%1.1f%%", pctdistance=0.75,
            wedgeprops=wedge_props, textprops={"color": "#c8cde4", "fontsize": 9},
            startangle=90)
    ax2.set_title("AQI Category Distribution (%)", fontweight="bold", pad=12)

    fig.tight_layout()
    _save(fig, "v5_aqi_distribution.png")


# ── V6: AQI histogram ─────────────────────────────────────────────────────────
def plot_aqi_histogram(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        return
    aqi = df[TARGET_COL].dropna()
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=PLOT_BG)
    n, bins, patches = ax.hist(aqi, bins=80, edgecolor="none")
    # Color by AQI range
    for patch, left_edge in zip(patches, bins[:-1]):
        if   left_edge < 50:   patch.set_facecolor(AQI_COLORS[0])
        elif left_edge < 100:  patch.set_facecolor(AQI_COLORS[1])
        elif left_edge < 150:  patch.set_facecolor(AQI_COLORS[2])
        elif left_edge < 200:  patch.set_facecolor(AQI_COLORS[3])
        elif left_edge < 300:  patch.set_facecolor(AQI_COLORS[4])
        else:                  patch.set_facecolor(AQI_COLORS[5])
    for b, label, color in zip(AQI_BINS[:-1], AQI_LABELS, AQI_COLORS):
        ax.axvline(b, color=color, linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("AQI Value")
    ax.set_ylabel("Frequency")
    ax.set_title("AQI Value Distribution", fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    _save(fig, "v6_aqi_histogram.png")


# ── V7: Pollutant distributions (box plots) ───────────────────────────────────
def plot_pollutant_boxplots(df: pd.DataFrame, feature_cols: list):
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), facecolor=PLOT_BG)
    axes = axes.flatten()
    palette = plt.cm.viridis(np.linspace(0.15, 0.9, len(feature_cols)))
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        data = df[col].dropna()
        # Cap at 99th percentile for readability
        cap = data.quantile(0.99)
        data_capped = data[data <= cap]
        bp = ax.boxplot(data_capped, vert=True, patch_artist=True,
                        medianprops={"color": "#fff", "linewidth": 2},
                        boxprops={"facecolor": palette[i], "alpha": 0.7},
                        whiskerprops={"color": "#8890b0"},
                        capprops={"color": "#8890b0"},
                        flierprops={"marker": ".", "color": palette[i], "alpha": 0.2, "markersize": 2})
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)
        ax.spines[:].set_visible(False)
        ax.tick_params(labelsize=8)
        # Annotate median
        median = data.median()
        ax.text(1.38, median, f"med={median:.1f}", va="center",
                fontsize=7.5, color="#c8cde4", transform=ax.get_yaxis_transform())
    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Pollutant Distributions (capped at 99th pct)", fontsize=14,
                 fontweight="bold", color="#e8ecff", y=1.01)
    fig.tight_layout()
    _save(fig, "v7_pollutant_boxplots.png")


# ── V8: Null-heatmap by feature ───────────────────────────────────────────────
def plot_nullity_heatmap(df: pd.DataFrame, feature_cols: list):
    cols = feature_cols + ([TARGET_COL] if TARGET_COL in df.columns else [])
    null_pct = df[cols].isnull().mean() * 100
    fig, ax = plt.subplots(figsize=(12, 2.5), facecolor=PLOT_BG)
    cmap = LinearSegmentedColormap.from_list("nullmap", ["#1e3a5f", "#ff4444"])
    im = ax.imshow([null_pct.values], cmap=cmap, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=10)
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.35, fraction=0.04, label="Null %")
    ax.set_title("Null Percentage by Feature", fontweight="bold", pad=12)
    for i, (col, val) in enumerate(null_pct.items()):
        ax.text(i, 0, f"{val:.1f}%", ha="center", va="center",
                fontsize=9, color="white" if val > 40 else "#c8cde4", fontweight="bold")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    _save(fig, "v8_nullity_heatmap.png")


# ── V9: Correlation heatmap ───────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list):
    cols = feature_cols + ([TARGET_COL] if TARGET_COL in df.columns else [])
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 9), facecolor=PLOT_BG)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=0.5, linecolor="#2e3248",
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    _save(fig, "v9_correlation_heatmap.png")


# ── V10: AQI trend over time (monthly median) ─────────────────────────────────
def plot_aqi_trend(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        return
    monthly = (df.set_index("Datetime")[TARGET_COL]
                 .resample("ME").median().dropna())
    fig, ax = plt.subplots(figsize=(14, 4), facecolor=PLOT_BG)
    ax.plot(monthly.index, monthly.values, color=ACCENT, linewidth=1.5)
    ax.fill_between(monthly.index, monthly.values,
                    monthly.min(), alpha=0.15, color=ACCENT)
    for threshold, label, color in zip(
        [50, 100, 150, 200], ["Good", "Satisfactory", "Moderate", "Poor"],
        AQI_COLORS[:4]
    ):
        ax.axhline(threshold, linestyle="--", alpha=0.35, linewidth=1,
                   color=color, label=f"{label} ({threshold})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Median AQI")
    ax.set_title("Monthly Median AQI Trend (All Stations)", fontweight="bold", pad=12)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.3)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    _save(fig, "v10_aqi_monthly_trend.png")


# ── V11: Hourly AQI pattern ───────────────────────────────────────────────────
def plot_hourly_aqi_pattern(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        return
    hourly = df.groupby(df["Datetime"].dt.hour)[TARGET_COL].median()
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=PLOT_BG)
    ax.plot(hourly.index, hourly.values, marker="o", color=ACCENT3,
            linewidth=2.5, markersize=6)
    ax.fill_between(hourly.index, hourly.values, hourly.min(),
                    alpha=0.15, color=ACCENT3)
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Hour of Day (0–23)")
    ax.set_ylabel("Median AQI")
    ax.set_title("Median AQI by Hour of Day", fontweight="bold", pad=12)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    _save(fig, "v11_hourly_aqi_pattern.png")


# ── V12: Records per station (top 30) ─────────────────────────────────────────
def plot_records_per_station(recs_df: pd.DataFrame):
    top = recs_df.head(30).sort_values("n_records")
    label_col = "City" if "City" in top.columns else "StationId"
    labels = top[label_col].fillna(top["StationId"]).astype(str)

    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.35)), facecolor=PLOT_BG)
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.95, len(top)))
    bars = ax.barh(labels, top["n_records"].values,
                   color=colors, edgecolor="none", height=0.7)
    for bar, val in zip(bars, top["n_records"].values):
        ax.text(val + top["n_records"].max() * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", ha="left", fontsize=8.5, color="#c8cde4")
    ax.set_xlabel("Number of Hourly Records")
    ax.set_title("Top 30 Stations by Record Count", fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    _save(fig, "v12_records_per_station.png")


# ── V13: AQI by State (boxplot) ───────────────────────────────────────────────
def plot_aqi_by_state(df: pd.DataFrame, meta: pd.DataFrame):
    if TARGET_COL not in df.columns or "State" not in meta.columns:
        return
    merged = df[["StationId", TARGET_COL]].merge(
        meta[["StationId", "State"]], on="StationId", how="left"
    ).dropna(subset=[TARGET_COL, "State"])

    state_median = merged.groupby("State")[TARGET_COL].median().sort_values(ascending=False)
    top_states   = state_median.head(20).index.tolist()
    filtered     = merged[merged["State"].isin(top_states)]

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=PLOT_BG)
    palette = {s: plt.cm.RdYlGn_r(i / len(top_states))
               for i, s in enumerate(top_states)}
    data_by_state = [filtered[filtered["State"] == s][TARGET_COL].dropna().values
                     for s in top_states]
    bp = ax.boxplot(data_by_state, labels=top_states, patch_artist=True,
                    medianprops={"color": "white", "linewidth": 2},
                    whiskerprops={"color": "#8890b0"},
                    capprops={"color": "#8890b0"},
                    flierprops={"marker": ".", "alpha": 0.1, "markersize": 2})
    for patch, state in zip(bp["boxes"], top_states):
        patch.set_facecolor(palette[state])
        patch.set_alpha(0.75)
    ax.set_xlabel("State")
    ax.set_ylabel("AQI")
    ax.set_title("AQI Distribution by State (Top 20 by Median AQI)", fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    _save(fig, "v13_aqi_by_state.png")


# ── V14: Monthly seasonality heatmap ──────────────────────────────────────────
def plot_monthly_heatmap(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        return
    df2 = df.copy()
    df2["Year"]  = df2["Datetime"].dt.year
    df2["Month"] = df2["Datetime"].dt.month
    pivot = df2.groupby(["Year", "Month"])[TARGET_COL].median().unstack(level=1)
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(13, max(3, len(pivot) * 0.7)), facecolor=PLOT_BG)
    cmap = LinearSegmentedColormap.from_list(
        "aqi", ["#00b050","#ffff00","#ff7c00","#ff0000","#7030a0"])
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=300)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str), fontsize=10)
    plt.colorbar(im, ax=ax, label="Median AQI", fraction=0.03, pad=0.02)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if val > 150 else "#0f1117",
                        fontweight="bold")
    ax.set_title("Monthly Median AQI Heatmap (Year × Month)", fontweight="bold", pad=12)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    _save(fig, "v14_monthly_seasonality_heatmap.png")


# ── V15: Station Status pie ────────────────────────────────────────────────────
def plot_station_status(meta: pd.DataFrame):
    if "Status" not in meta.columns:
        return
    counts = meta["Status"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=PLOT_BG)
    colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
    wedge_props = {"linewidth": 2, "edgecolor": PLOT_BG}
    ax.pie(counts.values, labels=counts.index, colors=colors,
           autopct="%1.1f%%", pctdistance=0.78,
           wedgeprops=wedge_props, textprops={"color": "#c8cde4", "fontsize": 10},
           startangle=90)
    ax.set_title("Station Status Breakdown", fontweight="bold", pad=12)
    fig.tight_layout()
    _save(fig, "v15_station_status.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nRunning EDA on CPCB data...")

    # ── Data loading ────────────────────────────────────────────────────────
    df, feature_cols = load_data()
    stations         = load_stations()

    # ── Textual analyses ────────────────────────────────────────────────────
    meta, recs_df = (None, None)
    if not stations.empty:
        result = analyze_stations(df, stations)
        if result:
            meta, recs_df = result

    analyze_temporal(df)
    analyze_nullity(df, feature_cols)

    stats = compute_stats(df, feature_cols)
    print_eda_summary(df, stats, feature_cols)
    save_baseline(stats)

    # ── Visualisations ──────────────────────────────────────────────────────
    print(f"\n[plots] Generating visualisations → {PLOTS_DIR}")

    # Station & geography
    if meta is not None:
        plot_stations_per_state(meta)
        plot_stations_per_city(meta)
        plot_station_status(meta)
        plot_aqi_by_state(df, meta)

    if recs_df is not None:
        plot_records_per_station(recs_df)

    # Temporal
    plot_records_per_year(df)
    plot_active_stations_per_year(df)
    plot_aqi_trend(df)
    plot_hourly_aqi_pattern(df)
    plot_monthly_heatmap(df)

    # AQI / pollutants
    plot_aqi_distribution(df)
    plot_aqi_histogram(df)
    plot_pollutant_boxplots(df, feature_cols)
    plot_nullity_heatmap(df, feature_cols)
    plot_correlation_heatmap(df, feature_cols)

    print(f"\n[done] All plots saved to {PLOTS_DIR}/")
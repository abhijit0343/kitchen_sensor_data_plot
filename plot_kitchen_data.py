"""
plot_kitchen_data.py
--------------------
Comprehensive visualization of kitchen_merged_labels data.
Sensors: temperature, humidity, PM2.5, VOC, CO2
Labels : oven1_label, oven2_label
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 120, "font.size": 9})

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
CSV_PATH = r"d:\nit_internship\Csv_plotting\kitchen_merged_labels_cleaned_oven1_activity_oven2_idle_or_viceversa.csv"

df = pd.read_csv(CSV_PATH)

# Parse timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
df = df.sort_values("timestamp").reset_index(drop=True)

SENSORS = ["data_temperature", "data_humidity", "data_pm2_5", "data_voc", "data_co2"]
SENSOR_LABELS = {
    "data_temperature": "Temperature (°C)",
    "data_humidity":    "Humidity (%)",
    "data_pm2_5":       "PM2.5 (µg/m³)",
    "data_voc":         "VOC",
    "data_co2":         "CO₂ (ppm)",
}
COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]

# ─────────────────────────────────────────────
# FIGURE 1 — Time-series of all sensors
# ─────────────────────────────────────────────
fig1, axes = plt.subplots(len(SENSORS), 1, figsize=(16, 12), sharex=True)
fig1.suptitle("Sensor Time-Series — Kitchen Merged Labels", fontsize=13, fontweight="bold")

for ax, sensor, color in zip(axes, SENSORS, COLORS):
    ax.plot(df["timestamp"], df[sensor], color=color, linewidth=0.7, alpha=0.85)
    ax.set_ylabel(SENSOR_LABELS[sensor], fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.fill_between(df["timestamp"], df[sensor], alpha=0.12, color=color)

axes[-1].set_xlabel("Timestamp", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig1.savefig(r"d:\nit_internship\Csv_plotting\fig1_time_series.png", bbox_inches="tight")
print("Saved fig1_time_series.png")

# ─────────────────────────────────────────────
# FIGURE 2 — Sensor distributions (violin + box overlay)
# ─────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, len(SENSORS), figsize=(18, 5))
fig2.suptitle("Sensor Value Distributions (Violin + Box)", fontsize=13, fontweight="bold")

for ax, sensor, color in zip(axes2, SENSORS, COLORS):
    data = df[sensor].dropna().values
    vp = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.55)
    bp = ax.boxplot(data, widths=0.18, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.8))
    bp["boxes"][0].set_facecolor(color + "88")
    ax.set_title(SENSOR_LABELS[sensor], fontsize=8, fontweight="bold")
    ax.set_xticks([])
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(r"d:\nit_internship\Csv_plotting\fig2_distributions.png", bbox_inches="tight")
print("Saved fig2_distributions.png")

# ─────────────────────────────────────────────
# FIGURE 3 — Mean sensor values per Oven-1 label (top-15)
# ─────────────────────────────────────────────
top15_oven1 = (df.groupby("oven1_label")[SENSORS]
               .mean()
               .sort_values("data_pm2_5", ascending=False)
               .head(15))

fig3, axes3 = plt.subplots(len(SENSORS), 1, figsize=(16, 14))
fig3.suptitle("Mean Sensor Values per Oven-1 Activity Label (Top-15 by PM2.5)", fontsize=12, fontweight="bold")

for ax, sensor, color in zip(axes3, SENSORS, COLORS):
    bars = ax.barh(top15_oven1.index, top15_oven1[sensor], color=color, alpha=0.82, edgecolor="white")
    ax.set_xlabel(SENSOR_LABELS[sensor], fontsize=8)
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    # annotate
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.005, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}", va="center", fontsize=6.5)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig3.savefig(r"d:\nit_internship\Csv_plotting\fig3_oven1_label_means.png", bbox_inches="tight")
print("Saved fig3_oven1_label_means.png")

# ─────────────────────────────────────────────
# FIGURE 4 — Mean sensor values per Oven-2 label (top-15)
# ─────────────────────────────────────────────
top15_oven2 = (df.groupby("oven2_label")[SENSORS]
               .mean()
               .sort_values("data_pm2_5", ascending=False)
               .head(15))

fig4, axes4 = plt.subplots(len(SENSORS), 1, figsize=(16, 14))
fig4.suptitle("Mean Sensor Values per Oven-2 Activity Label (Top-15 by PM2.5)", fontsize=12, fontweight="bold")

for ax, sensor, color in zip(axes4, SENSORS, COLORS):
    bars = ax.barh(top15_oven2.index, top15_oven2[sensor], color=color, alpha=0.82, edgecolor="white")
    ax.set_xlabel(SENSOR_LABELS[sensor], fontsize=8)
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.005, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}", va="center", fontsize=6.5)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig4.savefig(r"d:\nit_internship\Csv_plotting\fig4_oven2_label_means.png", bbox_inches="tight")
print("Saved fig4_oven2_label_means.png")

# ─────────────────────────────────────────────
# FIGURE 5 — Correlation heatmap
# ─────────────────────────────────────────────
corr = df[SENSORS].corr()
fig5, ax5 = plt.subplots(figsize=(7, 5))
fig5.suptitle("Sensor Correlation Heatmap", fontsize=12, fontweight="bold")

im = ax5.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
labels = [SENSOR_LABELS[s] for s in SENSORS]
ax5.set_xticks(range(len(SENSORS)))
ax5.set_yticks(range(len(SENSORS)))
ax5.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
ax5.set_yticklabels(labels, fontsize=8)

for i in range(len(SENSORS)):
    for j in range(len(SENSORS)):
        ax5.text(j, i, f"{corr.values[i, j]:.2f}",
                 ha="center", va="center", fontsize=8,
                 color="black" if abs(corr.values[i, j]) < 0.7 else "white")

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig5.savefig(r"d:\nit_internship\Csv_plotting\fig5_correlation_heatmap.png", bbox_inches="tight")
print("Saved fig5_correlation_heatmap.png")

# ─────────────────────────────────────────────
# FIGURE 6 — Scatter matrix (pair plot) for sensors
# ─────────────────────────────────────────────
n = len(SENSORS)
fig6, axes6 = plt.subplots(n, n, figsize=(16, 14))
fig6.suptitle("Sensor Pair Scatter Matrix", fontsize=12, fontweight="bold")

for i, s1 in enumerate(SENSORS):
    for j, s2 in enumerate(SENSORS):
        ax = axes6[i][j]
        if i == j:
            ax.hist(df[s1].dropna(), bins=30, color=COLORS[i], alpha=0.75, edgecolor="none")
        else:
            ax.scatter(df[s2], df[s1], s=2, alpha=0.25, color=COLORS[i])
        if j == 0:
            ax.set_ylabel(SENSOR_LABELS[s1], fontsize=6)
        if i == n - 1:
            ax.set_xlabel(SENSOR_LABELS[s2], fontsize=6)
        ax.tick_params(labelsize=5)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig6.savefig(r"d:\nit_internship\Csv_plotting\fig6_pair_scatter.png", bbox_inches="tight")
print("Saved fig6_pair_scatter.png")

# ─────────────────────────────────────────────
# FIGURE 7 — Oven-1 label count bar chart + Oven-2 pie
# ─────────────────────────────────────────────
oven1_counts = df["oven1_label"].value_counts()
oven2_counts = df["oven2_label"].value_counts()

fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(18, 7))
fig7.suptitle("Label Distribution — Oven-1 (bar) & Oven-2 (pie)", fontsize=12, fontweight="bold")

cmap = plt.get_cmap("tab20")
# Bar
bars7 = ax7a.barh(oven1_counts.index, oven1_counts.values,
                  color=[cmap(i / len(oven1_counts)) for i in range(len(oven1_counts))],
                  edgecolor="white")
ax7a.set_xlabel("Count")
ax7a.set_title("Oven-1 Label Frequency")
ax7a.invert_yaxis()
ax7a.grid(True, axis="x", linestyle="--", alpha=0.4)

# Pie (group small slices)
threshold = 0.03 * oven2_counts.sum()
oven2_major = oven2_counts[oven2_counts >= threshold]
oven2_other = oven2_counts[oven2_counts < threshold].sum()
if oven2_other > 0:
    oven2_major["Other"] = oven2_other

wedges, texts, autotexts = ax7b.pie(
    oven2_major.values,
    labels=oven2_major.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=[cmap(i / len(oven2_major)) for i in range(len(oven2_major))],
    pctdistance=0.82,
)
for t in texts:
    t.set_fontsize(7)
for a in autotexts:
    a.set_fontsize(6.5)
ax7b.set_title("Oven-2 Label Share")

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig7.savefig(r"d:\nit_internship\Csv_plotting\fig7_label_distribution.png", bbox_inches="tight")
print("Saved fig7_label_distribution.png")

# ─────────────────────────────────────────────
# FIGURE 8 — Grouped box plots: sensor variation per cooking method
#             (Boil / Fry / Curry / Reheat / IDLE / Other)
# ─────────────────────────────────────────────
def cooking_method(label: str) -> str:
    label_lower = label.lower()
    for method in ("boil", "fry", "curry", "reheat", "idle"):
        if method in label_lower:
            return method.capitalize()
    return "Other"

df["method_oven1"] = df["oven1_label"].apply(cooking_method)
METHODS = ["Boil", "Fry", "Curry", "Reheat", "Idle", "Other"]
method_colors = {"Boil": "#4CC9F0", "Fry": "#F72585", "Curry": "#7209B7",
                 "Reheat": "#FB8500", "Idle": "#8ECAE6", "Other": "#ADB5BD"}

fig8, axes8 = plt.subplots(len(SENSORS), 1, figsize=(14, 13))
fig8.suptitle("Sensor Variation by Cooking Method (Oven-1)", fontsize=12, fontweight="bold")

for ax, sensor in zip(axes8, SENSORS):
    data_by_method = [df.loc[df["method_oven1"] == m, sensor].dropna().values for m in METHODS]
    bp = ax.boxplot(data_by_method, tick_labels=METHODS, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker=".", markersize=2, alpha=0.4))
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set_facecolor(method_colors[m])
        patch.set_alpha(0.75)
    ax.set_ylabel(SENSOR_LABELS[sensor], fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig8.savefig(r"d:\nit_internship\Csv_plotting\fig8_method_boxplot.png", bbox_inches="tight")
print("Saved fig8_method_boxplot.png")

# ─────────────────────────────────────────────
# FIGURE 9 — Rolling statistics (mean ± std) for PM2.5 & CO2
# ─────────────────────────────────────────────
WINDOW = 20

fig9, (ax9a, ax9b) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig9.suptitle(f"Rolling Mean ± 1 Std (window={WINDOW}) — PM2.5 & CO₂", fontsize=12, fontweight="bold")

for ax, sensor, color, label in [
    (ax9a, "data_pm2_5", "#2A9D8F", "PM2.5 (µg/m³)"),
    (ax9b, "data_co2",   "#E63946", "CO₂ (ppm)"),
]:
    rm = df[sensor].rolling(WINDOW).mean()
    rs = df[sensor].rolling(WINDOW).std()
    ax.plot(df["timestamp"], rm, color=color, linewidth=1.2, label="Rolling Mean")
    ax.fill_between(df["timestamp"], rm - rs, rm + rs, color=color, alpha=0.2, label="±1 Std")
    ax.plot(df["timestamp"], df[sensor], color=color, linewidth=0.4, alpha=0.35, label="Raw")
    ax.set_ylabel(label, fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.35)

ax9b.set_xlabel("Timestamp", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig9.savefig(r"d:\nit_internship\Csv_plotting\fig9_rolling_stats.png", bbox_inches="tight")
print("Saved fig9_rolling_stats.png")

# ─────────────────────────────────────────────
# FIGURE 10 — Stacked area: normalised sensor trends over time
# ─────────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df[SENSORS]), columns=SENSORS)
df_norm["timestamp"] = df["timestamp"].values

fig10, ax10 = plt.subplots(figsize=(16, 5))
fig10.suptitle("Normalised Sensor Trends (Min-Max Scaled) — Stacked Area", fontsize=12, fontweight="bold")

ax10.stackplot(df_norm["timestamp"],
               [df_norm[s] for s in SENSORS],
               labels=[SENSOR_LABELS[s] for s in SENSORS],
               colors=COLORS, alpha=0.72)
ax10.legend(loc="upper left", fontsize=8, ncol=2)
ax10.set_xlabel("Timestamp", fontsize=9)
ax10.set_ylabel("Normalised Value (stacked)", fontsize=9)
ax10.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig10.savefig(r"d:\nit_internship\Csv_plotting\fig10_stacked_area.png", bbox_inches="tight")
print("Saved fig10_stacked_area.png")

# ─────────────────────────────────────────────
print("\nAll figures saved successfully!")
plt.show()

"""
4_plot_results.py — Generate comparison charts from benchmark_results.csv
Produces a single PNG with 4 subplots saved to results/benchmark_charts.png
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path(r"C:\Users\parig\PycharmProjects\quantized-edge-llm")
RES_DIR = BASE / "results"
CSV_IN  = RES_DIR / "benchmark_results.csv"
PNG_OUT = RES_DIR / "benchmark_charts.png"

# ── Load CSV ───────────────────────────────────────────────────────────────────
if not CSV_IN.exists():
    raise FileNotFoundError(f"Run 3_benchmark.py first — {CSV_IN} not found")

rows = []
with open(CSV_IN, newline="") as f:
    for row in csv.DictReader(f):
        rows.append({
            "label":        row["label"],
            "file_mb":      float(row["file_mb"]),
            "load_time_s":  float(row["load_time_s"]),
            "ram_delta_mb": float(row["ram_delta_mb"]),
            "avg_tps":      float(row["avg_tps"]),
            "avg_ttft_ms":  float(row["avg_ttft_ms"]),
        })

labels  = [r["label"]        for r in rows]
file_mb = [r["file_mb"]      for r in rows]
tps     = [r["avg_tps"]      for r in rows]
ttft    = [r["avg_ttft_ms"]  for r in rows]
load_s  = [r["load_time_s"]  for r in rows]
ram_mb  = [r["ram_delta_mb"] for r in rows]

x = np.arange(len(labels))
W = 0.55

PALETTE = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#1d3557"]
colors  = (PALETTE * 5)[:len(labels)]

# ── Helper: draw bars + value labels in one shot (no clearing needed) ──────────
def draw_bars(ax, values, fmt="{:.1f}", suffix=""):
    bars = ax.bar(x, values, width=W, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(values) * 0.02,
            fmt.format(val) + suffix,
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
    return bars

def style_ax(ax, title, ylabel):
    ax.set_facecolor("#ffffff")
    ax.set_title(title, fontsize=11, fontweight="semibold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle(
    "Llama-3.2-1B Mental Health — Quantization Benchmark\n(llama-cpp-python · CPU inference)",
    fontsize=13, fontweight="bold", y=1.01,
)
fig.patch.set_facecolor("#f8f9fa")

# Plot 1 — File size
ax = axes[0, 0]
draw_bars(ax, file_mb, fmt="{:.0f}", suffix=" MB")
style_ax(ax, "📦 Model File Size", "Size (MB)")

# Plot 2 — Tokens/sec
ax = axes[0, 1]
draw_bars(ax, tps, fmt="{:.1f}", suffix=" t/s")
style_ax(ax, "⚡ Inference Speed", "Tokens / second")
# Speedup annotations vs F16
if len(tps) > 1:
    f16_tps = tps[0]
    for i in range(1, len(tps)):
        speedup = tps[i] / f16_tps if f16_tps else 0
        ax.annotate(
            f"×{speedup:.1f}",
            xy=(i, tps[i]),
            xytext=(i, tps[i] + max(tps) * 0.10),
            ha="center", fontsize=7, color="#555555",
            arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.6),
        )

# Plot 3 — Time to first token
ax = axes[1, 0]
draw_bars(ax, ttft, fmt="{:.0f}", suffix=" ms")
style_ax(ax, "🕐 Time to First Token", "TTFT (ms)")

# Plot 4 — RAM delta
ax = axes[1, 1]
draw_bars(ax, ram_mb, fmt="{:.0f}", suffix=" MB")
style_ax(ax, "🧠 RAM Usage (delta)", "RAM (MB)")

plt.tight_layout(pad=2.0)
fig.savefig(PNG_OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n✓ Chart saved → {PNG_OUT}\n")
plt.close(fig)
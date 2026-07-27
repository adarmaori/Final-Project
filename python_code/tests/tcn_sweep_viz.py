"""
TCN Sweep Analysis
===================
Reads the sweep CSV and generates four ESR-focused plots:
1. Parameter count vs ESR% on a log x-axis
2. Average ESR% by channel count
3. Average ESR% by layer count
4. Average ESR% by kernel size

Usage:
    python tests/tcn_sweep_viz.py
"""

import os

import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "models", "sweep_results", "sweep_report_small.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "sweep_results", "analysis")
OUTPUT_FIGURE = os.path.join(OUTPUT_DIR, "tcn_sweep_esr_analysis.png")
OUTPUT_EFFECT_FIGURE = os.path.join(OUTPUT_DIR, "tcn_sweep_esr_by_effect.png")
OUTPUT_SUMMARY = os.path.join(OUTPUT_DIR, "tcn_sweep_esr_summary.csv")


def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find sweep report: {CSV_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    numeric_cols = ["Channels", "Layers", "Kernel", "Params", "ESR%", "MSE"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()
    df["Effect"] = df["Effect"].astype(str)

    summary_rows = []
    for effect, effect_df in df.groupby("Effect"):
        summary_rows.append(
            {
                "Effect": effect,
                "Avg_ESR%": effect_df["ESR%"].mean(),
                "Avg_MSE": effect_df["MSE"].mean(),
                "Min_Params": effect_df["Params"].min(),
                "Max_Params": effect_df["Params"].max(),
                "Count": len(effect_df),
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUTPUT_SUMMARY, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("TCN Sweep ESR Analysis", fontsize=16, fontweight="bold")

    colors = {"exciter": "tab:blue", "distortion": "tab:orange"}
    markers = {"exciter": "o", "distortion": "s"}

    ax = axes[0, 0]
    for effect, effect_df in df.groupby("Effect"):
        ax.scatter(
            effect_df["Params"],
            effect_df["ESR%"],
            alpha=0.75,
            s=40,
            label=effect,
            color=colors.get(effect, None),
            marker=markers.get(effect, "o"),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Parameter Count (log scale)")
    ax.set_ylabel("ESR %")
    ax.set_title("1. Parameter Count vs ESR %")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    ax = axes[0, 1]
    channel_avg = df.groupby("Channels", as_index=False)["ESR%"].mean().sort_values("Channels")
    ax.bar(channel_avg["Channels"].astype(str), channel_avg["ESR%"], color="tab:blue")
    ax.set_xlabel("Channel Count")
    ax.set_ylabel("Average ESR %")
    ax.set_title("2. Average ESR % by Channel Count")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    ax = axes[1, 0]
    layer_avg = df.groupby("Layers", as_index=False)["ESR%"].mean().sort_values("Layers")
    ax.bar(layer_avg["Layers"].astype(str), layer_avg["ESR%"], color="tab:green")
    ax.set_xlabel("Layer Count")
    ax.set_ylabel("Average ESR %")
    ax.set_title("3. Average ESR % by Layer Count")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    ax = axes[1, 1]
    kernel_avg = df.groupby("Kernel", as_index=False)["ESR%"].mean().sort_values("Kernel")
    ax.bar(kernel_avg["Kernel"].astype(str), kernel_avg["ESR%"], color="tab:red")
    ax.set_xlabel("Kernel Size")
    ax.set_ylabel("Average ESR %")
    ax.set_title("4. Average ESR % by Kernel Size")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    fig.savefig(OUTPUT_FIGURE, dpi=200)
    print(f"Saved plot figure to {OUTPUT_FIGURE}")

    effect_order = list(df["Effect"].dropna().unique())
    comparison_specs = [
        ("Channels", "Average ESR % by Channel Count"),
        ("Layers", "Average ESR % by Layer Count"),
        ("Kernel", "Average ESR % by Kernel Size"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    fig.suptitle("TCN Sweep ESR by Effect", fontsize=16, fontweight="bold")

    for ax, (column, title) in zip(axes, comparison_specs):
        pivot = (
            df.pivot_table(index=column, columns="Effect", values="ESR%", aggfunc="mean")
            .sort_index()
            .reindex(columns=effect_order)
        )

        x_positions = range(len(pivot.index))
        bar_width = 0.8 / max(len(effect_order), 1)

        for idx, effect in enumerate(effect_order):
            offsets = [x + (idx - (len(effect_order) - 1) / 2) * bar_width for x in x_positions]
            ax.bar(offsets, pivot[effect].values, width=bar_width, label=effect, color=colors.get(effect, None))

        ax.set_xticks(list(x_positions))
        ax.set_xticklabels([str(value) for value in pivot.index])
        ax.set_xlabel(column[:-1] if column.endswith("s") else column)
        ax.set_ylabel("Average ESR %")
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    axes[0].legend()
    fig.savefig(OUTPUT_EFFECT_FIGURE, dpi=200)
    print(f"Saved effect comparison figure to {OUTPUT_EFFECT_FIGURE}")
    print(f"Saved summary CSV to {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
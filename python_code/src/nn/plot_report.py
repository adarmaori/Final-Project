from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORT_PATH = Path("data/processed/nn_latency_report.csv")
OUTPUT_DIR = Path("data/processed")
TARGET_SAMPLE_RATE = 44_100
RT_BUDGET_US = 1_000_000.0 / TARGET_SAMPLE_RATE

ARCH_COLORS = {
    "tcn": "#1f77b4",
    "lstm": "#d62728",
}


def main() -> None:
    df = pd.read_csv(REPORT_PATH, comment="#")
    required_columns = {
        "experiment",
        "model",
        "nmse_percent",
        "avg_batch_ms",
        "avg_sample_us",
        "samples_per_second",
        "chunk_size",
        "num_parameters",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Report is missing required columns: {sorted(missing)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    enriched_df = enrich_dataframe(df)
    enriched_path = OUTPUT_DIR / "nn_latency_report_enriched.csv"
    enriched_df.to_csv(enriched_path, index=False)

    architecture_summary = summarize_by_architecture(enriched_df)
    architecture_summary_path = OUTPUT_DIR / "nn_architecture_summary.csv"
    architecture_summary.to_csv(architecture_summary_path, index=False)

    top_candidates = rank_top_candidates(enriched_df, top_k=12)
    top_candidates_path = OUTPUT_DIR / "nn_top_candidates.csv"
    top_candidates.to_csv(top_candidates_path, index=False)

    save_nmse_vs_rt_load_plot(
        df=enriched_df,
        output_path=OUTPUT_DIR / "presentation_nmse_vs_rt_load.png",
    )
    save_params_tradeoff_plot(
        df=enriched_df,
        output_path=OUTPUT_DIR / "presentation_params_tradeoffs.png",
    )
    save_architecture_boxplots(
        df=enriched_df,
        output_path=OUTPUT_DIR / "presentation_architecture_boxplots.png",
    )
    save_pareto_plot(
        df=enriched_df,
        output_path=OUTPUT_DIR / "presentation_pareto_frontier.png",
    )
    write_presentation_summary(
        df=enriched_df,
        architecture_summary=architecture_summary,
        top_candidates=top_candidates,
        output_path=OUTPUT_DIR / "nn_presentation_summary.md",
    )

    save_scatter_plot(
        df=enriched_df,
        x_column="avg_batch_ms",
        y_column="nmse_percent",
        x_label="Average Batch Latency (ms)",
        y_label="Normalized MSE (%)",
        title="NMSE vs Latency",
        output_path=OUTPUT_DIR / "nmse_vs_latency.png",
    )
    save_scatter_plot(
        df=enriched_df,
        x_column="num_parameters",
        y_column="nmse_percent",
        x_label="Parameter Count",
        y_label="Normalized MSE (%)",
        title="NMSE vs Parameter Count",
        output_path=OUTPUT_DIR / "nmse_vs_parameter_count.png",
    )
    save_scatter_plot(
        df=enriched_df,
        x_column="num_parameters",
        y_column="avg_batch_ms",
        x_label="Parameter Count",
        y_label="Average Batch Latency (ms)",
        title="Latency vs Parameter Count",
        output_path=OUTPUT_DIR / "latency_vs_parameter_count.png",
    )

    print("Saved plots:")
    print(f"- {OUTPUT_DIR / 'nmse_vs_latency.png'}")
    print(f"- {OUTPUT_DIR / 'nmse_vs_parameter_count.png'}")
    print(f"- {OUTPUT_DIR / 'latency_vs_parameter_count.png'}")
    print(f"- {OUTPUT_DIR / 'presentation_nmse_vs_rt_load.png'}")
    print(f"- {OUTPUT_DIR / 'presentation_params_tradeoffs.png'}")
    print(f"- {OUTPUT_DIR / 'presentation_architecture_boxplots.png'}")
    print(f"- {OUTPUT_DIR / 'presentation_pareto_frontier.png'}")
    print("Saved analysis tables:")
    print(f"- {enriched_path}")
    print(f"- {architecture_summary_path}")
    print(f"- {top_candidates_path}")
    print(f"- {OUTPUT_DIR / 'nn_presentation_summary.md'}")


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["model"] = enriched["model"].astype(str).str.lower()
    enriched["model_size_mb_fp32"] = enriched["num_parameters"] * 4.0 / (1024.0 * 1024.0)
    enriched["realtime_budget_us_per_sample"] = RT_BUDGET_US
    enriched["realtime_load_pct"] = 100.0 * enriched["avg_sample_us"] / RT_BUDGET_US
    enriched["headroom_x"] = RT_BUDGET_US / enriched["avg_sample_us"].clip(lower=1e-12)
    enriched["max_realtime_sample_rate_hz"] = 1_000_000.0 / enriched["avg_sample_us"].clip(lower=1e-12)
    chunk_budget_ms = 1000.0 * enriched["chunk_size"] / TARGET_SAMPLE_RATE
    enriched["chunk_budget_ms_at_44k"] = chunk_budget_ms
    enriched["chunk_budget_utilization_pct_at_44k"] = 100.0 * enriched["avg_batch_ms"] / chunk_budget_ms
    enriched["is_realtime_viable_44k"] = enriched["realtime_load_pct"] <= 100.0
    enriched["viability_tier"] = pd.cut(
        enriched["realtime_load_pct"],
        bins=[-np.inf, 50.0, 100.0, np.inf],
        labels=["strong_headroom", "viable", "not_viable"],
    ).astype(str)
    return enriched


def summarize_by_architecture(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("model", dropna=False)
        .agg(
            models=("experiment", "count"),
            nmse_median=("nmse_percent", "median"),
            nmse_best=("nmse_percent", "min"),
            nmse_p90=("nmse_percent", lambda x: float(np.percentile(x, 90))),
            params_median=("num_parameters", "median"),
            sample_us_median=("avg_sample_us", "median"),
            sample_us_p90=("avg_sample_us", lambda x: float(np.percentile(x, 90))),
            load_pct_median=("realtime_load_pct", "median"),
            load_pct_p90=("realtime_load_pct", lambda x: float(np.percentile(x, 90))),
            viable_models_44k=("is_realtime_viable_44k", "sum"),
        )
        .reset_index()
    )
    grouped["viable_share_44k_pct"] = 100.0 * grouped["viable_models_44k"] / grouped["models"]
    return grouped.sort_values("nmse_median")


def rank_top_candidates(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    candidates = df.copy()
    nmse_scale = max(candidates["nmse_percent"].max(), 1e-12)
    load_scale = max(candidates["realtime_load_pct"].max(), 1e-12)
    candidates["presentation_score"] = (
        0.65 * (candidates["nmse_percent"] / nmse_scale)
        + 0.35 * (candidates["realtime_load_pct"] / load_scale)
    )

    ranked = candidates.sort_values(
        ["is_realtime_viable_44k", "presentation_score", "nmse_percent", "avg_sample_us"],
        ascending=[False, True, True, True],
    )
    keep_columns = [
        "experiment",
        "model",
        "nmse_percent",
        "avg_sample_us",
        "avg_batch_ms",
        "num_parameters",
        "model_size_mb_fp32",
        "realtime_load_pct",
        "headroom_x",
        "is_realtime_viable_44k",
        "viability_tier",
        "presentation_score",
    ]
    return ranked[keep_columns].head(top_k)


def save_nmse_vs_rt_load_plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    for architecture, group in df.groupby("model"):
        sizes = 30.0 + 120.0 * np.sqrt(group["model_size_mb_fp32"].clip(lower=0.0) + 0.02)
        ax.scatter(
            group["realtime_load_pct"],
            group["nmse_percent"],
            s=sizes,
            label=architecture.upper(),
            color=ARCH_COLORS.get(architecture, "#7f7f7f"),
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
        )

    ax.axvline(100.0, color="#b22222", linestyle="--", linewidth=1.4, label="44.1 kHz real-time limit")
    ax.axvline(50.0, color="#ff8c00", linestyle=":", linewidth=1.2, label="50% load headroom")
    ax.set_title("Sound Quality vs Real-Time Load (Bubble size = model size)")
    ax.set_xlabel("Real-Time Processing Load at 44.1 kHz (%)")
    ax.set_ylabel("Normalized MSE (%)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_params_tradeoff_plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))

    for architecture, group in df.groupby("model"):
        color = ARCH_COLORS.get(architecture, "#7f7f7f")
        ax1.scatter(
            group["num_parameters"],
            group["nmse_percent"],
            label=architecture.upper(),
            color=color,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.35,
            s=58,
        )
        ax2.scatter(
            group["num_parameters"],
            group["realtime_load_pct"],
            label=architecture.upper(),
            color=color,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.35,
            s=58,
        )

    ax1.set_title("Model Size vs Sound Quality")
    ax1.set_xlabel("Parameter Count")
    ax1.set_ylabel("Normalized MSE (%)")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Model Size vs Real-Time Load")
    ax2.set_xlabel("Parameter Count")
    ax2.set_ylabel("Processing Load at 44.1 kHz (%)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.axhline(100.0, color="#b22222", linestyle="--", linewidth=1.2)
    ax2.grid(True, alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=True)
    fig.suptitle("Architecture Trade-Offs: Size, Quality, and Viability", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_architecture_boxplots(df: pd.DataFrame, output_path: Path) -> None:
    architectures = sorted(df["model"].dropna().unique())
    nmse_data = [df.loc[df["model"] == arch, "nmse_percent"].values for arch in architectures]
    load_data = [df.loc[df["model"] == arch, "realtime_load_pct"].values for arch in architectures]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 9.0))
    bp_nmse = ax1.boxplot(
        nmse_data,
        tick_labels=[arch.upper() for arch in architectures],
        patch_artist=True,
    )
    bp_load = ax2.boxplot(
        load_data,
        tick_labels=[arch.upper() for arch in architectures],
        patch_artist=True,
    )

    for patch, arch in zip(bp_nmse["boxes"], architectures, strict=False):
        patch.set_facecolor(ARCH_COLORS.get(arch, "#cccccc"))
        patch.set_alpha(0.5)
    for patch, arch in zip(bp_load["boxes"], architectures, strict=False):
        patch.set_facecolor(ARCH_COLORS.get(arch, "#cccccc"))
        patch.set_alpha(0.5)

    ax1.set_title("NMSE Distribution by Architecture")
    ax1.set_ylabel("Normalized MSE (%)")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.set_title("Real-Time Load Distribution by Architecture")
    ax2.set_ylabel("Processing Load at 44.1 kHz (%)")
    ax2.axhline(100.0, color="#b22222", linestyle="--", linewidth=1.2, label="44.1 kHz limit")
    ax2.set_yscale("log")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()

    fig.suptitle("Architecture-Level Robustness", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_pareto_plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    for architecture, group in df.groupby("model"):
        color = ARCH_COLORS.get(architecture, "#7f7f7f")
        ax.scatter(
            group["avg_sample_us"],
            group["nmse_percent"],
            color=color,
            alpha=0.6,
            s=50,
            edgecolors="black",
            linewidths=0.35,
            label=f"{architecture.upper()} candidates",
        )

        frontier = pareto_frontier(group, x_col="avg_sample_us", y_col="nmse_percent")
        ax.plot(
            frontier["avg_sample_us"],
            frontier["nmse_percent"],
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=4,
            label=f"{architecture.upper()} Pareto",
        )

    ax.axvline(RT_BUDGET_US, color="#b22222", linestyle="--", linewidth=1.3, label="44.1 kHz sample budget")
    ax.set_xscale("log")
    ax.set_title("Pareto Frontier: Per-Sample Latency vs Sound Quality")
    ax.set_xlabel("Average Compute Time per Sample (microseconds)")
    ax.set_ylabel("Normalized MSE (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    ordered = df.sort_values(x_col)
    best_y = float("inf")
    selected_indices: list[int] = []
    for idx, row in ordered.iterrows():
        current_y = float(row[y_col])
        if current_y < best_y:
            selected_indices.append(idx)
            best_y = current_y
    return ordered.loc[selected_indices]


def write_presentation_summary(
    df: pd.DataFrame,
    architecture_summary: pd.DataFrame,
    top_candidates: pd.DataFrame,
    output_path: Path,
) -> None:
    total_models = len(df)
    viable_models = int(df["is_realtime_viable_44k"].sum())
    viable_share = 100.0 * viable_models / max(total_models, 1)

    best_nmse_row = df.sort_values("nmse_percent").iloc[0]
    fastest_row = df.sort_values("avg_sample_us").iloc[0]

    lines = [
        "# NN Presentation Summary",
        "",
        f"- Total evaluated models: {total_models}",
        f"- Real-time viable at 44.1 kHz (avg-sample metric): {viable_models} ({viable_share:.1f}%)",
        f"- Best NMSE model: {best_nmse_row['experiment']} ({best_nmse_row['nmse_percent']:.4f}%)",
        f"- Fastest model: {fastest_row['experiment']} ({fastest_row['avg_sample_us']:.4f} us/sample)",
        "",
        "## Architecture Summary (median-centric)",
        "",
    ]

    lines.extend(format_markdown_table(architecture_summary))
    lines.append("")
    lines.append("## Top Candidates (balanced score)")
    lines.append("")
    lines.extend(format_markdown_table(top_candidates))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- Real-time viability threshold uses {TARGET_SAMPLE_RATE} Hz sample budget ({RT_BUDGET_US:.4f} us/sample).")
    lines.append("- Chunk-level latency is shown via chunk budget utilization; per-sample load is preferred for cross-architecture viability comparisons.")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def format_markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No data."]

    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")

    headers = [str(column) for column in display_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display_df.iterrows():
        row_values = [str(row[column]) for column in display_df.columns]
        lines.append("| " + " | ".join(row_values) + " |")
    return lines


def save_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for architecture, group in df.groupby("model"):
        color = ARCH_COLORS.get(architecture, "#7f7f7f")
        ax.scatter(
            group[x_column],
            group[y_column],
            label=architecture.upper(),
            color=color,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.4,
            s=55,
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

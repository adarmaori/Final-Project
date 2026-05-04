from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPORT_PATH = Path("data/processed/nn_latency_report.csv")
OUTPUT_DIR = Path("data/processed")

ARCH_COLORS = {
    "tcn": "#1f77b4",
    "lstm": "#d62728",
}


def main() -> None:
    df = pd.read_csv(REPORT_PATH, comment="#")
    required_columns = {"model", "nmse_percent", "avg_batch_ms", "num_parameters"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Report is missing required columns: {sorted(missing)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_scatter_plot(
        df=df,
        x_column="avg_batch_ms",
        y_column="nmse_percent",
        x_label="Average Batch Latency (ms)",
        y_label="Normalized MSE (%)",
        title="NMSE vs Latency",
        output_path=OUTPUT_DIR / "nmse_vs_latency.png",
    )
    save_scatter_plot(
        df=df,
        x_column="num_parameters",
        y_column="nmse_percent",
        x_label="Parameter Count",
        y_label="Normalized MSE (%)",
        title="NMSE vs Parameter Count",
        output_path=OUTPUT_DIR / "nmse_vs_parameter_count.png",
    )
    save_scatter_plot(
        df=df,
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

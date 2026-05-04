from __future__ import annotations

from itertools import product

from src.nn.dataset import DatasetConfig
from src.nn.evaluator import EvaluationConfig, evaluate_experiments
from src.nn.trainer import ExperimentConfig, ModelConfig, TrainingConfig, run_experiment


DATASET = DatasetConfig(
    data_root="data/datasets",
    sample_rate=44_100,
    chunk_size=16_384,
    overlap=0.5,
    normalize=False,
)


TRAINING = TrainingConfig(
    epochs=50,
    batch_size=16,
    learning_rate=1e-3,
    validation_split=0.2,
    checkpoint_root="models/checkpoints",
    save_every=10,
)


MODEL_SEARCH_SPACES = {
    "tcn": {
        "input_channels": [1],
        "output_channels": [1],
        "hidden_channels": [8, 16, 32],
        "kernel_size": [3, 5, 7],
        "dilation": [1, 2],
        "num_blocks": [1, 2, 3],
    },
    "lstm": {
        "input_size": [1],
        "hidden_size": [8, 16, 32, 64],
        "output_size": [1],
        "num_layers": [1, 2, 3],
    },
}


ACTIVE_MODEL_FAMILIES = [
    "tcn",
    "lstm",
]


EVALUATION = EvaluationConfig(
    batch_size=1,
    num_workers=0,
    warmup_batches=3,
    timed_batches=20,
    device=None,
    csv_path="data/processed/nn_latency_report.csv",
    description="Post-training latency and NMSE evaluation for selected audio-effect network experiments",
)


def main() -> None:
    experiments = build_experiments(
        model_families=ACTIVE_MODEL_FAMILIES,
        search_spaces=MODEL_SEARCH_SPACES,
        dataset=DATASET,
        training=TRAINING,
    )

    print(f"Generated {len(experiments)} experiments")

    summaries = []
    for experiment_name, config in experiments.items():
        summaries.append(run_experiment(config))

    evaluation_results = evaluate_experiments(summaries, experiments, EVALUATION)

    print("\nExperiment summary")
    for summary in summaries:
        print(
            f"- {summary['name']}: best_val_loss={summary['best_val_loss']:.6f} "
            f"run_dir={summary['run_dir']}"
        )

    print("\nLatency summary")
    for result in evaluation_results:
        print(
            f"- {result['experiment']}: avg_batch_ms={result['avg_batch_ms']:.4f} "
            f"avg_sample_us={result['avg_sample_us']:.4f} "
            f"nmse_percent={result['nmse_percent']:.4f} "
            f"samples_per_second={result['samples_per_second']:.1f}"
        )


def build_experiments(
    model_families: list[str],
    search_spaces: dict[str, dict[str, list[int]]],
    dataset: DatasetConfig,
    training: TrainingConfig,
) -> dict[str, ExperimentConfig]:
    experiments: dict[str, ExperimentConfig] = {}

    for family in model_families:
        family_space = search_spaces.get(family)
        if family_space is None:
            raise ValueError(f"Missing search space for model family: {family}")

        for kwargs in expand_search_space(family_space):
            experiment_name = build_experiment_name(family, kwargs)
            experiments[experiment_name] = ExperimentConfig(
                name=experiment_name,
                model=ModelConfig(name=family, kwargs=kwargs),
                dataset=dataset,
                training=training,
            )

    return experiments


def expand_search_space(search_space: dict[str, list[int]]) -> list[dict[str, int]]:
    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]
    return [dict(zip(keys, combo, strict=False)) for combo in product(*values)]


def build_experiment_name(model_family: str, kwargs: dict[str, int]) -> str:
    suffix = "_".join(f"{key}-{value}" for key, value in kwargs.items())
    return f"{model_family}_{suffix}"


if __name__ == "__main__":
    main()

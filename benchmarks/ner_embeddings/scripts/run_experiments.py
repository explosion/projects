import shlex
import subprocess
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import srsly
import typer
from wasabi import msg

from .constants import DATASET_VECTORS

Arg = typer.Argument
Opt = typer.Option

app = typer.Typer()


class StaticVectors(str, Enum):
    spacy = "spacy"
    fasttext = "fasttext"
    null = "null"


def _run_commands(cmds: List[str], dry_run: bool = False):
    """Run a set of commands in order"""
    for cmd in cmds:
        _cmd = shlex.split(cmd)
        if dry_run:
            print(cmd.strip())
        else:
            subprocess.run(_cmd)


def _make_train_command(
    dataset: str,
    config: str,
    lang: str,
    vectors: str,
    gpu_id: int,
    seed: int,
    include_static_vectors: bool,
    adjust_rows: bool = False,
    adjust_value: int = 1,
    tables_path: str = "tables",
    batch_size: int = 1000,
    num_hash: Optional[int] = None,
) -> str:
    """Construct train command based from a template"""
    cmd_vectors = ""
    cmd_rows = ""
    cmd_hash = ""
    modifier = ""

    if not include_static_vectors:
        cmd_vectors = "--vars.include_static_vectors false"
    if adjust_rows and config != "multiembed":
        modifier = "-custom-rows"
        new_rows = _get_computed_rows(tables_path, dataset, adjust_value)
        cmd_rows = f"--vars.rows '{new_rows}'"
    if num_hash:
        modifier = "-hash"
        cmd_hash = f"--vars.num_hashes {num_hash}"

    command = (
        f"spacy project run train{modifier} .  "
        f"--vars.dataset {dataset} "
        f"--vars.ner_config {config} "
        f"--vars.language {lang} "
        f"--vars.vectors {vectors} "
        f"--vars.gpu-id {gpu_id} "
        f"--vars.seed {seed} "
        f"--vars.tables_path {tables_path} "
        f"--vars.batch_size {batch_size} "
        f"{cmd_vectors} {cmd_rows} {cmd_hash}"
    )
    return command


def _format_attrs(attrs: List[str]) -> str:
    s = ""
    for idx, attr in enumerate(attrs):
        s += f'"{attr}"'
        if idx != len(attrs) - 1:
            s += ", "
    cmd_attrs = f"{s}"
    return cmd_attrs


def _get_computed_rows(tables_path: str, dataset: str, adjust_value: int) -> List:
    attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
    table_filepath = f"{tables_path}/{dataset}/{dataset}-train.tables"
    table = srsly.read_msgpack(table_filepath)
    attr_sizes = {attr: int(len(table[attr]) / adjust_value) for attr in attrs}
    msg.text(f"Adjusted to {attr_sizes}")
    return list(attr_sizes.values())


def _make_eval_command(
    dataset: str,
    config: str,
    gpu_id: int,
    vectors: str,
    seed: int,
    metrics_dir: str,
    eval_unseen: bool = False,
) -> str:
    """Construct eval command based from a template"""
    unseen_param = "-seen-unseen" if eval_unseen else ""
    command = f"""
    spacy project run evaluate{unseen_param} .
    --vars.ner_config {config}
    --vars.dataset {dataset}
    --vars.gpu-id {gpu_id}
    --vars.vectors {vectors}
    --vars.seed {seed}
    --vars.metrics_dir {metrics_dir}
    """
    return command


def _make_tables_command(
    config: str,
    dataset: str,
    min_freq: int = 10,
    tables_path: str = "tables",
):
    """Construct the make-tables command based on a template"""
    command = f"""
    spacy project run make-tables .
    --vars.ner_config {config}
    --vars.min_freq {min_freq}
    --vars.dataset {dataset}
    --vars.tables_path {tables_path}
    """
    return command


def _get_datasets(datasets: Optional[List[str]]) -> Dict:
    if datasets:
        dataset_vectors = {k: v for k, v in DATASET_VECTORS.items() if k in datasets}
    else:
        dataset_vectors = DATASET_VECTORS

    msg.info(f"Retrieving datasets: {', '.join(dataset_vectors.keys())}")
    return dataset_vectors


@app.command(name="main-results")
def run_main_results(
    # fmt: off
    datasets: Optional[List[str]] = Arg(None, help="Datasets to run the experiment on. If None is passed, then experiment is ran on all datasets.", show_default=True),
    config: str = Opt("multihashembed", help="The spaCy configuration file to use for training."),
    static_vectors: StaticVectors = Opt("null", help="Type of static vectors to use.", show_default=True),
    adjust_rows: bool = Opt(False, "--adjust-rows", help="Adjust the rows for MultiHashEmbed based on computed hash tables."),
    adjust_value: int = Opt(1, "--adjust-value", help="Setup how much should the rows be adjusted"),
    gpu_id: int = Opt(0, help="Set the random seed.", show_default=True),
    dry_run: bool = Opt(False, "--dry-run", help="Print the commands, don't run them."),
    eval_unseen: bool = Opt(False, "--eval-unseen", help="Evaluate on unseen entities."),
    batch_size: int = Opt(1000, "--batch-size", "-S", "--sz", help="Set the batch size.", show_default=True),
    seed: int = Opt(0, help="Set the random seed", show_default=True),
    experiment_id: str = Opt("main_results", "--experiment-id", "--id", help="Experiment ID for saving the metrics.", show_default=True),
    # fmt: on
):
    """Run experiment that compares MultiEmbed and MultiHashEmbed (default rows)"""
    msg.info("Running experiment that compares MultiEmbed and MultiHashEmbed")
    dataset_vectors = _get_datasets(datasets)
    for dataset, vectors in dataset_vectors.items():
        msg.divider(dataset, char="X")
        commands = []

        # Create hash tables
        if config == "multiembed":
            hash_command = _make_tables_command(
                config=config,
                dataset=dataset,
            )
            commands.append(hash_command)

        # Train command
        train_command = _make_train_command(
            dataset,
            config,
            lang=vectors.get("lang"),
            vectors=vectors.get(static_vectors.value, StaticVectors.null),
            gpu_id=gpu_id,
            seed=seed,
            adjust_rows=adjust_rows,
            adjust_value=adjust_value,
            batch_size=batch_size,
            include_static_vectors=static_vectors.value != StaticVectors.null,
        )
        commands.append(train_command)

        # Evaluate command
        eval_command = _make_eval_command(
            dataset=dataset,
            config=config,
            gpu_id=gpu_id,
            vectors=vectors.get(static_vectors.value, StaticVectors.null),
            seed=seed,
            metrics_dir=f"metrics-{experiment_id}",
            eval_unseen=eval_unseen,
        )
        commands.append(eval_command)

        # Run commands
        _run_commands(cmds=commands, dry_run=dry_run)


@app.command(name="characterize-min-freq")
def run_multiembed_min_freq_experiment(
    # fmt: off
    datasets: Optional[List[str]] = Arg(None, help="Datasets to run the experiment on. If None is passed, then experiment is ran on all datasets.", show_default=True),
    config: str = Opt("multiembed", help="The spaCy configuration file to use for training."),
    static_vectors: StaticVectors = Opt("null", help="Type of static vectors to use.", show_default=True),
    min_freqs: Tuple[int, int , int] = Opt((1, 5, 10), help="Values to check min_freq for.", show_default=True),
    gpu_id: int = Opt(0, help="Set the random seed.", show_default=True),
    dry_run: bool = Opt(False, "--dry-run", help="Print the commands, don't run them."),
    batch_size: int = Opt(1000, "--batch-size", "-S", "--sz", help="Set the batch size.", show_default=True),
    seed: int = Opt(0, help="Set the random seed", show_default=True),
    experiment_id: str = Opt("multiembed_min_freq", "--experiment-id", "--id", help="Experiment ID for saving the metrics", show_default=True),
    # fmt: on
):
    """Run experiment that compares different MultiEmbed min_freq values"""
    msg.info("Running experiment for MultiEmbed with different min_freq")
    config_path = Path("configs") / config
    dataset_vectors = _get_datasets(datasets)
    # Create hash table in another directory
    for min_freq in min_freqs:
        msg.divider(f"min_freq={min_freq}", char="X")
        table_path = Path(f"tables_{min_freq}")
        table_path.mkdir(parents=True, exist_ok=True)

        for dataset, vectors in dataset_vectors.items():
            msg.divider(dataset, char="x")
            commands = []

            # Create hash tables
            hash_command = _make_tables_command(
                config=config,
                dataset=dataset,
                min_freq=min_freq,
                tables_path=str(table_path),
            )
            commands.append(hash_command)

            # Train command
            train_command = _make_train_command(
                dataset,
                config,
                lang=vectors.get("lang"),
                vectors=vectors.get(static_vectors.value, StaticVectors.null),
                gpu_id=gpu_id,
                seed=seed,
                batch_size=batch_size,
                include_static_vectors=static_vectors.value != StaticVectors.null,
                tables_path=str(table_path),
            )
            commands.append(train_command)

            # Evaluate command
            eval_command = _make_eval_command(
                dataset=dataset,
                config=config,
                gpu_id=gpu_id,
                vectors=vectors.get(static_vectors.value, StaticVectors.null),
                seed=seed,
                metrics_dir=f"metrics-{experiment_id}-{min_freq}",
            )
            commands.append(eval_command)

            _run_commands(
                cmds=commands,
                dry_run=dry_run,
            )


@app.command(name="feature-ablation")
def run_multiembed_features_ablation(
    # fmt: off
    datasets: Optional[List[str]] = Arg(None, help="Datasets to run the experiment on. If None is passed, then experiment is ran on all datasets.", show_default=True),
    static_vectors: StaticVectors = Opt("null", help="Type of static vectors to use.", show_default=True),
    gpu_id: int = Opt(0, help="Set the random seed.", show_default=True),
    dry_run: bool = Opt(False, "--dry-run", help="Print the commands, don't run them."),
    eval_unseen: bool = Opt(False, "--eval-unseen", help="Evaluate on unseen entities."),
    seed: int = Opt(0, help="Set the random seed", show_default=True),
    batch_size: int = Opt(1000, "--batch-size", "-S", "--sz", help="Set the batch size.", show_default=True),
    experiment_id: str = Opt("feature_ablation", "--experiment-id", "--id", help="Experiment ID for saving the metrics", show_default=True),
    # fmt: on
):
    """Run ablation experiment for MultiEmbed features"""
    dataset_vectors = _get_datasets(datasets)
    attr_combinations = {
        # fmt: off
        "ablation/multihashembed_orth": ["ORTH"],
        "ablation/multihashembed_norm": ["NORM"],
        "ablation/multihashembed_norm_prefix": ["NORM", "PREFIX"],
        "ablation/multihashembed_norm_prefix_suffix": ["NORM", "PREFIX", "SUFFIX"],
        "ablation/multihashembed_norm_prefix_suffix_shape": ["NORM", "PREFIX", "SUFFIX", "SHAPE"],
        # fmt: on
    }

    for config, attrs in attr_combinations.items():
        for dataset, vectors in dataset_vectors.items():
            msg.divider(dataset, char="X")
            commands = []

            # Create hash tables
            if "multiembed" in config:
                hash_command = _make_tables_command(
                    config=config,
                    dataset=dataset,
                )
                commands.append(hash_command)

            # Train command
            train_command = _make_train_command(
                dataset,
                config,
                lang=vectors.get("lang"),
                vectors=vectors.get(static_vectors.value, StaticVectors.null),
                gpu_id=gpu_id,
                seed=seed,
                batch_size=batch_size,
                include_static_vectors=static_vectors.value != StaticVectors.null,
            )
            commands.append(train_command)

            # Evaluate command
            eval_command = _make_eval_command(
                dataset=dataset,
                config=config,
                gpu_id=gpu_id,
                vectors=vectors.get(static_vectors.value, StaticVectors.null),
                seed=seed,
                metrics_dir=f"metrics-{experiment_id}-{'-'.join(attrs)}",
                eval_unseen=eval_unseen,
            )
            commands.append(eval_command)

            # Run commands
            _run_commands(cmds=commands, dry_run=dry_run)


@app.command(name="characterize-hash")
def run_characterize_hash(
    # fmt: off
    datasets: Optional[List[str]] = Arg(None, help="Datasets to run the experiment on. If None is passed, then experiment is ran on all datasets.", show_default=True),
    config: str = Opt("multifewerhashembed", help="Name of the MultiHashFewerEmbed config", show_default=True),
    static_vectors: StaticVectors = Opt("null", help="Type of static vectors to use.", show_default=True),
    gpu_id: int = Opt(0, help="Set the GPU ID.", show_default=True),
    dry_run: bool = Opt(False, "--dry-run", help="Print the commands, don't run them."),
    seed: int = Opt(0, help="Set the random seed", show_default=True),
    batch_size: int = Opt(1000, "--batch-size", "-S", "--sz", help="Set the batch size.", show_default=True),
    experiment_id: str = Opt("characterize_hash", "--experiment-id", "--id", help="Experiment ID for saving the metrics", show_default=True),
    # fmt: on
):
    dataset_vectors = _get_datasets(datasets)
    num_hashes = [1, 2, 3, 4]

    for num_hash in num_hashes:
        for dataset, vectors in dataset_vectors.items():
            msg.divider(dataset, char="X")
            commands = []

            # Train command
            train_command = _make_train_command(
                dataset,
                config,
                lang=vectors.get("lang"),
                vectors=vectors.get(static_vectors.value, StaticVectors.null),
                gpu_id=gpu_id,
                seed=num_hash,
                batch_size=batch_size,
                include_static_vectors=static_vectors.value != StaticVectors.null,
                num_hash=num_hash,
            )
            commands.append(train_command)

            # Evaluate command
            eval_command = _make_eval_command(
                dataset=dataset,
                config=config,
                gpu_id=gpu_id,
                vectors=vectors.get(static_vectors.value, StaticVectors.null),
                seed=num_hash,
                metrics_dir=f"metrics-{experiment_id}-{num_hash}",
                eval_unseen=False,
            )
            commands.append(eval_command)

            # Run commands
            _run_commands(cmds=commands, dry_run=dry_run)


if __name__ == "__main__":
    app()

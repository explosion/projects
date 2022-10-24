import shlex
import subprocess
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

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
    tables_path: str = "tables",
    custom_attrs: Optional[List[str]] = None,
) -> str:
    """Construct train command based from a template"""
    cmd_vectors = ""
    cmd_rows = ""
    modifier = ""

    if not include_static_vectors:
        cmd_vectors = "--vars.include_static_vectors false"
    if adjust_rows and config != "ner_multiembed":
        modifier = "-custom-rows"
        new_rows = _get_computed_rows(tables_path, dataset)
        cmd_rows = f"--vars.rows '{new_rows}'"

    command = (
        f"spacy project run train{modifier} .  "
        f"--vars.dataset {dataset} "
        f"--vars.ner_config {config} "
        f"--vars.language {lang} "
        f"--vars.vectors {vectors} "
        f"--vars.gpu-id {gpu_id} "
        f"--vars.seed {seed} "
        f"--vars.tables_path {tables_path} "
        f"{cmd_vectors} {cmd_rows}"
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


def _get_computed_rows(tables_path: str, dataset: str) -> List:
    attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
    table_filepath = f"{tables_path}/{dataset}/{dataset}-train.tables"
    table = srsly.read_msgpack(table_filepath)
    attr_sizes = {attr: len(table[attr]) for attr in attrs}
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
    unseen_param = "-unseen" if eval_unseen else ""
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


def _make_hash_command(
    config: str,
    dataset: str,
    min_freq: int = 10,
    tables_path: str = "tables",
):
    """Construct the hash command based from a template"""
    command = f"""
    spacy project run make-tables .
    --vars.ner_config {config}
    --vars.min_freq {min_freq}
    --vars.dataset {dataset}
    --vars.tables_path {tables_path}
    """
    return command


@app.command(name="main-results")
def run_main_results(
    # fmt: off
    config: str = Opt("ner_multihashembed", help="The spaCy configuration file to use for training."),
    static_vectors: StaticVectors = Opt("null", help="Type of static vectors to use.", show_default=True),
    adjust_rows: bool = Opt(False, "--adjust-rows", help="Adjust the rows for MultiHashEmbed based on computed hash tables"),
    gpu_id: int = Opt(0, help="Set the random seed.", show_default=True),
    dry_run: bool = Opt(False, "--dry-run", help="Print the commands, don't run them."),
    eval_unseen: bool = Opt(False, "--eval-unseen", help="Evaluate on unseen entities."),
    seed: int = Opt(0, help="Set the random seed", show_default=True),
    # fmt: on
):
    """Run experiment that compares MultiEmbed and MultiHashEmbed (default rows)"""
    EXPERIMENT_ID = "main_results"
    msg.info("Running experiment that compares MultiEmbed and MultiHashEmbed")
    for dataset, vectors in DATASET_VECTORS.items():
        msg.divider(dataset, char="X")
        commands = []

        # Create hash tables
        if config == "ner_multiembed":
            hash_command = _make_hash_command(
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
            metrics_dir=f"metrics-{EXPERIMENT_ID}",
            eval_unseen=eval_unseen,
        )
        commands.append(eval_command)

        # Run commands
        _run_commands(cmds=commands, dry_run=dry_run)


@app.command(name="characterize-min-freq")
def run_multiembed_min_freq_experiment(
    # fmt: off
    config: str = Opt("ner_multiembed", help="The spaCy configuration file to use for training."),
    static_vectors: StaticVectors = Opt("null", help="Type of static vectors to use.", show_default=True),
    min_freqs: Tuple[int, int , int] = Opt((1, 5, 10), help="Values to check min_freq for.", show_default=True),
    gpu_id: int = Opt(0, help="Set the random seed.", show_default=True),
    dry_run: bool = Opt(False, "--dry-run", help="Print the commands, don't run them."),
    seed: int = Opt(0, help="Set the random seed", show_default=True),
    # fmt: on
):
    """Run experiment that compares different MultiEmbed min_freq values"""
    EXPERIMENT_ID = "multiembed_min_freq"
    msg.info("Running experiment for MultiEmbed with different min_freq")
    config_path = Path("configs") / config

    # Create hash table in another directory
    for min_freq in min_freqs:
        msg.divider(f"min_freq={min_freq}", char="X")
        table_path = Path(f"tables_{min_freq}")
        table_path.mkdir(parents=True, exist_ok=True)

        for dataset, vectors in DATASET_VECTORS.items():
            msg.divider(dataset, char="x")
            commands = []

            # Create hash tables
            hash_command = _make_hash_command(
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
                metrics_dir=f"metrics-{EXPERIMENT_ID}-{min_freq}",
            )
            commands.append(eval_command)

            _run_commands(
                cmds=commands,
                dry_run=dry_run,
            )


@app.command(name="feature-ablation")
def run_multiembed_features_ablation(
    # fmt: off
    static_vectors: StaticVectors = Opt("null", help="Type of static vectors to use.", show_default=True),
    gpu_id: int = Opt(0, help="Set the random seed.", show_default=True),
    dry_run: bool = Opt(False, "--dry-run", help="Print the commands, don't run them."),
    eval_unseen: bool = Opt(False, "--eval-unseen", help="Evaluate on unseen entities."),
    seed: int = Opt(0, help="Set the random seed", show_default=True),
    # fmt: on
):
    """Run ablation experiment for MultiEmbed features"""
    EXPERIMENT_ID = "multiembed_ablation"
    attr_combinations = {
        # fmt: off
        "ablation/ner_multihashembed_o": ["ORTH"],
        "ablation/ner_multihashembed_n": ["NORM"],
        "ablation/ner_multihashembed_np": ["NORM", "PREFIX"],
        "ablation/ner_multihashembed_nps": ["NORM", "PREFIX", "SUFFIX"],
        "ablation/ner_multihashembed_npss": ["NORM", "PREFIX", "SUFFIX", "SHAPE"],
        # fmt: on
    }

    for config, attrs in attr_combinations.items():
        for dataset, vectors in DATASET_VECTORS.items():
            msg.divider(dataset, char="X")
            commands = []

            # Create hash tables
            if "multiembed" in config:
                hash_command = _make_hash_command(
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
                custom_attrs=attrs,
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
                metrics_dir=f"metrics-{EXPERIMENT_ID}-{'-'.join(attrs)}",
                eval_unseen=eval_unseen,
            )
            commands.append(eval_command)

            # Run commands
            _run_commands(cmds=commands, dry_run=dry_run)


if __name__ == "__main__":
    app()

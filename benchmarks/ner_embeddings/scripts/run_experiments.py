import shlex
import subprocess
from typing import Literal, Optional, List

import typer
from wasabi import msg
from pathlib import Path

# A mapping of datasets and their vectors
DATASET_VECTORS = {
    "archaeo": {"spacy": "nl_core_news_lg", "fasttext": "fasttext-nl", "lang": "nl"},
    "anem": {"spacy": "en_core_web_lg", "fasttext": "fasttext-en", "lang": "en"},
    "conll-es": {"spacy": "es_core_news_lg", "fasttext": "fasttext-es", "lang": "es"},
    "conll-nl": {"spacy": "nl_core_news_lg", "fasttext": "fasttext-nl", "lang": "nl"},
}


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
    tables_path: str = "tables",
) -> str:
    """Construct train command based from a template"""
    cmd_vectors = (
        "--vars.include_static_vectors false" if not include_static_vectors else ""
    )
    command = f"""
    spacy project run train-ner . 
    --vars.dataset {dataset}
    --vars.ner_config {config} 
    --vars.language {lang}
    --vars.vectors {vectors}
    --vars.gpu-id {gpu_id}
    --vars.seed {seed}
    --vars.tables_path {tables_path}
    {cmd_vectors}
    """
    return command


def _make_eval_command(
    dataset: str, config: str, gpu_id: int, vectors: str, seed: int
) -> str:
    """Construct eval command based from a template"""
    command = f"""
    spacy project run evaluate-ner .
    --vars.ner_config {config}
    --vars.dataset {dataset}
    --vars.gpu-id {gpu_id}
    --vars.vectors {vectors}
    --vars.seed {seed}
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
    --vars.min-freq {min_freq}
    --vars.dataset {dataset}
    --vars.tables_path {tables_path}
    """
    return command


def run_main_results(
    config: str = "ner_multihashembed",
    static_vectors: Optional[Literal["spacy", "fasttext"]] = None,
    adjust_rows: bool = False,  # TODO
    gpu_id: int = 0,
    dry_run: bool = False,
    seed: int = 0,
):
    """Run experiment that compares MultiEmbed and MultiHashEmbed (default rows)"""
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
            vectors=vectors.get(static_vectors, "null"),
            gpu_id=gpu_id,
            seed=seed,
            include_static_vectors=bool(static_vectors),
        )
        commands.append(train_command)

        # Evaluate command
        eval_command = _make_eval_command(
            dataset=dataset,
            config=config,
            gpu_id=gpu_id,
            vectors=vectors.get(static_vectors, "null"),
            seed=seed,
        )
        commands.append(eval_command)

        # Run commands
        _run_commands(cmds=commands, dry_run=dry_run)


def run_multiembed_min_freq_experiment(
    config: str = "ner_multiembed",
    min_freqs: List[int] = [5, 10, 30],
    static_vectors: Optional[Literal["spacy", "fasttext"]] = None,
    gpu_id: int = 0,
    dry_run: bool = False,
    seed: int = 0,
):
    """Run experiment that compares different MultiEmbed min_freq values"""
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
                vectors=vectors.get(static_vectors, "null"),
                gpu_id=gpu_id,
                seed=seed,
                include_static_vectors=bool(static_vectors),
                tables_path=str(table_path),
            )
            commands.append(train_command)

            # Evaluate command
            eval_command = _make_eval_command(
                dataset=dataset,
                config=config,
                gpu_id=gpu_id,
                vectors=vectors.get(static_vectors, "null"),
                seed=seed,
            )
            commands.append(eval_command)

            _run_commands(
                cmds=commands,
                dry_run=dry_run,
            )


def run_multiembed_features_ablation():
    """Run ablation experiment for MultiEmbed features"""
    pass


if __name__ == "__main__":
    # run_main_results(dry_run=True)
    # run_main_results(static_vectors="spacy", dry_run=True)
    run_multiembed_min_freq_experiment(static_vectors="spacy", dry_run=True)

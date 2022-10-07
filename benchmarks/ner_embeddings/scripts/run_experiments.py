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

        # Train command
        _cmd_sv = "--vars.include_static_vectors false" if not static_vectors else ""
        _param_sv = "null" if not static_vectors else vectors.get(static_vectors)
        train_command = f"""
        spacy project run train-ner . 
        --vars.dataset {dataset}
        --vars.ner_config {config} 
        --vars.language {vectors.get("lang")}
        --vars.vectors {_param_sv}
        --vars.gpu-id {gpu_id}
        --vars.seed {seed}
        {_cmd_sv}
        """

        # Evaluate command
        eval_command = f"""
        spacy project run evaluate-ner .
        --vars.ner_config {config}
        --vars.dataset {dataset}
        --vars.gpu-id {gpu_id}
        --vars.vectors {_param_sv}
        --vars.seed {seed}
        """

        # Run commands
        _run_commands(cmds=[train_command, eval_command], dry_run=dry_run)


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

            # Create hash tables
            dataset_table = f"{str(table_path / dataset)}.table"
            token_map_command = f"""
            python scripts/token_map.py {str(config_path)} {dataset_table} 
            --min-freq {min_freq}
            --nlp.lang {vectors.get("lang")}
            --system.seed {seed}
            --training.train_corpus corpus/{dataset}
            """

            # Train command
            # fmt: off
            _cmd_sv = "--vars.include_static_vectors false" if not static_vectors else ""
            # fmt: on
            _param_sv = "null" if not static_vectors else vectors.get(static_vectors)
            train_command = f"""
            spacy project run train-ner . 
            --paths.tables {dataset_table}
            --vars.dataset {dataset}
            --vars.ner_config {config} 
            --vars.language {vectors.get("lang")}
            --vars.vectors {_param_sv}
            --vars.gpu-id {gpu_id}
            --vars.seed {seed}
            {_cmd_sv}
            """

            # Evaluate command
            eval_command = f"""
            spacy project run evaluate-ner .
            --vars.ner_config {config}
            --vars.dataset {dataset}
            --vars.gpu-id {gpu_id}
            --vars.vectors {_param_sv}
            --vars.seed {seed}
            """

            _run_commands(
                cmds=[token_map_command, train_command, eval_command],
                dry_run=dry_run,
            )


def run_multiembed_features_ablation():
    """Run ablation experiment for MultiEmbed features"""
    pass


# spacy project run train-ner . --vars.ner_config ner_multihashembed --vars.language nl
# --vars.dataset nl-conll --vars.vectors null --vars.gpu-id 0 --vars.seed {seed}
# --vars.include_static_vectors false

# spacy project run evaluate-ner . --vars.ner_config ner_multihashembed  --vars.dataset
# nl-conll --vars.gpu-id 0 --vars.vectors null --vars.seed {seed}

if __name__ == "__main__":
    pass
    # run_main_results(static_vectors="spacy", dry_run=True)
    # run_multiembed_min_freq_experiment(static_vectors="spacy", dry_run=True)

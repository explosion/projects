import shlex
import subprocess
from typing import Literal, Optional

import typer
from wasabi import msg

# A mapping of datasets and their vectors
DATASET_VECTORS = {
    "archaeo": {"spacy": "nl_core_news_lg", "fasttext": "fasttext-nl", "lang": "nl"},
    "anem": {"spacy": "en_core_web_lg", "fasttext": "fasttext-en", "lang": "en"},
    "conll-es": {"spacy": "es_core_news_lg", "fasttext": "fasttext-es", "lang": "es"},
    "conll-nl": {"spacy": "nl_core_news_lg", "fasttext": "fasttext-nl", "lang": "nl"},
}


def run_main_results(
    config: str = "ner_tok2vec",
    static_vectors: Optional[Literal["spacy", "fasttext"]] = None,
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

        if dry_run:
            print(train_command.strip())
            print(eval_command.strip())
        else:
            _train_cmd = shlex.split(train_command)
            subprocess.run(_train_cmd)
            _eval_cmd = shlex.split(eval_command)
            subprocess.run(_eval_cmd)


def run_multiembed_min_freq_experiment():
    """Run experiment that compares different MultiEmbed min_freq values"""
    pass


def run_multiembed_features_ablation():
    """Run ablation experiment for MultiEmbed features"""
    pass


# spacy project run train-ner . --vars.ner_config ner_tok2vec --vars.language nl
# --vars.dataset nl-conll --vars.vectors null --vars.gpu-id 0 --vars.seed {seed}
# --vars.include_static_vectors false

# spacy project run evaluate-ner . --vars.ner_config ner_tok2vec  --vars.dataset
# nl-conll --vars.gpu-id 0 --vars.vectors null --vars.seed {seed}

if __name__ == "__main__":
    run_main_results(static_vectors="spacy", dry_run=True)

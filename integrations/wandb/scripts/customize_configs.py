import typer
from pathlib import Path
from thinc.api import Config


def main(config_dir: Path, default_config: str):
    cfg = Config().from_disk(config_dir / default_config, interpolate=False)

    # task-specific settings
    cfg["components"]["textcat"]["positive_label"] = "pos"
    cfg["components"]["textcat"]["model"]["exclusive_classes"] = True

    # logging & evaluation
    cfg["training"]["max_steps"] = 2000
    cfg["training"]["eval_frequency"] = 100
    cfg["training"]["logger"] = {
        "@loggers": "spacy.WandbLogger.v1",
        "project_name": "IMDB_sentiment",
    }

    # corpus definition
    cfg["training"]["train_corpus"] = {
        "@readers": "ml_datasets.imdb_sentiment.v1",
        "path": "assets/aclImdb",
        "limit": 1000,
        "train": True,
    }
    cfg["training"]["dev_corpus"] = {
        "@readers": "ml_datasets.imdb_sentiment.v1",
        "path": "assets/aclImdb",
        "limit": 200,
        "train": False,
    }

    # hyperparameter grid search
    i = 1
    for depth in [2, 4]:
        for ngram in [1, 3]:
            for dropout in [0.05, 0.2]:
                for lr in [0.001, 0.01]:
                    cfg["training"]["dropout"] = dropout
                    cfg["training"]["optimizer"]["learn_rate"] = lr
                    cfg["components"]["textcat"]["model"]["ngram_size"] = ngram
                    cfg["components"]["tok2vec"]["model"]["encode"]["depth"] = depth
                    cfg.to_disk(config_dir / f"config_{i}.cfg")
                    i += 1


if __name__ == "__main__":
    typer.run(main)

import typer
from pathlib import Path
from thinc.api import Config


def main(config_dir: Path, default_config: str):
    cfg = Config().from_disk(config_dir / default_config, interpolate=False)
    # hyperparameter grid search
    i = 0
    for depth in [2, 4]:
        for ngram in [1, 3]:
            for dropout in [0.05, 0.2]:
                for lr in [0.001, 0.01]:
                    i += 1
                    cfg["training"]["dropout"] = dropout
                    cfg["training"]["optimizer"]["learn_rate"] = lr
                    cfg["components"]["textcat"]["model"]["ngram_size"] = ngram
                    cfg["components"]["tok2vec"]["model"]["encode"]["depth"] = depth
                    cfg.to_disk(config_dir / f"config_{i}.cfg")
    print(f"Saved {i} configs")


if __name__ == "__main__":
    typer.run(main)

from typing import Callable
from pathlib import Path


def create_logger(results_dir: Path) -> Callable:
    results_file = results_dir / "results.csv"
    with results_file.open("w", encoding="utf8") as f:
        f.write("name;gpu;articles;characters;words;seconds;k wps")
        f.write("\n")

    def log_result(
        name: str, gpu: bool, articles: int, characters: int, words: int, seconds: int
    ):
        wps = words / seconds
        wps = wps / 1000
        with results_file.open("a", encoding="utf8") as f:
            f.write(f"{name};{gpu};{articles};{characters};{words};{seconds};{wps}")
            f.write("\n")

    return log_result

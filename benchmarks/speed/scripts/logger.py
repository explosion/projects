from typing import Callable
from pathlib import Path
from datetime import datetime


def create_logger(results_dir: Path) -> Callable:
    results_file = results_dir / "results.csv"
    write_header = not results_file.exists()
    with results_file.open("a", encoding="utf8") as f:
        if write_header:
            f.write("library;name;gpu;articles;characters;words;seconds;k wps;time stamp")
            f.write("\n")

    def log_result(
        library: str, name: str, gpu: bool, articles: int, characters: int, words: int, seconds: int
    ):
        wps = words / seconds
        wps = wps / 1000

        result = f"{library};{name};{gpu};{articles};{characters};{words};{seconds};{wps};{datetime.now()}"
        print(result)
        with results_file.open("a", encoding="utf8") as f:
            f.write(result + "\n")

    return log_result

from typing import Callable
from pathlib import Path
from datetime import datetime
from wasabi import msg


def create_logger(results_dir: Path) -> Callable:
    results_file = results_dir / "results.csv"
    write_header = not results_file.exists()
    with results_file.open("a", encoding="utf8") as f:
        if write_header:
            f.write(
                "library;name;gpu;articles;characters;words;seconds;k wps;time stamp"
            )
            f.write("\n")

    header = ["Library", "Model", "GPU?", "# Texts", "# Chars", "# Words", "# Seconds", "W/S", "Timestamp"]
    widths = [max(15, len(head)) for head in header]
    widths[-1] = len(str(datetime.now().isoformat(timespec="seconds")))
    msg.row(header, widths=widths)
    def log_result(
        library: str,
        name: str,
        gpu: bool,
        articles: int,
        characters: int,
        words: int,
        seconds: int,
    ):
        wps = words / seconds
        wps = wps / 1000

        timestamp = datetime.now().isoformat(timespec="seconds")
        row = [library, name, gpu, articles, characters, words, int(seconds), "%.1fk" % wps, timestamp]
        msg.row(data=row, widths=widths)

        result = f"{library};{name};{gpu};{articles};{characters};{words};{seconds};{wps};{timestamp}"
        with results_file.open("a", encoding="utf8") as f:
            f.write(result + "\n")

    return log_result

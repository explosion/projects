from pathlib import Path

from wasabi import msg

DATA_DIR = Path(__file__).parent.parent / "assets"


def read_trial(pmid: int, verbose: bool = False) -> str:
    file_path = DATA_DIR / f"{pmid}.txt"
    msg.text(f"Reading article text from {file_path}", show=verbose)

    with open(file_path, "r", encoding="utf8") as file:
        data = file.read()

    return data

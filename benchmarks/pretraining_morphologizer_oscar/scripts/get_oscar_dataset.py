import typer
from pathlib import Path
import srsly
from datasets import load_dataset
from itertools import islice
from wasabi import msg


def main(max_texts: int, output_path: Path):
    """Uses the datasets API from HuggingFace to retrieve a set amount of data entries from the OSCAR corpus and saves it as a jsonl file"""

    msg.info(f"Start downloading {max_texts} data entries from the OSCAR corpus")
    dataset = load_dataset(
        "oscar", "unshuffled_deduplicated_en", split="train", streaming=True
    )
    data = [line for line in islice(iter(dataset), max_texts)]
    srsly.write_jsonl(output_path, data)
    msg.good(f"Saved data entries to {output_path}")


if __name__ == "__main__":
    typer.run(main)

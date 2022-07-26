import typer
from pathlib import Path
import srsly
from datasets import load_dataset
from itertools import islice
from wasabi import msg
from tqdm import tqdm


def main(max_texts: int, output_path: Path):
    """Uses the datasets API from HuggingFace to retrieve a set amount of data entries from the OSCAR corpus and saves it as a jsonl file"""

    msg.info(f"Start downloading {max_texts} data entries from the OSCAR corpus")
    dataset = load_dataset(
        "oscar", "unshuffled_deduplicated_en", split="train", streaming=True
    )
    data = []
    text_length = 0
    for line in tqdm(
        islice(iter(dataset), max_texts),
        total=max_texts,
        desc="Downloading OSCAR extract",
    ):
        data.append(line)
        text_length += len(line["text"]) - line["text"].count(" ")
    srsly.write_jsonl(output_path, data)
    msg.info(
        f"Downloaded extract contains about {int(round(text_length/1000000,0))} million characters"
    )
    msg.good(f"Saved to {output_path}")


if __name__ == "__main__":
    typer.run(main)

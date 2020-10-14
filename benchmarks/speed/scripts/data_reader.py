from typing import List
import srsly
from pathlib import Path


def read_data(txt_dir: Path) -> List[str]:
    texts = []
    for file in txt_dir.iterdir():
        if file.parts[-1].endswith("jsonl"):
            texts.extend(record["text"] for record in srsly.read_jsonl(file))
        else:
            texts.append(file.read_text())
    return texts

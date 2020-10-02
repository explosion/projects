from typing import List
from pathlib import Path


def read_data(txt_dir: Path) -> List[str]:
    texts = []
    for file in txt_dir.iterdir():
        texts.append(file.read_text())
    return texts

from typing import List
import re
from pathlib import Path
import srsly
from spacy.util import minibatch


def read_data(txt_dir: Path, limit: int = 0) -> List[str]:
    texts = []
    for file in txt_dir.iterdir():
        if file.parts[-1].endswith("jsonl"):
            texts.extend(record["text"] for record in srsly.read_jsonl(file))
        else:
            texts.append(file.read_text())
    texts = [text.strip() for text in texts if len(text.split()) >= 5]
    if limit:
        texts = texts[:limit]
    return texts


def rebatch_texts(texts, batch_size):
    newline_re = re.compile("\n+")
    for batch in minibatch(texts, size=batch_size):
        batch = [newline_re.sub("\n", text) for text in batch]
        batch = "\n\n".join(batch)
        yield batch

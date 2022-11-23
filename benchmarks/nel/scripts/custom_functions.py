from pathlib import Path
from typing import Callable

from spacy import registry, Vocab

from wikid.scripts.kb import WikiKB


@registry.misc("spacy.WikiKBFromFile.v1")
def load_kb(kb_path: Path) -> Callable[[Vocab], WikiKB]:
    """Loads WikiKB instance from disk.
    kb_path (Path): Path to WikiKB file.
    RETURNS (Callable[[Vocab], WikiKB]): Callable generating WikiKB from disk.
    """
    def kb_from_file(_: Vocab):
        return WikiKB.generate_from_disk(path=kb_path)

    return kb_from_file

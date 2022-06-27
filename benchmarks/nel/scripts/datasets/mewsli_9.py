""" Dataset class for Mewsli-9 dataset. """
import csv
from typing import Tuple, Set, List, Dict

from spacy.tokens import Doc

from datasets.dataset import Dataset
from datasets.schemas import Entity, Annotation


class Mewsli9Dataset(Dataset):
    """Mewsli-9 dataset."""

    @property
    def name(self) -> str:
        return "mewsli_9"

    def _parse_external_corpus(
        self, **kwargs
    ) -> Tuple[Dict[str, Entity], Set[str], Dict[str, List[Annotation]]]:
        entities: Dict[str, Entity] = {}
        annotations: Dict[str, List[Annotation]] = {}

        with open(
            self._paths["assets"] / "en" / "mentions.tsv", encoding="utf-8"
        ) as file_path:
            for i, row in enumerate(csv.reader(file_path, delimiter="\t")):
                if i == 0:
                    continue
                assert len(row) == 8

    def clean_assets(self) -> None:
        pass

    def _create_annotated_docs(self) -> List[Doc]:
        pass

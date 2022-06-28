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
    ) -> Tuple[Dict[str, Entity], Dict[str, List[Annotation]]]:
        entities: Dict[str, Entity] = {}
        annotations: Dict[str, List[Annotation]] = {}

        with open(self._paths["assets"] / "en" / "mentions.tsv", encoding="utf-8") as file_path:
            for i, row in enumerate(csv.DictReader(file_path, delimiter="\t")):
                assert len(row) == 9
                name = row["url"].split("/")[-1]
                if name not in entities:
                    entities[name] = Entity(names={name})
                entities[name].frequency += 1

                if row["docid"] not in annotations:
                    annotations[row["docid"]] = []
                annotations[row["docid"]].append(
                    Annotation(
                        entity_name=name,
                        entity_id=row["qid"],
                        start_pos=int(row["position"]),
                        end_pos=int(row["position"]) + int(row["length"])
                    )
                )

            return entities, annotations

    def clean_assets(self) -> None:
        pass

    def _create_annotated_docs(self) -> List[Doc]:
        pass

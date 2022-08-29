""" Dataset class for Mewsli-9 dataset. """
import csv
from typing import Tuple, Set, List, Dict, Optional

import tqdm
from spacy.tokens import Doc

from datasets.dataset import Dataset
from datasets.utils import fetch_entity_information, create_spans_from_doc_annotation
from schemas import Entity, Annotation


class Mewsli9Dataset(Dataset):
    """Mewsli-9 dataset."""

    @property
    def name(self) -> str:
        return "mewsli_9"

    def _parse_external_corpus(
        self, **kwargs
    ) -> Tuple[Dict[str, Entity], Set[str], Dict[str, List[Annotation]]]:
        entity_names: Set[str] = set()
        annotations: Dict[str, List[Annotation]] = {}

        with open(
            self._paths["assets"] / "en" / "mentions.tsv", encoding="utf-8"
        ) as file_path:
            for i, row in enumerate(csv.DictReader(file_path, delimiter="\t")):
                assert len(row) == 9
                name = row["url"].split("/")[-1]
                entity_names.add(name)

                if row["docid"] not in annotations:
                    annotations[row["docid"]] = []
                annotations[row["docid"]].append(
                    Annotation(
                        entity_name=name.replace("_", " "),
                        entity_id=row["qid"],
                        start_pos=int(row["position"]),
                        end_pos=int(row["position"]) + int(row["length"]),
                    )
                )

        entities, failed_entity_lookups, _ = fetch_entity_information(
            "name", tuple(entity_names)
        )

        return entities, failed_entity_lookups, annotations

    def clean_assets(self) -> None:
        pass

    def _create_annotated_docs(self, filter_terms: Optional[Set[str]] = None) -> List[Doc]:
        annotated_docs: List[Doc] = []
        texts: Dict[str, str]
        with open(
            self._paths["assets"] / "en" / "docs.tsv", encoding="utf-8"
        ) as title_file:
            row_count = sum(1 for _ in title_file)
            title_file.seek(0)

            with tqdm.tqdm(
                desc="Creating doc objects", total=row_count, leave=False
            ) as pbar:
                for row in csv.DictReader(title_file, delimiter="\t"):
                    with open(
                        self._paths["assets"] / "en" / "text" / row["docid"],
                        encoding="utf-8",
                    ) as text_file:
                        # Replace newlines with whitespace and \xa0 (non-breaking whitespace) appearing after titles
                        # with a period.
                        doc_text = "".join([
                            line.replace("\n", " ").replace("\xa0", ".") for line in text_file.readlines()
                        ])

                        if filter_terms and not any([ft in doc_text for ft in filter_terms]):
                            pbar.update(1)
                            continue

                        doc = self._nlp_base(doc_text)
                        doc.ents, _ = create_spans_from_doc_annotation(
                            doc=doc,
                            entities_info=self._entities,
                            annotations=self._annotations.get(row["docid"], []),
                            entities_failed_lookups=self._failed_entity_lookups,
                            harmonize_with_doc_ents=True,
                        )
                        annotated_docs.append(doc)
                    pbar.update(1)

        return annotated_docs

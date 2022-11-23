""" Dataset class for Mewsli-9 dataset. """
import csv
import distutils.dir_util
import time
from typing import Tuple, Set, List, Dict, Optional

import spacy
import tqdm
from spacy import Language
from spacy.tokens import Doc

from datasets.dataset import Dataset
from datasets.utils import fetch_entity_information, create_spans_from_doc_annotation
from wikid import schemas, load_entities


class Mewsli9Dataset(Dataset):
    """Mewsli-9 dataset."""

    @property
    def name(self) -> str:
        return "mewsli_9"

    def _extract_annotations_from_corpus(
        self, **kwargs
    ) -> Dict[str, List[schemas.Annotation]]:
        annotations: Dict[str, List[schemas.Annotation]] = {}

        with open(
            self._paths["assets"] / "clean" / "en" / "mentions.tsv", encoding="utf-8"
        ) as file_path:
            for i, row in enumerate(csv.DictReader(file_path, delimiter="\t")):
                assert len(row) == 9

                if row["docid"] not in annotations:
                    annotations[row["docid"]] = []
                annotations[row["docid"]].append(
                    schemas.Annotation(
                        entity_name=row["url"].split("/")[-1].replace("_", " "),
                        entity_id=row["qid"],
                        start_pos=int(row["position"]),
                        end_pos=int(row["position"]) + int(row["length"]),
                    )
                )

        return annotations

    def clean_assets(self) -> None:
        # No cleaning necessary, just copy all data into /clean.
        distutils.dir_util.copy_tree(str(self._paths["assets"] / "raw"), str(self._paths["assets"] / "clean"))

    def _create_annotated_docs(self, nlp: Language, filter_terms: Optional[Set[str]] = None) -> List[Doc]:
        annotated_docs: List[Doc] = []

        with open(
            self._paths["assets"] / "clean" / "en" / "docs.tsv", encoding="utf-8"
        ) as title_file:
            # todo
            #   - update nel.cfg with correct file path
            #   - add KB loader - code and to config
            #   - ensure training runs and uses WikiKB
            row_count = sum(1 for _ in title_file)
            title_file.seek(0)
            n_annots_available = 0
            n_annots_assigned = 0

            # Load entities batched to avoid hitting max. number of parameters supported by SQLite.
            batch_size = 2**14
            qids = tuple({annot.entity_id for annots in self._annotations.values() for annot in annots})
            entities = {
                qid: entity_info
                for entity_batch in
                [
                    load_entities(qids=qids[i:i + batch_size], language=self._language)
                    for i in range(0, len(qids), batch_size)
                ]
                for qid, entity_info in entity_batch.items()
            }

            with tqdm.tqdm(
                desc="Creating doc objects", total=row_count, leave=False
            ) as pbar:
                for row in csv.DictReader(title_file, delimiter="\t"):
                    with open(
                        self._paths["assets"] / "clean" / "en" / "text" / row["docid"],
                        encoding="utf-8",
                    ) as text_file:
                        # Replace newlines with whitespace and \xa0 (non-breaking whitespace) appearing after titles
                        # with a period. This maintains the correct offsets in the dataset annotations.
                        doc_text = "".join([
                            line.replace("\n", " ").replace("\xa0", ".") for line in text_file.readlines()
                        ])

                        if filter_terms and not any([ft in doc_text for ft in filter_terms]):
                            pbar.update(1)
                            continue

                        doc = nlp(doc_text)
                        doc_annots = self._annotations.get(row["docid"], [])
                        doc.ents, _ = create_spans_from_doc_annotation(
                            doc=doc, entities_info=entities, annotations=doc_annots,
                        )
                        annotated_docs.append(doc)
                        n_annots_available += len(doc_annots)
                        n_annots_assigned += len(doc.ents)
                    pbar.update(1)

        print(f"Assigned {n_annots_assigned} out of {n_annots_available} annotations "
              f"({(n_annots_assigned / n_annots_available * 100):.2f}%) in {pbar.n} docs.")

        return annotated_docs

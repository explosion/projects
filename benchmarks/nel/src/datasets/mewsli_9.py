""" Dataset class for Mewsli-9 dataset. """
import copy
import csv
import distutils.dir_util
import pickle
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
            curr_article: Optional[str] = None
            curr_docid: Optional[str] = None

            for i, row in enumerate(csv.DictReader(file_path, delimiter="\t")):
                assert len(row) == 9
                if row["docid"] not in annotations:
                    annotations[row["docid"]] = []

                # Read article, if this annotation refers to a new entity.
                if row["docid"] != curr_docid:
                    curr_docid = row["docid"]
                    curr_article = self._read_article_file(curr_docid)

                # Correct leading/trailing whitespaces.
                annot_start = int(row["position"])
                annot_end = annot_start + int(row["length"])
                while curr_article[annot_start] == " ":
                    annot_start += 1
                while curr_article[annot_end - 1] == " ":
                    annot_end -= 1
                annot_text = curr_article[annot_start:annot_end]
                assert annot_text.startswith(" ") is False and annot_text.endswith(" ") is False

                annotations[row["docid"]].append(
                    schemas.Annotation(
                        entity_name=row["url"].split("/")[-1].replace("_", " "),
                        entity_id=row["qid"],
                        start_pos=annot_start,
                        end_pos=annot_end,
                    )
                )

        return annotations

    def clean_assets(self) -> None:
        # No cleaning necessary, just copy all data into /clean.
        distutils.dir_util.copy_tree(str(self._paths["assets"] / "raw"), str(self._paths["assets"] / "clean"))

    def _read_article_file(self, doc_id: str) -> str:
        """Reads article file for specified doc ID.
        doc_id (str): Doc ID of article to read.
        RETURNS (str): Article text as single string.
        """
        with open(
            self._paths["assets"] / "clean" / "en" / "text" / doc_id,
            encoding="utf-8",
        ) as text_file:
            # Replace newlines with whitespace and \xa0 (non-breaking whitespace) appearing after titles
            # with a period. This maintains the correct offsets in the dataset annotations.
            return "".join([
                line.replace("\n", " ").replace("\xa0", ".") for line in text_file.readlines()
            ])

    def _create_annotated_docs(self, nlp: Language, filter_terms: Optional[Set[str]] = None) -> List[Doc]:
        annotated_docs: List[Doc] = []

        with open(
            self._paths["assets"] / "clean" / "en" / "docs.tsv", encoding="utf-8"
        ) as title_file:
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
                desc="Reading files", total=row_count, leave=False
            ) as pbar:
                docs_info_rows: List[Dict[str, str]] = []
                doc_texts: List[str] = []
                for row in csv.DictReader(title_file, delimiter="\t"):
                    doc_text = self._read_article_file(row["docid"])
                    if filter_terms and not any([ft in doc_text for ft in filter_terms]):
                        pbar.update(1)
                        continue
                    docs_info_rows.append(row)
                    doc_texts.append(doc_text)
                    pbar.update(1)

            docs = list(
                nlp.pipe(
                    tqdm.tqdm(
                        doc_texts,
                        desc="Creating doc objects",
                        leave=False
                    ),
                    n_process=-1,
                    batch_size=64,
                )
            )

            # This is an embarrassingly parallel scenario - speed is fine for ~10k articles though.
            with tqdm.tqdm(
                desc="Extracting annotations", total=len(docs), leave=False
            ) as pbar:
                for doc, row in zip(docs, docs_info_rows):
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

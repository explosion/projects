""" Dataset for Reddit EL data. """

import csv
from typing import Set, List, Tuple
import fileinput

from spacy.tokens import Doc

from .utils import _resolve_wiki_titles, _create_spans_from_doc_annotation, ENTITIES_TYPE, ANNOTATIONS_TYPE
from .dataset import Dataset


class RedditDataset(Dataset):
    """ RedditEL dataset. """

    @property
    def dataset_id(self) -> str:
        return "reddit"

    def _parse_external_corpus(self) -> Tuple[ENTITIES_TYPE, Set[str], ANNOTATIONS_TYPE]:
        file_names = [
            f"{quality}_{source}_annotations.tsv"
            for quality in ("gold", "silver", "bronze")
            for source in ("post", "comment")
        ]
        rows: List[List[str]] = []
        entities: ENTITIES_TYPE = {}
        annotations: ANNOTATIONS_TYPE = {}

        # Load data from .tsv files, track entity frequency.
        for file_name in file_names:
            with open(self._paths["assets"] / file_name) as file_path:
                quality = file_name.split("_")[0]
                for row in csv.reader(file_path, delimiter="\t"):
                    assert len(row) == 7
                    # Ditch anchor information in article URLs, as we can't use this in Wikidata lookups anyway.
                    row[3] = row[3].split("#")[0].split("?")[0]
                    rows.append(row)
                    if row[3] not in entities:
                        entities[row[3]] = {
                            "names": {row[3]},
                            "frequency": 0,
                            "description": None,
                            "quality": quality,
                            "source_id": row[0]
                        }
                    entities[row[3]]["frequency"] += 1

                    if row[0] not in annotations:
                        annotations[row[0]] = []
                    annotations[row[0]].append({
                        "name": row[3],
                        "entity_id": None,
                        "start_pos": int(row[4]),
                        "end_pos": int(row[5])
                    })

        # Fetch Wikidata IDs (QIDs). Some entities won't be resolved properly because of messy situations with redirects
        # and normalizations (e.g.: two different titles are redirected to the same entity, Wikipedia only returns this
        # one entity. Associating the remaining title with the correct entity can bloat up the code).
        # Since we don't expect many failures, we instead run failed lookups again individually. This should avoid any
        # situations with entity interdependencies at the cost of lookup speed.
        entities, failed_entity_lookups, title_qid_mappings = _resolve_wiki_titles(entities)
        if len(failed_entity_lookups):
            print(f"Trying to salvage {len(failed_entity_lookups)} failed lookups")
            entities, failed_entity_lookups, _title_qid_mapping = _resolve_wiki_titles(
                entities=entities, entity_titles=failed_entity_lookups, batch_size=1
            )
            title_qid_mappings = {**title_qid_mappings, **_title_qid_mapping}
        for entity_title in failed_entity_lookups:
            entities.pop(entity_title)

        # Update mentions with corresponding entity IDs.
        for source_id in annotations:
            for annotation in annotations[source_id]:
                if annotation["name"] not in failed_entity_lookups:
                    annotation["entity_id"] = title_qid_mappings[annotation["name"]]

        return entities, failed_entity_lookups, annotations

    def _create_annotated_docs(self) -> List[Doc]:
        annotated_docs: List[Doc] = []
        file_names: List[str] = []
        if self._options["titles"]:
            file_names.append("posts.tsv")
        if self._options["comments"]:
            file_names.append("comments.tsv")
        assert file_names, "Either 'titles' or 'comments' have to be True in corpus config."

        # Join records with line breaks.
        rows: List[List[str]] = []
        for file_name in [self._paths["assets"] / file_name for file_name in file_names]:
            row_length = 3 if file_name.name.endswith("posts.tsv") else 5
            with open(file_name) as file_path:
                for row in csv.reader(file_path, delimiter="\t"):
                    assert len(row) <= row_length
                    # If row has fewer than the specified number of entries: newlines from comments have been
                    # maintained, content is part of last valid comment.
                    if 0 < len(row) < row_length:
                        assert len(row) <= 1
                        rows[-1][-1] += " " + row[0]
                    elif len(row) == row_length:
                        rows.append(row)

        # Create spans from annotations.
        for row in rows:
            doc = self._nlp_temp(row[-1])
            # There might be multiple annotations for the same tokens/spans. This is handled by (1) sorting all
            # entities for this document by their frequency and (2) afterwards moving all overlapping entities to
            # the doc's _ attribute, so we might still consider that during evaluation.
            # Additionally, there is a number of index errors in the annotations (especially in the bronze dataset).
            # Some of these are resolved by aligning annotation with token indices.
            doc_spans, overlapping_doc_annotations = _create_spans_from_doc_annotation(
                doc=doc,
                entities_info=self._entities,
                annotations=self._annotations.get(row[0], []),
                entities_failed_lookups=self._failed_entity_lookups
            )
            doc.ents = doc_spans
            doc._.overlapping_annotations = overlapping_doc_annotations
            annotated_docs.append(doc)

        return annotated_docs

    def clean_assets(self) -> None:
        to_remove = {
            "bronze_post_annotations.tsv": {},
            "bronze_comment_annotations.tsv": {}
        }
        to_replace = {
            "bronze_comment_annotations.tsv": {
                "`` How to Lose a Guy in 10 Days": "How to Lose a Guy in 10 Days",
                "\"How to Lose a Guy in 10 Days": "\"How to Lose a Guy in 10 Days\"",
                "Money\t325\t330\tmoney\n": "Money\t323\t328\tmoney\n"
            },
            "bronze_post_annotations.tsv": {
                # "8qntul\tsports\tad\tAdvertising\t56\t58\tparade\n": "8qntul\tsports\tad\tParade\t56\t58\tparade\n",
                # "dtjq8bn\tpolitics\tGOOD\tGood\t102\t106\tGOOD\n": "dtjq8bn\tpolitics\tGOOD\tGood\t94\t98\tGOOD\n"
            }
        }

        for file_name in to_replace:
            for line in fileinput.input(self._paths["assets"] / file_name, inplace=True):
                if line not in to_remove[file_name]:
                    new_line = line
                    for snippet in to_replace[file_name]:
                        if snippet in line:
                            new_line = line.replace(snippet, to_replace[file_name][snippet])
                            break
                    print(new_line, end='')
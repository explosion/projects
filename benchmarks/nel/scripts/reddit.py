"""
Utilities for Reddit Entity Linking dataset.
todo @RM create ELDataset class(es) with standardized interface
"""
import random
from typing import Tuple, List, Set, Dict, Any, Union, Optional

import csv
from pathlib import Path
import numpy
import spacy
from spacy.tokens import DocBin, Doc, Span, Token

from utils import ENTITIES_TYPE, ANNOTATIONS_TYPE, resolve_wiki_titles


def _does_token_overlap_with_annotation(token: Token, annot_start: int, annot_end: int) -> bool:
    """ Checks whether token overlaps with annotation span.
    token (Token): Token to check.
    annot_start (int): Annotation's start index.
    annot_end (int): Annotation's end index.
    RETURNS (bool): Whether token overlaps with annotation span.
    """

    return annot_start <= token.idx <= annot_end or token.idx <= annot_start <= token.idx + len(token)


def _create_spans_from_doc_annotation(
    doc: Doc,
    entities_info: ENTITIES_TYPE,
    annotations: List[Dict[str, Union[Set[str], str, int]]],
    entities_failed_lookups: Set[str]
) -> Tuple[List[Span], List[Dict[str, Union[Set[str], str, int]]]]:
    """ Creates spans from annotations for one document.
    doc (Doc): Document for whom to create spans.
    entities_info (Dict[str, Dict[str, Union[Set[str], str, int]]]):
    annotation (List[Dict[str, Union[Set[str], str, int]]]): Annotations for this post/comment.
    entities_entities_failed_lookups (Set[str]): Set of entity names for whom Wiki API lookup failed).
    source_id (str): Unique source ID to look up annotation.
    RETURNS (Tuple[List[Span], List[Dict[str, Union[Set[str], str, int]]]]): List of doc spans for annotated entities;
        list of overlapping entities.
    """

    doc_annots: List[Dict[str, Union[Set[str], str, int]]] = []
    overlapping_doc_annotations: List[Dict[str, Union[Set[str], str, int]]] = []
    for i, annot in enumerate(
            sorted(
                [
                    {**annot, "frequency": entities_info.get(annot["entity_id"], {"frequency": -1})["frequency"]}
                    for annot in annotations
                ],
                key=lambda a: a["frequency"],
                reverse=True
            )
    ):
        # The Reddit EL dataset has some indexing mistakes that may result in . We align the annotation indexing
        # with spaCy's token indices.
        for t in doc:
            if _does_token_overlap_with_annotation(t, annot["start_pos"], annot["end_pos"]):
                annot["start_pos"] = t.idx
                break
        for t in reversed([t for t in doc]):
            if _does_token_overlap_with_annotation(t, annot["start_pos"], annot["end_pos"] - 1):
                annot["end_pos"] = t.idx + len(t)
                break

        # If there is an overlap between annotation's start and end position and this token's parsed start
        # and end, we try to create a span with this token's position.
        overlaps = False
        if annot["frequency"] == -1:
            assert annot["entity_id"] not in entities_info and annot["name"] in entities_failed_lookups
            continue
        for j in range(0, len(doc_annots)):
            if not (annot["end_pos"] < doc_annots[j]["start_pos"] or annot["start_pos"] > doc_annots[j]["end_pos"]):
                overlaps = True
                overlapping_doc_annotations.append(annot)
                break
        if not overlaps:
            doc_annots.append(annot)

    doc_spans = [
        doc.char_span(annot["start_pos"], annot["end_pos"], label=annot["name"], kb_id=annot["entity_id"])
        for annot in doc_annots
    ]
    assert all([span is not None for span in doc_spans])

    return doc_spans, overlapping_doc_annotations


def create_corpus(
    nlp_dir: Path,
    data_path_reddit: Path,
    entities_info: ENTITIES_TYPE,
    entities_failed_lookups: Set[str],
    all_annotations: ANNOTATIONS_TYPE,
    options: Dict[str, Any]
) -> List[Doc]:
    """ Loads corpus for Reddit entity linking dataset.
    data_path_reddit (Path): Path for Reddit entity data.
    nlp_path (Path): Path to serialized pipeline.
    entities_info (Dict[str, Dict[str, Union[Set[str], str, int]]]): Dictionary with information on entities (QID,
    description etc.).
    entities_failed_lookups (Set[str]): List of entity names for which the Wikipedia lookup failed.
    all_annotations (Dict[str, List[Dict[str, Union[Set[str], str, int]]]]): Corpus annotations.
    options (Dict[str, Any]): Options for corpus creation.
    RETURNS (List[Doc]): List of docs with entity annotations.
    """

    nlp = spacy.load(nlp_dir)
    docs: List[Doc] = []
    file_names: List[str] = []
    if options["titles"]:
        file_names.append("posts.tsv")
    if options["comments"]:
        file_names.append("comments.tsv")
    assert file_names, "Either 'titles' or 'comments' have to be True in corpus config."
    Doc.set_extension("overlapping_annotations", default=None)

    # Join records with line breaks.
    rows: List[List[str]] = []
    for file_name in [data_path_reddit / file_name for file_name in file_names]:
        row_length = 3 if file_name.name.endswith("posts.tsv") else 5
        with open(file_name) as file_path:
            for row in csv.reader(file_path, delimiter="\t"):
                assert len(row) <= row_length
                # If row has fewer than the specified number of entries: newlines from comments have been maintained,
                # content is part of last valid comment.
                if 0 < len(row) < row_length:
                    assert len(row) <= 1
                    rows[-1][-1] += " " + row[0]
                elif len(row) == row_length:
                    rows.append(row)

    # Create spans from annotations.
    for row in rows:
        doc = nlp.make_doc(row[-1])

        # There might be multiple annotations for the same tokens/spans. This is handled by (1) sorting all
        # entities for this document by their frequency and (2) afterwards moving all overlapping entities to
        # the doc's _ attribute, so we might still consider that during evaluation.
        # Additionally, there is a number of index errors in the annotations (especially in the bronze dataset). Some
        # of these are resolved by aligning annotation with token indices.
        doc_spans, overlapping_doc_annotations = _create_spans_from_doc_annotation(
            doc=doc,
            entities_info=entities_info,
            annotations=all_annotations.get(row[0], []),
            entities_failed_lookups=entities_failed_lookups
        )
        doc.ents = doc_spans
        doc._.overlapping_annotations = overlapping_doc_annotations
        docs.append(doc)

    return docs


def parse_external_corpus(data_path_reddit: Path) -> Tuple[ENTITIES_TYPE, ANNOTATIONS_TYPE, Set[str]]:
    """ Parses external corpus. Loads data on entities and mentions.

    data_path_reddit (Path): Path for Reddit entity data.
    RETURNS (Tuple[Dict[str, Dict[str, Union[str, int]]], Dict[str, Dict[str, Union[str, int]]], Set[str]): Collection
        of entity info, annotations info, entities for whom external KB lookup failed.
    """

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
        with open(data_path_reddit / file_name) as file_path:
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
    # and normalizations (e.g.: two different titles are redirected to the same entity, Wikipedia only returns this one
    # entity. Associating the remaining title with the correct entity can bloat up the code).
    # Since we don't expect many failures, we instead run failed lookups again individually. This should avoid any
    # situations with entity interdependencies at the cost of lookup speed.
    entities, failed_lookups, title_qid_mappings = resolve_wiki_titles(entities)
    if len(failed_lookups):
        print(f"Trying to salvage {len(failed_lookups)} failed lookups")
        entities, failed_lookups, _title_id_mapping = resolve_wiki_titles(entities, entity_titles=list(failed_lookups), batch_size=1)
        title_qid_mappings = {**title_qid_mappings, **_title_id_mapping}
    for entity_title in failed_lookups:
        entities.pop(entity_title)

    # Update mentions with corresponding entity IDs.
    for source_id in annotations:
        for annotation in annotations[source_id]:
            if annotation["name"] not in failed_lookups:
                annotation["entity_id"] = title_qid_mappings[annotation["name"]]

    return entities, annotations, failed_lookups

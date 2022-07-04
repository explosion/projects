""" Utilities for NEL benchmark. """

from typing import Tuple, Set, List, Dict

import tqdm
from spacy.tokens import Token, Doc, Span

from schemas import Entity, Annotation
from wiki import wiki_dump_api


def _does_token_overlap_with_annotation(
    token: Token, annot_start: int, annot_end: int
) -> bool:
    """Checks whether token overlaps with annotation span.
    token (Token): Token to check.
    annot_start (int): Annotation's start index.
    annot_end (int): Annotation's end index.
    RETURNS (bool): Whether token overlaps with annotation span.
    """

    return (
        annot_start <= token.idx <= annot_end
        or token.idx <= annot_start <= token.idx + len(token)
    )


def _fetch_entity_information(
    key: str,
    values: Tuple[str, ...],
    batch_size: int = 1000,
) -> Tuple[Dict[str, Entity], Set[str], Dict[str, str]]:
    """
    Fetches information on entities from database.
    key (str): Attribute to match values to. Must be one of ("id", "name").
    values (Tuple[str]): Values for key to look up.
    db_conn (sqlite3.Connection): Database connection.
    batch_size (int): Number of entity titles to resolve in the same API request. Between 1 and 50.
    RETURNS (Tuple[Dict[str, Entity], Set[str], Dict[str, str]]): Updated entities, failed lookups, mappings of titles
        to QIDs.
    """

    assert 1 <= batch_size, f"Batch size has to be at least 1."

    pbar = tqdm.tqdm(total=len(values))
    failed_lookups: Set[str] = set()
    name_qid_map: Dict[str, str] = {}
    entities: Dict[str, Entity] = {}

    for i in range(0, len(values), batch_size):
        chunk = values[i : i + batch_size]
        entities_chunk = wiki_dump_api.load_entities(key, chunk)
        _failed_lookups = set(chunk)

        # Replace entity titles keys in dict with Wikidata QIDs. Add entity description.
        for entity in entities_chunk.values():
            entities[entity.qid] = entity
            name_qid_map[entity.name] = entity.qid
            _failed_lookups.remove(entity.qid if key == "id" else entity.name)

        failed_lookups |= _failed_lookups
        pbar.update(len(chunk))

    pbar.close()

    return entities, failed_lookups, name_qid_map


def _create_spans_from_doc_annotation(
    doc: Doc,
    entities_info: Dict[str, Entity],
    annotations: List[Annotation],
    entities_failed_lookups: Set[str],
) -> Tuple[List[Span], List[Annotation]]:
    """Creates spans from annotations for one document.
    doc (Doc): Document for whom to create spans.
    entities_info (ENTITIES_TYPE): All available entities.
    annotation (List[Dict[str, Union[Set[str], str, int]]]): Annotations for this post/comment.
    entities_failed_lookups (Set[str]): Set of entity names for whom Wiki API lookup failed.
    RETURNS (Tuple[List[Span], List[Dict[str, Union[Set[str], str, int]]]]): List of doc spans for annotated entities;
        list of overlapping entities.
    """

    doc_annots: List[Annotation] = []
    overlapping_doc_annotations: List[Annotation] = []
    for i, annot_data in enumerate(
        sorted(
            [
                {
                    "annot": annot,
                    "freq": entities_info[annot.entity_id].count
                    if annot.entity_id in entities_info
                    else -1,
                }
                for annot in annotations
            ],
            key=lambda a: a["freq"],
            reverse=True,
        )
    ):
        annot, count = annot_data["annot"], annot_data["freq"]
        # Indexing mistakes in the dataset might lead to wrong and/or overlapping annotations. We align the annotation
        # indices with spaCy's token indices to avoid at least some of these.
        for t in doc:
            if _does_token_overlap_with_annotation(
                t, annot.start_pos, annot.end_pos
            ):
                annot.start_pos = t.idx
                break
        for t in reversed([t for t in doc]):
            if _does_token_overlap_with_annotation(
                t, annot.start_pos, annot.end_pos - 1
            ):
                annot.end_pos = t.idx + len(t)
                break

        # If there is an overlap between annotation's start and end position and this token's parsed start
        # and end, we try to create a span with this token's position.
        overlaps = False
        if count == -1:
            assert (
                annot.entity_id not in entities_info
                and annot.entity_name in entities_failed_lookups
            )
            continue
        for j in range(0, len(doc_annots)):
            if not (
                annot.end_pos < doc_annots[j].start_pos
                or annot.start_pos > doc_annots[j].end_pos
            ):
                overlaps = True
                overlapping_doc_annotations.append(annot)
                break
        if not overlaps:
            doc_annots.append(annot)

    doc_spans = [
        # No label/entity type information available.
        doc.char_span(
            annot.start_pos, annot.end_pos, label="NIL", kb_id=annot.entity_id
        )
        for annot in doc_annots
    ]
    assert all([span is not None for span in doc_spans])

    return doc_spans, overlapping_doc_annotations

""" Utilities for NEL benchmark. """

from typing import Dict, List, Set, Tuple
import tqdm
from spacy.tokens import Token, Span, Doc
from wikid import schemas, load_entities


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


def fetch_entity_information(
    values: Tuple[str, ...],
    language: str,
    batch_size: int = 5000,
) -> Tuple[Dict[str, schemas.Entity], Set[str], Dict[str, str]]:
    """
    Fetches information on entities from database.
    values (Tuple[str]): Values for key to look up.
    language (str): Language.
    db_conn (sqlite3.Connection): Database connection.
    batch_size (int): Number of entity titles to resolve in the same API request. Between 1 and 50.
    RETURNS (Tuple[Dict[str, Entity], Set[str], Dict[str, str]]): Updated entities, failed lookups, mappings of titles
        to QIDs.
    """

    assert 1 <= batch_size, f"Batch size has to be at least 1."

    pbar = tqdm.tqdm(total=len(values))
    failed_lookups: Set[str] = set()
    name_qid_map: Dict[str, str] = {}
    entities: Dict[str, schemas.Entity] = {}

    for i in range(0, len(values), batch_size):
        chunk = tuple([v.replace("_", " ") for v in values[i : i + batch_size]])
        entities_chunk = load_entities(language, chunk)
        _failed_lookups = set(chunk)

        # Replace entity titles keys in dict with Wikidata QIDs. Add entity description.
        for entity in entities_chunk.values():
            entities[entity.qid] = entity
            name_qid_map[entity.name] = entity.qid
            _failed_lookups.remove(entity.qid)

        failed_lookups |= _failed_lookups
        pbar.update(len(chunk))

    pbar.close()

    return entities, failed_lookups, name_qid_map


def create_spans_from_doc_annotation(
    doc: Doc,
    entities_info: Dict[str, schemas.Entity],
    annotations: List[schemas.Annotation],
    harmonize_with_doc_ents: bool,
) -> Tuple[List[Span], List[schemas.Annotation]]:
    """Creates spans from annotations for one document.
    doc (Doc): Document for whom to create spans.
    entities_info (Dict[str, Entity]): All available entities.
    annotation (List[Dict[str, Union[Set[str], str, int]]]): Annotations for this post/comment.
    harmonize_harmonize_with_doc_ents (Language): Whether to only keep those annotations matched by entities in the
        provided Doc object.
    RETURNS (Tuple[List[Span], List[Dict[str, Union[Set[str], str, int]]]]): List of doc spans for annotated entities;
        list of overlapping entities.
    """
    doc_ents_idx = {
        # spaCy sometimes includes leading articles in entities, our benchmark datasets don't. Hence we drop all leading
        # "the " and adjust the entity positions accordingly.
        (ent.start_char + (0 if not ent.text.lower().startswith("the ") else 4), ent.end_char)
        for ent in doc.ents
    }
    doc_annots: List[schemas.Annotation] = []
    overlapping_doc_annotations: List[schemas.Annotation] = []

    if harmonize_with_doc_ents and len(doc_ents_idx) == 0:
        return [], []

    for i, annot_data in enumerate(
        sorted(
            [
                {
                    "annot": annot,
                    "count": entities_info[annot.entity_id].count
                    if annot.entity_id in entities_info
                    else -1,
                }
                for annot in annotations
            ],
            key=lambda a: a["count"],
            reverse=True,
        )
    ):
        annot, count = annot_data["annot"], annot_data["count"]

        # Indexing mistakes in the dataset might lead to wrong and/or overlapping annotations. We align the annotation
        # indices with spaCy's token indices to avoid at least some of these.
        for token in doc:
            if _does_token_overlap_with_annotation(token, annot.start_pos, annot.end_pos):
                annot.start_pos = token.idx
                break
        for token in reversed([t for t in doc]):
            if _does_token_overlap_with_annotation(
                token, annot.start_pos, annot.end_pos - 1
            ):
                annot.end_pos = token.idx + len(token)
                break

        # After token alignment: filter with NER pipeline, if available.
        if harmonize_with_doc_ents and (annot.start_pos, annot.end_pos) not in doc_ents_idx:
            continue

        # If there is an overlap between annotation's start and end position and this token's parsed start
        # and end, we try to create a span with this token's position.
        overlaps = False
        if count == -1:
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

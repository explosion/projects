""" Utilities for NEL benchmark. """

import copy
import math
import urllib.parse
from typing import Tuple, Set, List, Dict, Optional, Union, Any, Iterable

import requests
import tqdm
from spacy import Language
from spacy.tokens import Token, Doc, Span

# todo @RM replace with pydantic schemas
ENTITY_TYPE = Dict[str, Union[Set[str], str, int, Dict[str, Set[str]]]]
ENTITIES_TYPE = Dict[str, ENTITY_TYPE]
ANNOTATIONS_TYPE = Dict[str, List[Dict[str, Union[Set[str], str, int]]]]

MAX_WIKI_API_QUERY_BATCH_SIZE = 20
MAX_WIKI_API_WBENTITIES_BATCH_SIZE = 50


def _does_token_overlap_with_annotation(token: Token, annot_start: int, annot_end: int) -> bool:
    """ Checks whether token overlaps with annotation span.
    token (Token): Token to check.
    annot_start (int): Annotation's start index.
    annot_end (int): Annotation's end index.
    RETURNS (bool): Whether token overlaps with annotation span.
    """

    return annot_start <= token.idx <= annot_end or token.idx <= annot_start <= token.idx + len(token)


def _resolve_wiki_mentions(
    entities: ENTITIES_TYPE,
    batch_size: int = MAX_WIKI_API_WBENTITIES_BATCH_SIZE
) -> ENTITIES_TYPE:
    """ Resolves IDs of Wikipedia mentions, i.e. entities mentioned in articles/other entities' descriptions, to their
    titles and aliases.
    entities (ENTITIES_TYPE): Entities to resolve.
    batch_size (int): Number of entity titles to resolve in the same API request. Between 1 and 50.
    """

    assert 1 <= batch_size <= MAX_WIKI_API_WBENTITIES_BATCH_SIZE, \
        f"Batch size has to be between 1 and {MAX_WIKI_API_WBENTITIES_BATCH_SIZE}."
    _entities = copy.deepcopy(entities)
    entity_qids = list(entities.keys())
    mentions_qids = list({mqid for qid in entity_qids for mqid in entities[qid]["mentions"]})
    mentions_info: Dict[str, Any] = {}
    pbar = tqdm.tqdm(desc="Resolving Wiki mentions", total=len(mentions_qids), leave=False)

    for i in range(0, len(mentions_qids), batch_size):
        chunk = mentions_qids[i:i + batch_size]
        response = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbgetentities",
                "props": "aliases|labels",
                "languages": "en",
                "ids": "|".join(chunk),
                "format": "json"
            }
        )

        # Parse API responses.
        _mentions_info = response.json()
        assert _mentions_info["success"] == 1
        mentions_info = {**mentions_info, **_mentions_info["entities"]}
        pbar.update(len(chunk))
    pbar.close()

    for qid in entity_qids:
        for mention_qid in _entities[qid]["mentions"]:
            # Add label and aliases.
            if mentions_info[mention_qid]["labels"]:
                _entities[qid]["mentions"][mention_qid].add(mentions_info[mention_qid]["labels"]["en"]["value"])
            for alias_info in mentions_info[mention_qid]["aliases"].get("en", []):
                _entities[qid]["mentions"][mention_qid].add(alias_info["value"])

    return _entities


def _prune_category_titles(cat_titles: Iterable[Dict[str, str]]) -> Set[str]:
    """ Cleans and filters category titles.
    cat_titles (Tuple[str, ...]): List of category titles to prune.
    RETURNS (Set[str]): Set of pruned category titles.
    """
    return {
        "".join(cat["title"].split("Category:")[1:])
        for cat in cat_titles if not any([
            cat["title"].startswith(f"Category:{prefix}") for prefix in ("Articles", "Pages", "Use ")
        ])
    }


def _resolve_wiki_titles(
    entities: ENTITIES_TYPE,
    entity_titles: Optional[Set[str]] = None,
    batch_size: int = MAX_WIKI_API_QUERY_BATCH_SIZE,
    progress_bar_desc: str = ""
) -> Tuple[ENTITIES_TYPE, Set[str], Dict[str, str]]:
    """ Resolves Wikipedia titles to Wikidata IDs. Also fetches descriptions for entities.
    This is currently very slow (~10 entries/s) due to having to limit the batch size as to not exceed the Wiki API's
    limitations. This could be circumvented by parallel querying (even better: preprocessed dump instead of accessing
    API, which isn't fit for production anyway).
    entities (ENTITIES_TYPE): Entities to resolve.
    entity_titles (Set[str]): List of entity titles to resolve.
    batch_size (int): Number of entity titles to resolve in the same API request. Between 1 and 50.
    tqdm_str (str): String to pass on to TQDM progress bar as description.
    RETURNS (Dict[str, str]): Mapping of titles to entity IDs.
    """

    assert 1 <= batch_size <= MAX_WIKI_API_QUERY_BATCH_SIZE, \
        f"Batch size has to be between 1 and {MAX_WIKI_API_QUERY_BATCH_SIZE}."
    entity_titles = list(entity_titles if entity_titles else entities.keys())
    pbar = tqdm.tqdm(desc=progress_bar_desc, total=len(entity_titles), leave=False)
    failed_lookups: Set[str] = set()
    _entities = copy.deepcopy(entities)
    title_qid_mappings: Dict[str, str] = {}

    for i in range(0, len(entity_titles), batch_size):
        # Resolve article names to Wikidata QIDs. See
        # https://stackoverflow.com/questions/37024807/how-to-get-wikidata-id-for-an-wikipedia-article-by-api and
        # https://stackoverflow.com/questions/8555320/is-there-a-wikipedia-api-just-for-retrieve-the-content-summary.
        chunk = entity_titles[i:i + batch_size]
        request_params = {
            "action": "query",
            "prop": "pageprops|extracts|categories|pageterms|pageviews",
            "ppprop": "wikibase_item",
            # Some special chars such as & are not escaped by requests, so we do that with urllib.parse.quote().
            "titles": urllib.parse.quote("|".join(chunk)),
            "format": "json",
            "exintro": None,
            "explaintext": None,
            # Make sure descriptions are available for all requested articles.
            "exlimit": min(len(chunk), 20),
            "redirects": 1,
            "cllimit": 500,
            "wbptlanguage": "en",
            "pvipdays": 5
        }
        request = requests.get(
            "https://en.wikipedia.org/w/api.php",
            # Reformat to keep flags/parameters without values.
            params='&'.join([k if v is None else f"{k}={v}" for k, v in request_params.items()])
        )

        # Parse API responses.
        entities_info = request.json()
        # Implementing response continuation would allow larger batch sizes and do away with the need for complete
        # batches. Couldn't get that work properly yet.
        assert entities_info["batchcomplete"] == ""
        # Titles might be normalized and/or redirected by the Wikipedia API. Use .lower() to avoid capitalization
        # differences for lookups.
        normalizations_from_to = {entry["from"]: entry["to"] for entry in
                                  entities_info["query"].get("normalized", {})}
        normalizations_to_from = {v.lower(): k for k, v in normalizations_from_to.items()}
        redirections_to_from = {entry["to"]: entry["from"] for entry in entities_info["query"].get("redirects", {})}
        entities_info = entities_info["query"]["pages"]

        # Replace entity titles keys in dict with Wikidata QIDs. Add entity description.
        for page_id, entity_info in entities_info.items():
            redirection_names: Set[str] = set()
            # Fetch original (non-normalized, non-redirected) entity title to ensure correct association of Wikidata
            # ID with entity. There may be multiple redirections, so we loop through them.
            entity_title = entity_info["title"]
            while entity_title in redirections_to_from:
                redirection_names.add(entity_title)
                entity_title = redirections_to_from[entity_title]
            entity_title = normalizations_to_from.get(entity_title.lower(), entity_title).replace(" ", "_")

            # Ignore lookup failures.
            if page_id.startswith("-"):
                failed_lookups.add(entity_title)
                continue

            # Rename entry in info dict from title to Wikidata ID; add entry description.
            try:
                qid = entity_info["pageprops"]["wikibase_item"]
                # Don't overwrite existing entries when resolving names pointing to the same entity. Example:
                # "United_Kingdom" and "UK" point to the same QID and thus might overwrite each other otherwise.
                if qid not in _entities:
                    _entities[qid] = _entities.pop(entity_title)
                    _entities[qid]["categories"] = _prune_category_titles(entity_info["categories"])
                    _entities[qid]["pageviews"] = sum([int(vc) for vc in entity_info["pageviews"].values() if vc])
                else:
                    _entities[qid]["names"].add(entity_title)
                    _entities[qid]["frequency"] += _entities.pop(entity_title)["frequency"]
                _entities[qid]["names"] |= redirection_names | \
                    set(entity_info["terms"].get("alias", {})) | set(entity_info["terms"].get("label", {}))
                _entities[qid]["description"] = entity_info["extract"]
                _entities[qid]["short_description"] = " ".join(entity_info["terms"].get("description", []))
                title_qid_mappings[entity_title] = qid
            except KeyError:
                failed_lookups.add(entity_title)

        # The Wikipedia API only returns one result if two mentions in the same batch are resolved to the same
        # entity via redirects. So if any entities haven't been updated, this is counted as a failed lookup.
        for entity_title in [et for et in chunk if et in _entities and et not in failed_lookups]:
            failed_lookups.add(entity_title)

        pbar.update(len(chunk))
    pbar.close()

    for qid in _entities:
        _entities[qid]["names"] = {name.replace("_", " ") for name in _entities[qid]["names"]}

    return _entities, failed_lookups, title_qid_mappings


def _create_spans_from_doc_annotation(
    doc: Doc,
    entities_info: ENTITIES_TYPE,
    annotations: List[Dict[str, Union[Set[str], str, int]]],
    entities_failed_lookups: Set[str],
    harmonize_with_doc_ents: bool
) -> Tuple[List[Span], List[Dict[str, Union[Set[str], str, int]]]]:
    """ Creates spans from annotations for one document.
    doc (Doc): Document for whom to create spans.
    entities_info (ENTITIES_TYPE): All available entities.
    annotation (List[Dict[str, Union[Set[str], str, int]]]): Annotations for this post/comment.
    entities_entities_failed_lookups (Set[str]): Set of entity names for whom Wiki API lookup failed).
    source_id (str): Unique source ID to look up annotation.
    harmonize_harmonize_with_doc_ents (Language): Whether to only keep those annotations matched by entities in the
        provided Doc object.
    RETURNS (Tuple[List[Span], List[Dict[str, Union[Set[str], str, int]]]]): List of doc spans for annotated entities;
        list of overlapping entities.
    """

    final_annots: List[Dict[str, Union[Set[str], str, int]]] = []
    doc_ents = {(ent.start_char, ent.end_char) for ent in doc.ents}
    overlapping_doc_annotations: List[Dict[str, Union[Set[str], str, int]]] = []

    if harmonize_with_doc_ents and not doc_ents:
        return [], []

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
        # Indexing mistakes in the dataset might lead to wrong and/or overlapping annotations. We align the annotation
        # indices with spaCy's token indices to avoid at least some of these.
        for t in doc:
            if _does_token_overlap_with_annotation(t, annot["start_pos"], annot["end_pos"]):
                annot["start_pos"] = t.idx
                break
        for t in reversed([t for t in doc]):
            if _does_token_overlap_with_annotation(t, annot["start_pos"], annot["end_pos"] - 1):
                annot["end_pos"] = t.idx + len(t)
                break

        # After token alignment: filter with NER pipeline, if available.
        if harmonize_with_doc_ents and (annot["start_pos"], annot["end_pos"]) not in doc_ents:
            continue

        # If there is an overlap between annotation's start and end position and this token's parsed start
        # and end, we try to create a span with this token's position.
        overlaps = False
        if annot["frequency"] == -1:
            assert annot["entity_id"] not in entities_info and annot["name"] in entities_failed_lookups
            continue
        for j in range(0, len(final_annots)):
            if not (annot["end_pos"] < final_annots[j]["start_pos"] or annot["start_pos"] > final_annots[j]["end_pos"]):
                overlaps = True
                overlapping_doc_annotations.append(annot)
                break
        if not overlaps:
            final_annots.append(annot)

    doc_spans = [
        # No label/entity type information available.
        doc.char_span(annot["start_pos"], annot["end_pos"], label="NIL", kb_id=annot["entity_id"])
        for annot in final_annots
    ]
    assert all([span is not None for span in doc_spans])

    return doc_spans, overlapping_doc_annotations

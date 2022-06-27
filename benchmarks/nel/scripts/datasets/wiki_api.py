""" Utilities for extracting information from Wikipedia/Wikidata APIs. """
import copy
import urllib
from typing import Dict, List, Set, Tuple, Optional, Iterable
import tqdm
import requests

# More elegant way to resolve import conflicts between training and evaluation calls?
try:
    from ..datasets.schemas import Entity, Annotation
except ValueError:
    from datasets.schemas import Entity, Annotation


MAX_WIKI_API_QUERY_BATCH_SIZE = 20
MAX_WIKI_API_WBENTITIES_BATCH_SIZE = 50


def _prune_category_titles(cat_titles: Iterable[Dict[str, str]]) -> Set[str]:
    """Cleans and filters category titles.
    cat_titles (Tuple[str, ...]): List of category titles to prune.
    RETURNS (Set[str]): Set of pruned category titles.
    """
    return {
        "".join(cat["title"].split("Category:")[1:])
        for cat in cat_titles
        if not any(
            [
                cat["title"].startswith(f"Category:{prefix}")
                for prefix in ("Articles", "Pages", "Use ")
            ]
        )
    }


def _resolve_wiki_titles(
    entities: Dict[str, Entity],
    entity_titles: Optional[Set[str]] = None,
    batch_size: int = MAX_WIKI_API_QUERY_BATCH_SIZE,
    progress_bar_desc: str = "",
) -> Tuple[Dict[str, Entity], Set[str], Dict[str, str]]:
    """Resolves Wikipedia titles to Wikidata IDs. Also fetches descriptions for entities.
    This is currently very slow (~10 entries/s) due to having to limit the batch size as to not exceed the Wiki API's
    limitations. This could be circumvented by parallel querying (even better: preprocessed dump instead of accessing
    API, which isn't fit for production anyway).
    entities (Dict[str, Entity]): Entities to resolve.
    entity_titles (Set[str]): List of entity titles to resolve.
    batch_size (int): Number of entity titles to resolve in the same API request. Between 1 and 50.
    tqdm_str (str): String to pass on to TQDM progress bar as description.
    RETURNS (Dict[str, str]): Mapping of titles to entity IDs.
    """

    assert (
        1 <= batch_size <= MAX_WIKI_API_QUERY_BATCH_SIZE
    ), f"Batch size has to be between 1 and {MAX_WIKI_API_QUERY_BATCH_SIZE}."
    entity_titles = list(entity_titles if entity_titles else entities.keys())
    pbar = tqdm.tqdm(desc=progress_bar_desc, total=len(entity_titles), leave=False)
    failed_lookups: Set[str] = set()
    _entities = copy.deepcopy(entities)
    title_qid_mappings: Dict[str, str] = {}

    for i in range(0, len(entity_titles), batch_size):
        # Resolve article names to Wikidata QIDs. See
        # https://stackoverflow.com/questions/37024807/how-to-get-wikidata-id-for-an-wikipedia-article-by-api and
        # https://stackoverflow.com/questions/8555320/is-there-a-wikipedia-api-just-for-retrieve-the-content-summary.
        chunk = entity_titles[i : i + batch_size]
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
            "pvipdays": 5,
        }
        request = requests.get(
            "https://en.wikipedia.org/w/api.php",
            # Reformat to keep flags/parameters without values.
            params="&".join(
                [k if v is None else f"{k}={v}" for k, v in request_params.items()]
            ),
        )

        # Parse API responses.
        entities_info = request.json()
        # Implementing response continuation would allow larger batch sizes and do away with the need for complete
        # batches. Couldn't get that work properly yet.
        assert entities_info["batchcomplete"] == ""
        # Titles might be normalized and/or redirected by the Wikipedia API. Use .lower() to avoid capitalization
        # differences for lookups.
        normalizations_from_to = {
            entry["from"]: entry["to"]
            for entry in entities_info["query"].get("normalized", {})
        }
        normalizations_to_from = {
            v.lower(): k for k, v in normalizations_from_to.items()
        }
        redirections_to_from = {
            entry["to"]: entry["from"]
            for entry in entities_info["query"].get("redirects", {})
        }
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
            entity_title = normalizations_to_from.get(
                entity_title.lower(), entity_title
            ).replace(" ", "_")

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
                    _entities[qid].categories = _prune_category_titles(
                        entity_info["categories"]
                    )
                    _entities[qid].pageviews = sum(
                        [int(vc) for vc in entity_info["pageviews"].values() if vc]
                    )
                else:
                    _entities[qid].names.add(entity_title)
                    _entities[qid].frequency += _entities.pop(entity_title).frequency
                _entities[qid].names |= (
                    redirection_names
                    | set(entity_info["terms"].get("alias", {}))
                    | set(entity_info["terms"].get("label", {}))
                )
                _entities[qid].description = entity_info["extract"]
                _entities[qid].short_description = " ".join(
                    entity_info["terms"].get("description", [])
                )
                title_qid_mappings[entity_title] = qid
            except KeyError:
                failed_lookups.add(entity_title)

        # The Wikipedia API only returns one result if two mentions in the same batch are resolved to the same
        # entity via redirects. So if any entities haven't been updated, this is counted as a failed lookup.
        for entity_title in [
            et for et in chunk if et in _entities and et not in failed_lookups
        ]:
            failed_lookups.add(entity_title)

        pbar.update(len(chunk))
    pbar.close()

    for qid in _entities:
        _entities[qid].names = {name.replace("_", " ") for name in _entities[qid].names}

    return _entities, failed_lookups, title_qid_mappings


def enrich_entities(
    entities: Dict[str, Entity], annotations: Dict[str, List[Annotation]]
) -> Tuple[Dict[str, Entity], Set[str], Dict[str, List[Annotation]]]:
    """Requests additional information from Wikipedia/Wikidata API to enrich entity data. Updates annotations with
    Wiki QID.
    entities (Dict[str, Entity]): All entities found in annotations.
    annotations (Dict[str, List[Annotations]): All annotations per document ID.
    RETURNS (Tuple[Dict[str, Entity], Set[str], Dict[str, List[Annotation]]]):
    """

    # Fetch Wikidata IDs (QIDs). Some entities won't be resolved properly because of messy situations with redirects
    # and normalizations (e.g.: two different titles are redirected to the same entity, Wikipedia only returns this
    # one entity. Associating the remaining title with the correct entity can bloat up the code).
    # Since we don't expect many failures, we instead run failed lookups again individually. This should avoid any
    # situations with entity interdependencies at the cost of lookup speed.
    entities, failed_entity_lookups, title_qid_mappings = _resolve_wiki_titles(
        entities, batch_size=5
    )
    if len(failed_entity_lookups):
        entities, failed_entity_lookups, _title_qid_mapping = _resolve_wiki_titles(
            entities=entities,
            entity_titles=failed_entity_lookups,
            batch_size=1,
            progress_bar_desc=f"Trying to salvage {len(failed_entity_lookups)} failed lookups",
        )
        title_qid_mappings = {**title_qid_mappings, **_title_qid_mapping}
    for entity_title in failed_entity_lookups:
        entities.pop(entity_title)

    # Update annotations with corresponding entity IDs.
    for source_id in annotations:
        for annotation in annotations[source_id]:
            if annotation.entity_name not in failed_entity_lookups:
                annotation.entity_id = title_qid_mappings[annotation.entity_name]

    return entities, failed_entity_lookups, annotations

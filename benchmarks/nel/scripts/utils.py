"""
Various utilities for benchmarking.
"""

import copy
import urllib.parse
from typing import Dict, Tuple, Union, List, Set, Optional

import tqdm
import requests

# todo @RM replace with pydantic schemas
ENTITIES_TYPE = Dict[str, Dict[str, Union[Set[str], str, int]]]
ANNOTATIONS_TYPE = Dict[str, List[Dict[str, Union[Set[str], str, int]]]]

# Max. allowed Wikipedia API batch size with descriptions.
MAX_WIKI_API_BATCH_SIZE = 20


def resolve_wiki_titles(
    entity_data: ENTITIES_TYPE,
    entity_titles: Optional[List[str]] = None,
    batch_size: int = MAX_WIKI_API_BATCH_SIZE
) -> Tuple[ENTITIES_TYPE, Set[str], Dict[str, str]]:
    """
    Resolves Wikipedia titles to Wikidata IDs. Also fetches descriptions for entities.

    entity_data (Dict[str, Dict[str, Union[str, int]]]): Entity data. An updated copy will be returned by this function.
    entity_titles (List[str]): List of entity titles to resolve. Set to list(entity_data.keys()), if None.
    batch_size (int): Number of entity titles to resolve in the same API request. Between 1 and 50.
    RETURNS (Tuple[Dict[str, Dict[str, Union[str, int]]], Set[str]], Dict[str, str]): Updated copy of entity_data; set
        of titles of failed lookups; mapping of titles to entity IDs.
    """

    assert 1 <= batch_size <= MAX_WIKI_API_BATCH_SIZE, f"Batch size has to be between 1 and {MAX_WIKI_API_BATCH_SIZE}."
    entity_titles = entity_titles if entity_titles is not None else list(entity_data.keys())
    pbar = tqdm.tqdm(total=len(entity_titles))
    failed_lookups: Set[str] = set()
    data = copy.deepcopy(entity_data)
    title_qid_mappings: Dict[str, str] = {}

    for i in range(0, len(entity_titles), batch_size):
        # Resolve article names to Wikidata QIDs. See
        # https://stackoverflow.com/questions/37024807/how-to-get-wikidata-id-for-an-wikipedia-article-by-api and
        # https://stackoverflow.com/questions/8555320/is-there-a-wikipedia-api-just-for-retrieve-the-content-summary.
        chunk = entity_titles[i:i + batch_size]
        request_params = {
            "action": "query",
            "prop": "pageprops|extracts",
            "ppprop": "wikibase_item",
            # Some special chars such as & are not escaped by requests, so we do that manually.
            "titles": urllib.parse.quote("|".join(chunk)),
            "format": "json",
            "exintro": None,
            "explaintext": None,
            # Make sure descriptions are available for all requested articles.
            "exlimit": len(chunk),
            "redirects": 1
        }
        # 'Special_Counsel_investigation_(2017–2019)', 'Ball', 'Amyloid_plaque', 'Ionizing_radiation', 'Ethiopian'
        request = requests.get(
            "https://en.wikipedia.org/w/api.php",
            # Reformat to keep flags/parameters without values.
            params='&'.join([k if v is None else f"{k}={v}" for k, v in request_params.items()])
        )

        # Parse API responses.
        entities_info = request.json()
        assert entities_info["batchcomplete"] == ""
        # Titles might be normalized and/or redirected by the Wikipedia API. Use .lower() to avoid capitalization
        # differences for lookups.
        normalizations_from_to = {entry["from"]: entry["to"] for entry in entities_info["query"].get("normalized", {})}
        normalizations_to_from = {v.lower(): k for k, v in normalizations_from_to.items()}
        redirections_to_from = {entry["to"]: entry["from"] for entry in entities_info["query"].get("redirects", {})}
        entities_info = entities_info["query"]["pages"]

        # Replace entity titles keys in dict with Wikidata QIDs. Add entity description.
        for page_id, entity_info in entities_info.items():
            # Fetch original (non-normalized, non-redirected) entity title to ensure correct association of Wikidata ID
            # with entity.
            entity_title = redirections_to_from.get(entity_info["title"], entity_info["title"])
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
                if qid not in data:
                    data[qid] = data.pop(entity_title)
                else:
                    data[qid]["names"].add(entity_title)
                    data[qid]["frequency"] += data.pop(entity_title)["frequency"]
                data[qid]["description"] = entity_info["extract"]
                title_qid_mappings[entity_title] = qid
            except KeyError:
                failed_lookups.add(entity_title)

        # The Wikipedia API only returns one result if two mentions in the same batch are resolved to the same entity
        # via redirects. So if any entities haven't been updated, this is counted as a failed lookup.
        for entity_title in [et for et in chunk if et in data and et not in failed_lookups]:
            failed_lookups.add(entity_title)

        pbar.update(len(chunk))

    pbar.close()

    return data, failed_lookups, title_qid_mappings

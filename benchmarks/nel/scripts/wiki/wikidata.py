""" Functionalities for processing Wikidata dump.
Modified from https://github.com/explosion/projects/blob/master/nel-wikipedia/wikidata_processor.py.
"""

import bz2
import json
import sqlite3
from pathlib import Path
from typing import Union, Optional, Dict, Tuple, Any, List, Set
import tqdm

from wiki.namespaces import WD_META_ITEMS


def read_entities(
    wikidata_file: Union[str, Path],
    db_conn: sqlite3.Connection,
    batch_size: int = 5000,
    limit: Optional[int] = None,
    to_print: bool = False,
    lang: str = "en",
    parse_descr: bool = True,
    parse_properties: bool = True,
    parse_sitelinks: bool = True,
    parse_labels: bool = True,
    parse_aliases: bool = True,
    parse_claims: bool = True,
) -> None:
    """Reads entity information from wikidata dump.
    wikidata_file (Union[str, Path]): Path of wikidata dump file.
    db_conn (sqlite3.Connection): DB connection.
    batch_size (int): Batch size for DB commits.
    limit (Optional[int]): Max. number of entities to parse.
    to_print (bool): Whether to print information during the parsing process.
    lang (str): Language with which to filter entity information.
    parse_descr (bool): Whether to parse entity descriptions.
    parse_properties (bool): Whether to parse entity properties.
    parse_sitelinks (bool): Whether to parse entity sitelinks.
    parse_labels (bool): Whether to parse entity labels.
    parse_aliases (bool): Whether to parse entity aliases.
    parse_claims (bool): Whether to parse entity claims.
    """

    # Read the JSON wiki data and parse out the entities. Takes about 7-10h to parse 55M lines.
    # get latest-all.json.bz2 from https://dumps.wikimedia.org/wikidatawiki/entities/

    site_filter = "{}wiki".format(lang)

    # filter: currently defined as OR: one hit suffices to be removed from further processing
    exclude_list = WD_META_ITEMS

    # punctuation
    exclude_list.extend(["Q1383557", "Q10617810"])

    # letters etc
    exclude_list.extend(
        ["Q188725", "Q19776628", "Q3841820", "Q17907810", "Q9788", "Q9398093"]
    )

    neg_prop_filter = {
        "P31": exclude_list,  # instance of
        "P279": exclude_list,  # subclass
    }

    title_to_id: Dict[str, str] = {}
    id_to_attrs: Dict[str, Dict[str, Any]] = {}

    with bz2.open(wikidata_file, mode="rb") as file:
        pbar_params = {"total": limit} if limit else {}
        with tqdm.tqdm(desc="Parsing entity data", leave=True, **pbar_params) as pbar:
            for cnt, line in enumerate(file):
                if limit and cnt >= limit:
                    break
                clean_line = line.strip()
                if clean_line.endswith(b","):
                    clean_line = clean_line[:-1]
                if len(clean_line) > 1:
                    obj = json.loads(clean_line)
                    entry_type = obj["type"]

                    if entry_type == "item":
                        keep = True

                        claims = obj["claims"]
                        filtered_claims: List[Dict[str, str]] = []
                        if parse_claims:
                            for prop, value_set in neg_prop_filter.items():
                                claim_property = claims.get(prop, None)
                                if claim_property:
                                    filtered_claims.append(claim_property)
                                    for cp in claim_property:
                                        cp_id = (
                                            cp["mainsnak"]
                                            .get("datavalue", {})
                                            .get("value", {})
                                            .get("id")
                                        )
                                        cp_rank = cp["rank"]
                                        if (
                                            cp_rank != "deprecated"
                                            and cp_id in value_set
                                        ):
                                            keep = False

                        if keep:
                            unique_id = obj["id"]
                            if unique_id not in id_to_attrs:
                                id_to_attrs[unique_id] = {}
                            if parse_claims:
                                id_to_attrs[unique_id]["claims"] = filtered_claims

                            if to_print:
                                print("ID:", unique_id)
                                print("type:", entry_type)

                            # parsing all properties that refer to other entities
                            if parse_properties:
                                id_to_attrs[unique_id]["properties"] = []
                                for prop, claim_property in claims.items():
                                    cp_dicts = [
                                        cp["mainsnak"]["datavalue"].get("value")
                                        for cp in claim_property
                                        if cp["mainsnak"].get("datavalue")
                                    ]
                                    cp_values = [
                                        cp_dict.get("id")
                                        for cp_dict in cp_dicts
                                        if isinstance(cp_dict, dict)
                                        if cp_dict.get("id") is not None
                                    ]
                                    if cp_values:
                                        if to_print:
                                            print("prop:", prop, cp_values)
                                        id_to_attrs[unique_id]["properties"].append(
                                            (prop, cp_values)
                                        )

                            found_link = False
                            if parse_sitelinks:
                                site_value = obj["sitelinks"].get(site_filter, None)
                                if site_value:
                                    site = site_value["title"]
                                    if to_print:
                                        print(site_filter, ":", site)
                                    title_to_id[site] = unique_id
                                    found_link = True
                                    id_to_attrs[unique_id]["sitelinks"] = site_value

                            if parse_labels:
                                labels = obj["labels"]
                                if labels:
                                    lang_label = labels.get(lang, None)
                                    if lang_label:
                                        if to_print:
                                            print(
                                                "label (" + lang + "):",
                                                lang_label["value"],
                                            )
                                        id_to_attrs[unique_id]["labels"] = lang_label

                            if found_link and parse_descr:
                                descriptions = obj["descriptions"]
                                if descriptions:
                                    lang_descr = descriptions.get(lang, None)
                                    if lang_descr:
                                        if to_print:
                                            print(
                                                "description (" + lang + "):",
                                                lang_descr["value"],
                                            )
                                        id_to_attrs[unique_id][
                                            "description"
                                        ] = lang_descr["value"]

                            if parse_aliases:
                                id_to_attrs[unique_id]["aliases"] = []
                                aliases = obj["aliases"]
                                if aliases:
                                    lang_aliases = aliases.get(lang, None)
                                    if lang_aliases:
                                        for item in lang_aliases:
                                            if to_print:
                                                print(
                                                    "alias (" + lang + "):",
                                                    item["value"],
                                                )
                                            id_to_attrs[unique_id]["aliases"].append(
                                                item["value"]
                                            )

                pbar.update(1)

                # Save batch.
                if pbar.n % batch_size == 0:
                    _write_to_db(db_conn, title_to_id, id_to_attrs)
                    title_to_id = {}
                    id_to_attrs = {}

    if pbar.n % batch_size != 0:
        _write_to_db(db_conn, title_to_id, id_to_attrs)


def _write_to_db(
    db_conn: sqlite3.Connection,
    title_to_id: Dict[str, str],
    id_to_attrs: Dict[str, Dict[str, Any]],
) -> None:
    """Persists entity information to database.
    db_conn (Connection): Database connection.
    title_to_id (Dict[str, str]): Titles to QIDs.
    id_to_attrs (Dict[str, Dict[str, Any]]): For QID a dictionary with property name to property value(s).
    """

    entities: List[Tuple[Optional[str], ...]] = []
    props_in_ents: Set[Tuple[str, str, str]] = set()
    for title, qid in title_to_id.items():
        entities.append(
            (
                qid,
                title,
                id_to_attrs[qid].get("description", None),
                id_to_attrs[qid].get("labels", {}).get("value", None),
                json.dumps(id_to_attrs[qid]["aliases"]),
                json.dumps(id_to_attrs[qid]["claims"]),
            )
        )
        for prop in id_to_attrs[qid]["properties"]:
            for second_qid in prop[1]:
                props_in_ents.add((prop[0], qid, second_qid))

    cur = db_conn.cursor()
    cur.executemany(
        "INSERT INTO entities (id, name, description, label, aliases, claims) VALUES (?, ?, ?, ?, ?, ?)",
        entities,
    )
    cur.executemany(
        "INSERT INTO properties_in_entities (property_id, from_entity_id, to_entity_id) VALUES (?, ?, ?)",
        props_in_ents,
    )
    db_conn.commit()

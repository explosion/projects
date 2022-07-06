""" Wiki dataset for unified access to information from Wikipedia and Wikidata dumps. """
import os.path
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List, Set
import sqlite3

from schemas import Entity
from wiki import wikidata, wikipedia

_assets_dir = Path(os.path.abspath(__file__)).parent.parent.parent / "assets" / "wiki"
_paths = {
    "db": _assets_dir / "wiki.sqlite3",
    "wikidata_dump": _assets_dir / "wikidata_entity_dump.json.bz2",
    "wikipedia_dump": _assets_dir / "wikipedia_dump.xml.bz2",
}


def establish_db_connection() -> sqlite3.Connection:
    """Estabished database connection.
    RETURNS (sqlite3.Connection): Database connection.
    """
    db_conn = sqlite3.connect(_paths["db"])
    db_conn.row_factory = sqlite3.Row
    return db_conn


def parse(
    db_conn: Optional[sqlite3.Connection] = None,
    entity_config: Optional[Dict[str, Any]] = None,
    article_text_config: Optional[Dict[str, Any]] = None,
    alias_prior_prob_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Parses Wikipedia and Wikidata dumps. Writes parsing results to a database. Note that this takes hours.
    db_conn (Optional[sqlite3.Connection]): Database connection.
    entity_config (Dict[str, Any]): Arguments to be passed on to wikidata.read_entities().
    article_text_config (Dict[str, Any]): Arguments to be passed on to wikipedia.read_text().
    alias_prior_prob_config (Dict[str, Any]): Arguments to be passed on to wikipedia.read_prior_probs().
    """

    msg = "Database exists already. Execute `spacy project run delete_wiki_db` to remove it."
    assert not os.path.exists(_paths["db"]), msg

    db_conn = db_conn if db_conn else establish_db_connection()
    with open(Path(os.path.abspath(__file__)).parent / "ddl.sql", "r") as ddl_sql:
        db_conn.cursor().executescript(ddl_sql.read())

    wikidata.read_entities(
        _paths["wikidata_dump"],
        db_conn,
        **(entity_config if entity_config else {}),
    )

    wikipedia.read_prior_probs(
        _paths["wikipedia_dump"],
        db_conn,
        **(alias_prior_prob_config if alias_prior_prob_config else {}),
    )

    wikipedia.read_texts(
        _paths["wikipedia_dump"],
        db_conn,
        **(article_text_config if article_text_config else {}),
    )


def load_entities(
    key: str, values: Tuple[str, ...], db_conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Entity]:
    """Loads information for entity or entities by querying information from DB.
    Note that this doesn't return all available information, only the part used in the current benchmark solution.
    key (str): Attribute to match values to. Must be one of ("id", "name").
    values (Tuple[str]): Values for key to look up.
    db_conn (Optional[sqlite3.Connection]): Database connection.
    RETURNS (Dict[str, Entity]): Information on requested entities.
    """

    assert key in ("id", "name")
    db_conn = db_conn if db_conn else establish_db_connection()

    return {
        rec["id"]: Entity(
            qid=rec["id"],
            name=rec["entity_title"],
            aliases={
                alias
                for alias in {
                    rec["entity_title"],
                    rec["article_title"],
                    rec["label"],
                    *(rec["aliases"] if rec["aliases"] else "").split(","),
                }
                if alias
            },
            article_title=rec["article_title"],
            article_text=rec["text"],
            description=rec["description"],
            count=rec["count"] if rec["count"] else 0,
        )
        for rec in db_conn.cursor().execute(
            f"""
                SELECT 
                    e.id,
                    e.name as entity_title,
                    e.description,
                    e.label,
                    a.title as article_title,
                    a.text,
                    GROUP_CONCAT(afe.alias) as aliases,
                    SUM(afe.count) as count
                FROM 
                    entities e
                LEFT JOIN articles a on
                    a.entity_id = e.id
                LEFT JOIN aliases_for_entities afe on
                    afe.entity_id = e.id                                         
                WHERE 
                    e.{key} IN (%s)
                GROUP BY
                    e.id,
                    e.name,
                    e.description,
                    e.label,
                    a.title,
                    a.text
            """
            % ",".join("?" * len(values)),
            tuple(set(values)),
        )
    }


def load_alias_entity_prior_probabilities(
    entity_ids: Set[str], db_conn: Optional[sqlite3.Connection] = None
) -> Dict[str, List[Tuple[str, float]]]:
    """Loads alias-entity counts from database and transforms them into prior probabilities per alias.
    entity_ids (Set[str]): Set of entity IDs to allow.
    RETURN (Dict[str, Tuple[Tuple[str, ...], Tuple[float, ...]]]): Mapping of alias to tuples of entities and the
        corresponding prior probabilities.
    """

    db_conn = db_conn if db_conn else establish_db_connection()

    alias_entity_prior_probs = {
        rec["alias"]: [
            (entity_id, int(count))
            for entity_id, count in zip(rec["entity_ids"].split(","), rec["counts"].split(","))
        ]
        for rec in db_conn.cursor().execute(
            """
                SELECT 
                    alias,
                    GROUP_CONCAT(entity_id) as entity_ids,
                    GROUP_CONCAT(count) as counts
                FROM 
                    aliases_for_entities                                   
                WHERE 
                    entity_id IN (%s)
                GROUP BY
                    alias
            """
            % ",".join("?" * len(entity_ids)),
            tuple(entity_ids),
        )
    }

    for alias, entity_counts in alias_entity_prior_probs.items():
        total_count = sum([ec[1] for ec in entity_counts])
        alias_entity_prior_probs[alias] = [(ec[0], ec[1] / max(total_count, 1)) for ec in entity_counts]

    return alias_entity_prior_probs

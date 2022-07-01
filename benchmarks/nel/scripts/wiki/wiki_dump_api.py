""" Wiki dataset for unified access to information from Wikipedia and Wikidata dumps. """
import os.path
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, Union
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
    force: bool = False,
    entity_config: Optional[Dict[str, Any]] = None,
    article_text_config: Optional[Dict[str, Any]] = None,
    alias_prior_prob_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Parses Wikipedia and Wikidata dumps. Writes parsing results to file. Note that this takes hours.
    db_conn (Optional[sqlite3.Connection]): Database connection.
    force (bool): Whether to parse dumps even if all artefacts are available. Note: this deletes the entire database
        file and builds it from scratch.
    entity_config (Dict[str, Any]): Arguments to be passed on to wikidata.read_entities().
    article_text_config (Dict[str, Any]): Arguments to be passed on to wikipedia.read_text().
    alias_prior_prob_config (Dict[str, Any]): Arguments to be passed on to wikipedia.read_prior_probs().
    """

    if os.path.exists(_paths["db"]):
        if not force:
            return
        else:
            # Database exists and is overwritten.
            if db_conn:
                db_conn.close()
                db_conn = None
            os.remove(_paths["db"])

    db_conn = db_conn if db_conn else establish_db_connection()

    with open("ddl.sql", "r") as ddl_sql:
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


def load_entities(key: str, values: Tuple[str, ...], db_conn: Optional[sqlite3.Connection] = None) -> Dict[str, Entity]:
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


if __name__ == "__main__":
    parse(
        force=True,
        entity_config={"limit": 1000000},
        article_text_config={"limit": 100000},
        alias_prior_prob_config={"limit": 10000000},
    )
    load_entities("id", ("Q10",))


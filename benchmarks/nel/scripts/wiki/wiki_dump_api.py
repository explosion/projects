""" Wiki dataset for unified access to information from Wikipedia and Wikidata dumps. """
import csv
import os.path
import pickle
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, Union, Set, Iterable
import sqlite3

from schemas import Entity
from wiki import wikidata, wikipedia


class WikiDumpAPI:
    """Contains and makes available data from Wikipedia and Wikidata dumps, as well as functionality for parsing those
    dumps and writing the parsing results to disk."""

    def __init__(self):
        """Creates new Wikidataset instance. If assets_dir contains all artefacts produced by parsing process, parsing
        is skipped completely.
        """

        self._entity_title_to_id: Optional[Dict[str, str]] = None
        self._entity_attributes: Optional[Dict[str, Dict[str, Any]]] = None
        self._alias_entity_prior_probs: Optional[Dict[str, Dict[str, int]]] = None
        self._article_texts: Optional[Dict[str, Dict[str, str]]] = None
        assets_dir = (
            Path(os.path.abspath(__file__)).parent.parent.parent / "assets" / "wiki"
        )
        self._paths = {
            "db": assets_dir / "wiki.sqlite3",
            "wikidata_dump": assets_dir / "wikidata_entity_dump.json.bz2",
            "wikipedia_dump": assets_dir / "wikipedia_dump.xml.bz2",
        }

        self._db_conn: Optional[sqlite3.Connection] = None
        if os.path.exists(self._paths["db"]):
            self._establish_db_connection()

    def _establish_db_connection(self) -> None:
        """Estabished database connection."""
        self._db_conn = sqlite3.connect(self._paths["db"])
        self._db_conn.row_factory = sqlite3.Row

    def parse(
        self,
        force: bool = False,
        entity_config: Optional[Dict[str, Any]] = None,
        article_text_config: Optional[Dict[str, Any]] = None,
        alias_prior_prob_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Parses Wikipedia and Wikidata dumps. Writes parsing results to file. Note that this takes hours.
        force (bool): Whether to parse dumps even if all artefacts are available. Note: this deletes the entire database
            file and builds it from scratch.
        entity_config (Dict[str, Any]): Arguments to be passed on to wikidata.read_entities().
        article_text_config (Dict[str, Any]): Arguments to be passed on to wikipedia.read_text().
        alias_prior_prob_config (Dict[str, Any]): Arguments to be passed on to wikipedia.read_prior_probs().
        """

        if self._db_conn:
            if force:
                self._db_conn.close()
                os.remove(self._paths["db"])
            else:
                return

        self._establish_db_connection()
        with open("ddl.sql", "r") as ddl_sql:
            self._db_conn.cursor().executescript(ddl_sql.read())

        wikidata.read_entities(
            self._paths["wikidata_dump"],
            self._db_conn,
            **(entity_config if entity_config else {}),
        )

        wikipedia.read_prior_probs(
            self._paths["wikipedia_dump"],
            self._db_conn,
            **(alias_prior_prob_config if alias_prior_prob_config else {}),
        )

        wikipedia.read_texts(
            self._paths["wikipedia_dump"],
            self._db_conn,
            **(article_text_config if article_text_config else {}),
        )

    def __getitem__(self, *keys: Tuple[str, ...]) -> Union[Entity, Dict[str, Entity]]:
        """Returns basic information for entity or entities by querying information from DB.
        Note that this doesn't return all available information, only the part used in the current benchmark solution.
        key (Tuple[str]): Entity ID(s) to look up.
        RETURNS (Union[Entity, Dict[str, Entity]]): Information on requested entity/entities.
        """

        if self._db_conn is None:
            raise ValueError(f"Database not found at {self._paths['db']}.")

        keys = keys[0]
        entities = {
            rec["id"]: Entity(
                qid=rec["id"],
                aliases={
                    alias
                    for alias in {
                        rec["entity_title"],
                        rec["article_title"],
                        rec["label"],
                        *rec["aliases"].split(","),
                    }
                    if alias
                },
                article_title=rec["article_title"],
                article_text=rec["text"],
                description=rec["description"],
                count=rec["count"],
            )
            for rec in self._db_conn.cursor().execute(
                """
                    SELECT 
                        e.id,
                        e.title as entity_title,
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
                        e.id IN (%s)
                    GROUP BY
                        e.id,
                        e.title,
                        e.description,
                        e.label,
                        a.title,
                        a.text
                """
                % ",".join("?" * len(keys)),
                tuple(set(keys)),
            )
        }

        return entities[keys[0]] if len(keys) == 1 else entities


if __name__ == "__main__":
    wd = WikiDumpAPI()
    wd.parse(
        force=True,
        entity_config={"limit": 1000},
        article_text_config={"limit": 1000},
        alias_prior_prob_config={"limit": 1000},
    )

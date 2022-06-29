""" Wiki dataset for unified access to information from Wikipedia and Wikidata dumps. """
import csv
import os.path
import pickle
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import sqlite3

from wiki import wikidata, wikipedia


class WikiDataset:
    """ Contains and makes available data from Wikipedia and Wikidata dumps, as well as functionality for parsing those
    dumps and writing the parsing results to disk. """

    def __init__(self):
        """ Creates new Wikidataset instance. If assets_dir contains all artefacts produced by parsing process, parsing
        is skipped completely.
        """

        self._entity_title_to_id: Optional[Dict[str, str]] = None
        self._entity_attributes: Optional[Dict[str, Dict[str, Any]]] = None
        self._alias_entity_prior_probs: Optional[Dict[str, Dict[str, int]]] = None
        self._article_texts: Optional[Dict[str, Dict[str, str]]] = None
        assets_dir = Path(os.path.abspath(__file__)).parent.parent.parent / "assets" / "wiki"
        self._paths = {
            "db": assets_dir / "wiki.sqlite3",
            "wikidata_dump": assets_dir / "wikidata_entity_dump.json.bz2",
            "wikipedia_dump": assets_dir / "wikipedia_dump.xml.bz2"
        }

        self._db_conn: Optional[sqlite3.Connection] = None
        if os.path.exists(self._paths["db"]):
            self._db_conn = sqlite3.connect(self._paths["db"])

    def load(self) -> None:
        """ Loads all parsing artefacts, if available. Raises an exception if not all are available.
        """

        if not self._db_conn:
            raise ValueError("Parsed data not available. Run .parse_dumps().")

        # Load parsing artefacts.
        # todo measure memory impact. acceptable or do we have to switch to file indexing (.h5?)?
        with open(self._paths["entity_title_to_id"], 'wb') as handle:
            self._entity_title_to_id = pickle.load(handle)
        with open(self._paths["entity_attributes"], 'wb') as handle:
            self._entity_attributes = pickle.load(handle)
        with open(self._paths["article_texts"], "r") as handle:
            self._article_texts = {
                line["entity_id"]: {"title": line["title"], "text": line["text"]}
                for line in csv.DictReader(handle, delimiter=',', quotechar='"')
            }
        with open(self._paths["article_texts"], "r") as handle:
            self._article_texts = {
                line["entity_id"]: {"title": line["title"], "text": line["text"]}
                for line in csv.DictReader(handle, delimiter=',', quotechar='"')
            }
        with open(self._paths["entity_alias_prior_probs"], "r") as handle:
            self._alias_entity_prior_probs = {}
            for line in csv.DictReader(handle, delimiter=',', quotechar='"'):
                if line["alias"] not in self._alias_entity_prior_probs:
                    self._alias_entity_prior_probs[line["alias"]] = {}
                assert line["entity_id"] not in self._alias_entity_prior_probs[line["alias"]]
                self._alias_entity_prior_probs[line["alias"]][line["entity_id"]] = line["count"]

    def parse(
        self,
        force: bool = False,
        entity_parse_kwargs: Optional[Dict[str, Any]] = None,
        article_text_parse_kwargs: Optional[Dict[str, Any]] = None,
        entity_alias_prior_prob_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """ Parses Wikipedia and Wikidata dumps. Writes parsing results to file. Note that this takes hours.
        force (bool): Whether to parse dumps even if all artefacts are available. Note: this deletes the entire database
            file and builds it from scratch.
        entity_parse_kwargs (Dict[str, Any]): Arguments to be passed on to wikidata.read_entities().
        article_text_parse_kwargs (Dict[str, Any]): Arguments to be passed on to wikipedia.read_text().
        entity_alias_prior_prob_kwargs (Dict[str, Any]): Arguments to be passed on to wikipedia.read_prior_probs().
        """

        if self._db_conn:
            if force:
                self._db_conn.close()
                os.remove(self._paths["db"])
            else:
                return

        print("Initializing database.")
        self._db_conn = sqlite3.connect(self._paths["db"])
        with open("ddl.sql", "r") as ddl_sql:
            self._db_conn.cursor().executescript(ddl_sql.read())

        # print("Parsing Wikidata dump.")
        # self._entity_title_to_id, self._entity_attributes = wikidata.read_entities(
        #     self._paths["wikidata_dump"], self._db_conn, **(entity_parse_kwargs if entity_parse_kwargs else {})
        # )

        # Note: Can be done in parallel with parsing of Wikidata dump.
        print("Parse Wikipedia dump for alias-entity prior probabilities.")
        wikipedia.read_prior_probs(
            self._paths["wikipedia_dump"],
            self._db_conn,
            **(entity_alias_prior_prob_kwargs if entity_alias_prior_prob_kwargs else {})
        )


        print("Parsing Wikidata for article texts.")
        wikipedia.read_texts(
            self._paths["wikipedia_dump"],
            self._entity_title_to_id,
            self._paths["article_texts"],
            **(article_text_parse_kwargs if article_text_parse_kwargs else {})
        )


if __name__ == '__main__':
    wd = WikiDataset()
    wd.parse(force=True)
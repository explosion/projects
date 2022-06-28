""" Wiki dataset for unified access to information from Wikipedia and Wikidata dumps. """
import csv
import os.path
import pickle
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

from wiki import wikidata, wikipedia


class WikiDataset:
    """ Contains and makes available data from Wikipedia and Wikidata dumps, as well as functionality for parsing those
    dumps and writing the parsing results to disk. """

    def __init__(self, assets_dir: Path):
        """ Creates new Wikidataset instance. If assets_dir contains all artefacts produced by parsing process, parsing
        is skipped completely.
        assets_dir (Path): Path of directory with Wiki assets. This includes the actual dumps as well as the artefacts
            produced by the parsing process.
        """

        self._entity_title_to_id: Optional[Dict[str, str]] = None
        self._entity_attributes: Optional[Dict[str, Dict[str, Any]]] = None
        self._alias_entity_prior_probs: Optional[Dict[str, Dict[str, int]]] = None
        self._article_texts: Optional[Dict[str, Dict[str, str]]] = None
        self._paths = {
            "entity_title_to_id": assets_dir / "entity_title_to_id.pkl",
            "entity_attributes": assets_dir / "entity_attributes.pkl",
            "alias_entity_prior_probs": assets_dir / "alias_entity_prior_probs.csv",
            "article_texts": assets_dir / "article_texts.csv",
            "wikidata_dump": assets_dir / "wikidata_entity_dump.json.bz2",
            "wikipedia_dump": assets_dir / "wikipedia_dump.xml.bz2"
        }

    def _are_parsing_artefacts_complete(self) -> bool:
        """ Checks whether all parsing artefacts are available.
        RETURNS (bool): Whether all parsing artefacts are available.
        """

        return all([os.path.exists(path) for path in self._paths.values()])

    def load(self) -> None:
        """ Loads all parsing artefacts, if available. Raises an exception if not all are available.
        """

        if not self._are_parsing_artefacts_complete():
            raise ValueError("Parsing artefacts incomplete. Run .parse_dumps().")

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

    def parse_dumps(
        self,
        force: bool = False,
        entity_parse_kwargs: Optional[Dict[str, Any]] = None,
        article_text_parse_kwargs: Optional[Dict[str, Any]] = None,
        entity_alias_prior_prob_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """ Parses Wikipedia and Wikidata dumps. Writes parsing results to file. Note that this takes hours.
        force (bool): Whether to parse dumps even if all artefacts are available.
        entity_parse_kwargs (Dict[str, Any]): Arguments to be passed on to wikidata.read_entities().
        article_text_parse_kwargs (Dict[str, Any]): Arguments to be passed on to wikipedia.read_text().
        entity_alias_prior_prob_kwargs (Dict[str, Any]): Arguments to be passed on to wikipedia.read_prior_probs().
        """

        if self._are_parsing_artefacts_complete() and force is False:
            return

        print("Parsing Wikidata dump.")
        self._entity_title_to_id, self._entity_attributes = wikidata.read_entities(
            self._paths["wikidata_dump"], **entity_parse_kwargs
        )
        with open(self._paths["entity_title_to_id"], 'wb') as handle:
            pickle.dump(self._entity_title_to_id, handle)
        with open(self._paths["entity_attributes"], 'wb') as handle:
            pickle.dump(self._entity_attributes, handle)

        print("Parsing Wikidata for article texts.")
        wikipedia.read_texts(
            self._paths["wikipedia_dump"],
            self._entity_title_to_id,
            self._paths["article_texts"],
            **article_text_parse_kwargs
        )

        print("Parse Wikipedia dump for alias-entity prior probabilities.")
        wikipedia.read_prior_probs(
            self._paths["wikipedia_dump"],
            self._paths["entity_alias_prior_probs"],
            **entity_alias_prior_prob_kwargs
        )

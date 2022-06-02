""" Dataset class. """
import abc
import copy
import os
import pickle
import urllib.parse
from pathlib import Path
from typing import Tuple, Set, List, Dict, Optional, Union

import numpy
import spacy
import yaml

import requests
import tqdm
from spacy import Language
from spacy.kb import KnowledgeBase
from spacy.tokens import Token, Doc, Span, DocBin

# todo @RM replace with pydantic schemas
ENTITIES_TYPE = Dict[str, Dict[str, Union[Set[str], str, int]]]
ANNOTATIONS_TYPE = Dict[str, List[Dict[str, Union[Set[str], str, int]]]]

# Max. allowed Wikipedia API batch size with descriptions.
MAX_WIKI_API_BATCH_SIZE = 20


def _does_token_overlap_with_annotation(token: Token, annot_start: int, annot_end: int) -> bool:
    """ Checks whether token overlaps with annotation span.
    token (Token): Token to check.
    annot_start (int): Annotation's start index.
    annot_end (int): Annotation's end index.
    RETURNS (bool): Whether token overlaps with annotation span.
    """

    return annot_start <= token.idx <= annot_end or token.idx <= annot_start <= token.idx + len(token)


class Dataset(abc.ABC):
    """ Base class for all datasets used in this benchmark. """

    def __init__(self):
        """ Initializes new Dataset.
        """

        self._root_dir = Path(os.path.abspath(__file__)).parent.parent.parent
        self._assets_dir = self._root_dir / "assets" / self.dataset_id
        self._nlp_dir = self._root_dir / "temp" / f"{self.dataset_id}.nlp"
        self._kb_dir = self._root_dir / "temp" / f"{self.dataset_id}.kb"
        self._corpora_dir = self._root_dir / "corpora" / self.dataset_id
        with open(self._root_dir / "configs" / "datasets.yml", "r") as stream:
            self._options = yaml.safe_load(stream)[self.dataset_id]
        self._entities_path = self._assets_dir / "entities.pkl"
        self._failed_entity_lookups_path = self._assets_dir / "entities_failed_lookups.pkl"
        self._annotations_path = self._assets_dir / "annotations.pkl"

        self._entities: Optional[ENTITIES_TYPE] = None
        self._failed_entity_lookups: Optional[Set[str]] = None
        self._annotations: Optional[ANNOTATIONS_TYPE] = None
        self._kb: Optional[KnowledgeBase] = None
        self._nlp: Optional[Language] = None
        self._annotated_docs: Optional[List[Doc]] = None

    @property
    def dataset_id(self) -> str:
        """ Returns dataset ID.
        """
        raise NotImplementedError

    def create_knowledge_base(self, model_name: str, **kwargs) -> None:
        """ Creates and serializes knowledge base.
        vectors_model (str): Name of model with word vectors to use.
        """

        self._nlp = spacy.load(model_name, exclude=["tagger", "lemmatizer"])
        self._entities, self._failed_entity_lookups, self._annotations = self._parse_external_corpus(**kwargs)
        print(f"Constructing knowledge base with {len(self._entities)} entries")
        kb = KnowledgeBase(vocab=self._nlp.vocab, entity_vector_length=300)
        for qid, info in self._entities.items():
            desc_doc = self._nlp(info["description"])
            kb.add_entity(entity=qid, entity_vector=desc_doc.vector, freq=info["frequency"])
            for name in info["names"]:
                kb.add_alias(alias=name.replace("_", " "), entities=[qid], probabilities=[1])

        # Serialize knowledge base & entity information.
        for to_serialize in (
            (self._entities_path, self._entities),
            (self._failed_entity_lookups_path, self._failed_entity_lookups),
            (self._annotations_path, self._annotations)
        ):
            with open(to_serialize[0], 'wb') as fp:
                pickle.dump(to_serialize[1], fp)
        kb.to_disk(self._kb_dir)
        if not os.path.exists(self._nlp_dir):
            os.mkdir(self._nlp_dir)
        self._nlp.to_disk(self._nlp_dir)

    def create_corpora(self) -> None:
        """ Creates train/dev/test corpora for Reddit entity linking dataset.
        """

        self._load_entities()
        self._load_annotations()
        self._load_nlp()
        Doc.set_extension("overlapping_annotations", default=None)
        self._annotated_docs = self._create_annotated_docs()
        self._serialize_corpora()

    def _create_annotated_docs(self) -> List[Doc]:
        """ Creates docs annotated with entities.
        RETURN (List[Doc]): List of docs reflecting all entity annotations.
        """
        raise NotImplementedError

    def _parse_external_corpus(self, **kwargs) -> Tuple[ENTITIES_TYPE, Set[str], ANNOTATIONS_TYPE]:
        """ Parses external corpus. Loads data on entities and mentions.
        Populates self._entities, self._failed_entity_lookups, self._annotations.
        RETURNS (Tuple[ENTITIES_TYPE, Set[str], ANNOTATIONS_TYPE]): entities, titles of failed entity lookups,
            annotations.
        """
        raise NotImplementedError

    @staticmethod
    def _resolve_wiki_titles(
        entities: ENTITIES_TYPE,
        entity_titles: Optional[Set[str]] = None,
        batch_size: int = MAX_WIKI_API_BATCH_SIZE
    ) -> Tuple[ENTITIES_TYPE, Set[str], Dict[str, str]]:
        """
        Resolves Wikipedia titles to Wikidata IDs. Also fetches descriptions for entities.

        entities (ENTITIES_TYPE): Entities to resolve.
        entity_titles (Set[str]): List of entity titles to resolve.
        batch_size (int): Number of entity titles to resolve in the same API request. Between 1 and 50.
        RETURNS (Dict[str, str]): Mapping of titles to entity IDs.
        """

        assert 1 <= batch_size <= MAX_WIKI_API_BATCH_SIZE, \
            f"Batch size has to be between 1 and {MAX_WIKI_API_BATCH_SIZE}."
        entity_titles = list(entity_titles if entity_titles else entities.keys())
        pbar = tqdm.tqdm(total=len(entity_titles))
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
            normalizations_from_to = {entry["from"]: entry["to"] for entry in
                                      entities_info["query"].get("normalized", {})}
            normalizations_to_from = {v.lower(): k for k, v in normalizations_from_to.items()}
            redirections_to_from = {entry["to"]: entry["from"] for entry in entities_info["query"].get("redirects", {})}
            entities_info = entities_info["query"]["pages"]

            # Replace entity titles keys in dict with Wikidata QIDs. Add entity description.
            for page_id, entity_info in entities_info.items():
                # Fetch original (non-normalized, non-redirected) entity title to ensure correct association of Wikidata
                # ID with entity.
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
                    if qid not in _entities:
                        _entities[qid] = _entities.pop(entity_title)
                    else:
                        _entities[qid]["names"].add(entity_title)
                        _entities[qid]["frequency"] += _entities.pop(entity_title)["frequency"]
                    _entities[qid]["description"] = entity_info["extract"]
                    title_qid_mappings[entity_title] = qid
                except KeyError:
                    failed_lookups.add(entity_title)

            # The Wikipedia API only returns one result if two mentions in the same batch are resolved to the same
            # entity via redirects. So if any entities haven't been updated, this is counted as a failed lookup.
            for entity_title in [et for et in chunk if et in _entities and et not in failed_lookups]:
                failed_lookups.add(entity_title)

            pbar.update(len(chunk))

        pbar.close()

        return _entities, failed_lookups, title_qid_mappings

    @staticmethod
    def _create_spans_from_doc_annotation(
        doc: Doc,
        entities_info: ENTITIES_TYPE,
        annotations: List[Dict[str, Union[Set[str], str, int]]],
        entities_failed_lookups: Set[str]
    ) -> Tuple[List[Span], List[Dict[str, Union[Set[str], str, int]]]]:
        """ Creates spans from annotations for one document.
        doc (Doc): Document for whom to create spans.
        entities_info (ENTITIES_TYPE): All available entities.
        annotation (List[Dict[str, Union[Set[str], str, int]]]): Annotations for this post/comment.
        entities_entities_failed_lookups (Set[str]): Set of entity names for whom Wiki API lookup failed).
        source_id (str): Unique source ID to look up annotation.
        RETURNS (Tuple[List[Span], List[Dict[str, Union[Set[str], str, int]]]]): List of doc spans for annotated entities;
            list of overlapping entities.
        """

        doc_annots: List[Dict[str, Union[Set[str], str, int]]] = []
        overlapping_doc_annotations: List[Dict[str, Union[Set[str], str, int]]] = []
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
            # The Reddit EL dataset has some indexing mistakes that may result in . We align the annotation indexing
            # with spaCy's token indices.
            for t in doc:
                if _does_token_overlap_with_annotation(t, annot["start_pos"], annot["end_pos"]):
                    annot["start_pos"] = t.idx
                    break
            for t in reversed([t for t in doc]):
                if _does_token_overlap_with_annotation(t, annot["start_pos"], annot["end_pos"] - 1):
                    annot["end_pos"] = t.idx + len(t)
                    break

            # If there is an overlap between annotation's start and end position and this token's parsed start
            # and end, we try to create a span with this token's position.
            overlaps = False
            if annot["frequency"] == -1:
                assert annot["entity_id"] not in entities_info and annot["name"] in entities_failed_lookups
                continue
            for j in range(0, len(doc_annots)):
                if not (annot["end_pos"] < doc_annots[j]["start_pos"] or annot["start_pos"] > doc_annots[j]["end_pos"]):
                    overlaps = True
                    overlapping_doc_annotations.append(annot)
                    break
            if not overlaps:
                doc_annots.append(annot)

        doc_spans = [
            doc.char_span(annot["start_pos"], annot["end_pos"], label=annot["name"], kb_id=annot["entity_id"])
            for annot in doc_annots
        ]
        assert all([span is not None for span in doc_spans])

        return doc_spans, overlapping_doc_annotations

    def _serialize_corpora(self) -> None:
        """ Serializes corpora.
        docs (List[Doc]): List of documents with entity annotations.
        """

        assert self._options["frac_train"] + self._options["frac_dev"] + self._options["frac_test"] == 1

        indices = {
            dataset: idx
            for dataset, idx in zip(
                ("train", "dev", "test"),
                numpy.split(
                    numpy.asarray(range(len(self._annotated_docs))),
                    [
                        int(self._options["frac_train"] * len(self._annotated_docs)),
                        int((self._options["frac_train"] + self._options["frac_dev"]) * len(self._annotated_docs))
                    ]
                )
            )
        }

        for key in indices:
            corpus = DocBin(store_user_data=True)
            for idx in indices[key]:
                corpus.add(self._annotated_docs[idx])
            if not self._corpora_dir.exists():
                self._corpora_dir.mkdir()
            corpus.to_disk(self._corpora_dir / f"{key}.spacy")

    def _load_entities(self, force: bool = False) -> None:
        """ Loads serialized, preprocessed entity data. Includes failed entity lookups.
        force (bool): Load from disk even if already not None.
        """

        if force or not self._entities:
            with open(self._entities_path, "rb") as file:
                self._entities = pickle.load(file)
        if force or not self._failed_entity_lookups:
            with open(self._failed_entity_lookups_path, "rb") as file:
                self._failed_entity_lookups = pickle.load(file)

    def _load_annotations(self, force: bool = False) -> None:
        """ Loads serialized, preprocessed annotation data.
        force (bool): Load from disk even if already not None.
        """

        if force or not self._annotations:
            with open(self._annotations_path, "rb") as file:
                self._annotations = pickle.load(file)

    def _load_nlp(self, force: bool = False) -> None:
        """ Loads serialized pipeline.
        force (bool): Load from disk even if already not None.
        """

        if force or not self._nlp:
            self._nlp = spacy.load(self._nlp_dir)



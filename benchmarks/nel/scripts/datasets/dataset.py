""" Dataset class. """
import abc
import importlib
import inspect
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Set, List, Optional, TypeVar, Type, Dict

import numpy
import spacy
import tqdm
import yaml
from spacy import Language
from spacy.kb import KnowledgeBase
from spacy.tokens import Doc, DocBin
from spacy.training import Example

from . import evaluation
from .utils import ENTITIES_TYPE, ANNOTATIONS_TYPE

DatasetType = TypeVar('DatasetType', bound='Dataset')


class Dataset(abc.ABC):
    """ Base class for all datasets used in this benchmark. """

    KB_VECTOR_LENGTH = 300

    def __init__(self):
        """ Initializes new Dataset.
        """

        self._paths = self.assemble_paths(self.dataset_id)

        with open(self._paths["root"] / "configs" / "datasets.yml", "r") as stream:
            self._options = yaml.safe_load(stream)[self.dataset_id]

        self._entities: Optional[ENTITIES_TYPE] = None
        self._failed_entity_lookups: Optional[Set[str]] = None
        self._annotations: Optional[ANNOTATIONS_TYPE] = None
        self._kb: Optional[KnowledgeBase] = None
        self._nlp_temp: Optional[Language] = None
        self._nlp_best: Optional[Language] = None
        self._annotated_docs: Optional[List[Doc]] = None

    @staticmethod
    def assemble_paths(dataset_id: str) -> Dict[str, Path]:
        """ Assemble paths w.r.t. dataset ID.
        dataset_id (str): Dataset ID.
        RETURNS (Dict[str, Path]): Dictionary with internal resource name to path.
        """

        root_path = Path(os.path.abspath(__file__)).parent.parent.parent
        assets_path = root_path / "assets" / dataset_id

        return {
            "root": root_path,
            "assets": assets_path,
            "nlp_temp": root_path / "temp" / f"{dataset_id}.nlp",
            "nlp_best": root_path / "training" / dataset_id / "model-best",
            "kb": root_path / "temp" / f"{dataset_id}.kb",
            "corpora": root_path / "corpora" / dataset_id,
            "entities": assets_path / "entities.pkl",
            "failed_entity_lookups": assets_path / "entities_failed_lookups.pkl",
            "annotations": assets_path / "annotations.pkl"
        }

    @property
    def dataset_id(self) -> str:
        """ Returns dataset ID.
        """
        raise NotImplementedError

    def create_knowledge_base(self, model_name: str, **kwargs) -> None:
        """ Creates and serializes knowledge base.
        vectors_model (str): Name of model with word vectors to use.
        """

        self._nlp_temp = spacy.load(model_name, exclude=["tagger", "lemmatizer"])
        self._entities, self._failed_entity_lookups, self._annotations = self._parse_external_corpus(**kwargs)

        print(f"Constructing knowledge base with {len(self._entities)} entries")
        self._kb = KnowledgeBase(vocab=self._nlp_temp.vocab, entity_vector_length=Dataset.KB_VECTOR_LENGTH)
        self._kb.initialize_entities(len(self._entities))
        self._kb.initialize_vectors(len(self._entities))
        entity_list: List[str] = []
        freq_list: List[int] = []
        vector_list: List[numpy.ndarray] = []
        for qid, info in self._entities.items():
            entity_list.append(qid)
            freq_list.append(info["frequency"])
            vector_list.append(self._nlp_temp(info["description"]).vector)
        self._kb.set_entities(entity_list=entity_list, vector_list=vector_list, freq_list=freq_list)
        for qid, info in self._entities.items():
            for name in info["names"]:
                self._kb.add_alias(alias=name.replace("_", " "), entities=[qid], probabilities=[1])

        # Serialize knowledge base & entity information.
        for to_serialize in (
            (self._paths["entities"], self._entities),
            (self._paths["failed_entity_lookups"], self._failed_entity_lookups),
            (self._paths["annotations"], self._annotations)
        ):
            with open(to_serialize[0], 'wb') as fp:
                pickle.dump(to_serialize[1], fp)
        self._kb.to_disk(self._paths["kb"])
        if not os.path.exists(self._paths["nlp_temp"]):
            os.mkdir(self._paths["nlp_temp"])
        self._nlp_temp.to_disk(self._paths["nlp_temp"])

    def compile_corpora(self) -> None:
        """ Creates train/dev/test corpora for Reddit entity linking dataset.
        """

        self._load_resource("entities")
        self._load_resource("failed_entity_lookups")
        self._load_resource("annotations")
        self._load_resource("nlp_temp")

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
            if not self._paths["corpora"].exists():
                self._paths["corpora"].mkdir()
            corpus.to_disk(self._paths["corpora"] / f"{key}.spacy")

    def _load_resource(self, key: str, force: bool = False) -> None:
        """ Loads serialized resource.
        key (str): Resource key. Must be in self._paths.
        force (bool): Load from disk even if already not None.
        """

        path = self._paths[key]

        if key == "nlp_temp" and (force or not self._nlp_temp):
            self._nlp_temp = spacy.load(path)
        elif key == "nlp_best" and (force or not self._nlp_best):
            self._nlp_best = spacy.load(path)
        elif key == "kb" and (force or not self._kb):
            self._load_resource("nlp_temp")
            self._kb = KnowledgeBase(vocab=self._nlp_temp.vocab, entity_vector_length=Dataset.KB_VECTOR_LENGTH)
            self._kb.from_disk(path)
        elif key == "annotations" and (force or not self._annotations):
            with open(path, "rb") as file:
                self._annotations = pickle.load(file)
        elif key == "entities" and (force or not self._entities):
            with open(path, "rb") as file:
                self._entities = pickle.load(file)
        elif key == "failed_entity_lookups" and (force or not self._failed_entity_lookups):
            with open(self._paths["failed_entity_lookups"], "rb") as file:
                self._failed_entity_lookups = pickle.load(file)

    def evaluate(
        self,
        candidate_generation: bool = True,
        baseline: bool = True,
        context: bool = True,
        n_items: Optional[int] = None
    ) -> None:
        """ Evaluates trained pipeline on test set.
        baseline (bool): Whether to include baseline results in evaluation.
        context (bool): Whether to include the local context in the model.
        n_items (Optional[int]): How many items to consider in evaluation. If None, all items in test set are used.
        """

        # Load resources.
        self._load_resource("nlp_best")
        self._load_resource("kb")
        test_set_path = self._paths["corpora"] / "test.spacy"
        with open(test_set_path, "rb"):
            test_set = [
                Example(self._nlp_best(doc.text), doc)
                for doc in DocBin().from_disk(test_set_path).get_docs(self._nlp_best.vocab)
            ]
        self._nlp_best.config["incl_prior"] = False

        # Evaluation loop.
        label_counts = dict()
        cand_gen_label_counts = defaultdict(int)
        baseline_results = evaluation.DisambiguationBaselineResults()
        context_results = evaluation.EvaluationResults("Context only")
        combo_results = evaluation.EvaluationResults("Context and Prior")
        candidate_results = evaluation.EvaluationResults("Candidate gen.")

        for example in tqdm.tqdm(test_set, total=n_items, leave=False, desc='Processing test set'):
            if len(example) > 0:
                correct_ents = {
                    evaluation.offset(ent.start_char, ent.end_char): ent.kb_id_ for ent in example.reference.ents
                }
                ent_labels = {(ent.start_char, ent.end_char): ent.label_ for ent in example.predicted.ents}

                # Update candidate generation stats.
                if candidate_generation:
                    for ent in example.reference.ents:
                        # For the candidate generation evaluation also mis-aligned entities are considered.
                        label = ent_labels.get((ent.start_char, ent.end_char), "NIL")
                        cand_gen_label_counts[label] += 1
                        # print(ent.text, ent.kb_id_, label, "\t", {(cand.entity_, cand.alias_) for cand in self._kb.get_alias_candidates(ent.text)})
                        candidate_results.update_metrics(
                            label, ent.kb_id_, {cand.entity_ for cand in self._kb.get_alias_candidates(ent.text)}
                        )

                # Update entity disambiguation stats.
                if baseline:
                    evaluation.add_disambiguation_baseline(
                        baseline_results, label_counts, example.predicted, correct_ents, self._kb
                    )

                if context:
                    # Using only context.
                    self._nlp_best.config["incl_context"] = True
                    self._nlp_best.config["incl_prior"] = False
                    evaluation.add_disambiguation_eval_result(
                        context_results, example.predicted, correct_ents, self._nlp_best
                    )

                    # measuring combined accuracy (prior + context)
                    self._nlp_best.config["incl_context"] = True
                    self._nlp_best.config["incl_prior"] = True
                    evaluation.add_disambiguation_eval_result(
                        combo_results, example.predicted, correct_ents, self._nlp_best
                    )

        # Print result table.
        eval_results: List[evaluation.EvaluationResults] = []
        if candidate_generation:
            eval_results.append(candidate_results)
        if baseline:
            eval_results.extend([baseline_results.random, baseline_results.prior, baseline_results.oracle])
        if context:
            eval_results.extend([context_results, combo_results])
        print(dict(cand_gen_label_counts))
        evaluation.EvaluationResults.report(tuple(eval_results))

        self._nlp_best.config["incl_context"] = False
        self._nlp_best.config["incl_prior"] = False

    @classmethod
    def generate_dataset_from_id(cls: Type[DatasetType], dataset_id: str, **kwargs) -> DatasetType:
        """ Generates dataset instance from ID.
        dataset_id (str): Dataset ID.
        RETURNS (DatasetType): Instance of dataset with type determined by dataset ID.
        """

        # Assuming dataset class is in same package and name is identical to dataset ID.
        module_name = f'{__name__.split(".")[0]}.{dataset_id}'
        classes = [
            m for m in inspect.getmembers(importlib.import_module(module_name), inspect.isclass)
            if m[1].__module__ == module_name and issubclass(m[1], Dataset)
        ]
        assert len(classes) == 1, f"Module {module_name} should contain exactly one Dataset class definition."

        return classes[0][1](**kwargs)

    def clean_assets(self) -> None:
        """ Cleans assets, i.e. removes/changes errors in the external datasets that cannot easily be cleaned
        automatically.
        """
        raise NotImplementedError

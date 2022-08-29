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
from spacy.pipeline.legacy import EntityLinker_v1

from spacy.tokens import Doc, DocBin, Span
from spacy.training import Example
from schemas import Annotation, Entity
from wiki import wiki_dump_api
from . import evaluation
from utils import get_logger

logger = get_logger(__name__)
DatasetType = TypeVar("DatasetType", bound="Dataset")


class Dataset(abc.ABC):
    """Base class for all datasets used in this benchmark."""

    def __init__(self):
        """Initializes new Dataset."""

        self._paths = self.assemble_paths(self.name)

        with open(self._paths["root"] / "configs" / "datasets.yml", "r") as stream:
            self._options = yaml.safe_load(stream)[self.name]

        self._entities: Optional[Dict[str, Entity]] = None
        self._failed_entity_lookups: Optional[Set[str]] = None
        self._annotations: Optional[Dict[str, List[Annotation]]] = None
        self._kb: Optional[KnowledgeBase] = None
        self._nlp_base: Optional[Language] = None
        self._nlp_best: Optional[Language] = None
        self._annotated_docs: Optional[List[Doc]] = None

    @staticmethod
    def assemble_paths(dataset_name: str) -> Dict[str, Path]:
        """Assemble paths w.r.t. dataset ID.
        dataset_name (str): Dataset name.
        RETURNS (Dict[str, Path]): Dictionary with internal resource name to path.
        """

        root_path = Path(os.path.abspath(__file__)).parent.parent.parent
        assets_path = root_path / "assets" / dataset_name

        return {
            "root": root_path,
            "assets": assets_path,
            "nlp_base": root_path / "temp" / dataset_name / "nlp",
            "nlp_best": root_path / "training" / dataset_name / "model-best",
            "kb": root_path / "temp" / dataset_name / "kb",
            "corpora": root_path / "corpora" / dataset_name,
            "entities": assets_path / "entities.pkl",
            "failed_entity_lookups": assets_path / "entities_failed_lookups.pkl",
            "annotations": assets_path / "annotations.pkl",
        }

    @property
    def name(self) -> str:
        """Returns dataset name."""
        raise NotImplementedError

    def create_knowledge_base(self, model_name: str, **kwargs) -> None:
        """Creates and serializes knowledge base.
        vectors_model (str): Name of model with word vectors to use.
        """

        self._nlp_base = spacy.load(
            model_name, exclude=["tagger", "lemmatizer", "attribute_ruler"]
        )
        logger.info("Parsing external corpus")
        (
            self._entities,
            self._failed_entity_lookups,
            self._annotations,
        ) = self._parse_external_corpus(**kwargs)

        logger.info(
            f"Constructing knowledge base with {len(self._entities)} entries and "
            f"{len(self._failed_entity_lookups)} failed lookups."
        )
        self._kb = KnowledgeBase(
            vocab=self._nlp_base.vocab,
            entity_vector_length=self._nlp_base.vocab.vectors_length,
        )
        entity_list: List[str] = []
        count_list: List[int] = []
        vector_list: List[numpy.ndarray] = []  # type: ignore
        for qid, info in self._entities.items():
            entity_list.append(qid)
            count_list.append(info.count)
            desc_vector = self._nlp_base(
                info.description if info.description else info.name
            ).vector
            vector_list.append(
                desc_vector
                if isinstance(desc_vector, numpy.ndarray)
                else desc_vector.get()
            )
        self._kb.set_entities(
            entity_list=entity_list, vector_list=vector_list, freq_list=count_list
        )

        # Add aliases with normalized priors to KB.
        alias_entity_prior_probs = wiki_dump_api.load_alias_entity_prior_probabilities(
            set(self._entities.keys())
        )
        for alias, entity_prior_probs in alias_entity_prior_probs.items():
            self._kb.add_alias(
                alias=alias,
                entities=[epp[0] for epp in entity_prior_probs],
                probabilities=[epp[1] for epp in entity_prior_probs],
            )
        # Add pseudo aliases for easier lookup with new candidate generators.
        for entity_id in entity_list:
            self._kb.add_alias(
                alias="_" + entity_id + "_", entities=[entity_id], probabilities=[1]
            )

        # Serialize knowledge base & entity information.
        for to_serialize in (
            (self._paths["entities"], self._entities),
            (self._paths["failed_entity_lookups"], self._failed_entity_lookups),
            (self._paths["annotations"], self._annotations),
        ):
            with open(to_serialize[0], "wb") as fp:
                pickle.dump(to_serialize[1], fp)
        self._kb.to_disk(self._paths["kb"])
        if not os.path.exists(self._paths["nlp_base"]):
            os.mkdir(self._paths["nlp_base"])
        self._nlp_base.to_disk(self._paths["nlp_base"])
        logger.info("Successfully constructed knowledge base.")

    def compile_corpora(self, filter_terms: Optional[Set[str]] = None) -> None:
        """Creates train/dev/test corpora for dataset.
        filter_terms (Optional[Set[str]]): Set of filter terms. Only documents containing at least one of the specified
            terms will be included in corpora. If None, all documents are included.
        """

        self._load_resource("entities")
        self._load_resource("failed_entity_lookups")
        self._load_resource("annotations")
        self._load_resource("nlp_base")

        Doc.set_extension("overlapping_annotations", default=None)
        self._annotated_docs = self._create_annotated_docs(filter_terms)
        self._serialize_corpora()

    def _create_annotated_docs(self, filter_terms: Optional[Set[str]] = None) -> List[Doc]:
        """Creates docs annotated with entities.
        filter_terms (Optional[Set[str]]): Set of filter terms. Only documents containing at least one of the specified
            terms will be included in corpora. If None, all documents are included.
        RETURN (List[Doc]): List of docs reflecting all entity annotations.
        """
        raise NotImplementedError

    def _parse_external_corpus(
        self, **kwargs
    ) -> Tuple[Dict[str, Entity], Set[str], Dict[str, List[Annotation]]]:
        """Parses external corpus. Loads data on entities and mentions.
        Populates self._entities, self._failed_entity_lookups, self._annotations.
        RETURNS (Tuple[Dict[str, Entity], Set[str], Dict[str, List[Annotation]]]): entities, titles of failed entity
            lookups, annotations.
        """
        raise NotImplementedError

    def _serialize_corpora(self) -> None:
        """Serializes corpora."""

        assert (
            self._options["frac_train"]
            + self._options["frac_dev"]
            + self._options["frac_test"]
            == 1
        )

        indices = {
            dataset: idx
            for dataset, idx in zip(
                ("train", "dev", "test"),
                numpy.split(
                    numpy.asarray(range(len(self._annotated_docs))),
                    [
                        int(self._options["frac_train"] * len(self._annotated_docs)),
                        int(
                            (self._options["frac_train"] + self._options["frac_dev"])
                            * len(self._annotated_docs)
                        ),
                    ],
                ),
            )
        }

        for key, value in indices.items():
            corpus = DocBin(store_user_data=True)
            for idx in value:
                corpus.add(self._annotated_docs[idx])
            if not self._paths["corpora"].exists():
                self._paths["corpora"].mkdir()
            corpus.to_disk(self._paths["corpora"] / f"{key}.spacy")
        logger.info(f"Completed serializing corpora at {self._paths['corpora']}.")

    def _load_resource(self, key: str, force: bool = False) -> None:
        """Loads serialized resource.
        key (str): Resource key. Must be in self._paths.
        force (bool): Load from disk even if already not None.
        """

        path = self._paths[key]

        if key == "nlp_base" and (force or not self._nlp_base):
            self._nlp_base = spacy.load(path)
        elif key == "nlp_best" and (force or not self._nlp_best):
            self._nlp_best = spacy.load(path)
        elif key == "kb" and (force or not self._kb):
            self._load_resource("nlp_base")
            self._kb = KnowledgeBase(
                vocab=self._nlp_base.vocab,
                entity_vector_length=self._nlp_base.vocab.vectors_length,
            )
            self._kb.from_disk(path)
        elif key == "annotations" and (force or not self._annotations):
            with open(path, "rb") as file:
                self._annotations = pickle.load(file)
        elif key == "entities" and (force or not self._entities):
            with open(path, "rb") as file:
                self._entities = pickle.load(file)
        elif key == "failed_entity_lookups" and (
            force or not self._failed_entity_lookups
        ):
            with open(self._paths["failed_entity_lookups"], "rb") as file:
                self._failed_entity_lookups = pickle.load(file)

    def evaluate(
        self,
        candidate_generation: bool = True,
        baseline: bool = True,
        context: bool = True,
        spacyfishing: bool = True,
        n_items: Optional[int] = None,
    ) -> None:
        """Evaluates trained pipeline on test set.
        baseline (bool): Whether to include baseline results in evaluation.
        context (bool): Whether to include the local context in the model.
        spacyfishing (bool): Whether to include evaluation with spacyfishing.
        n_items (Optional[int]): How many items to consider in evaluation. If None, all items in test set are used.
        """

        # Load resources.
        self._load_resource("nlp_best")
        self._load_resource("nlp_base")
        self._load_resource("kb")

        # Compile test set.
        test_set_path = self._paths["corpora"] / "test.spacy"
        with open(test_set_path, "rb"):
            test_set: List[Example] = []
            for doc in DocBin().from_disk(test_set_path).get_docs(self._nlp_best.vocab):
                predicted_doc = self._nlp_best(doc.text)
                ents: List[Span] = []
                for ent in predicted_doc.ents:
                    # spaCy includes leading articles in entities, our benchmark datasets don't. Hence we drop all
                    # leading "the " and adjust the entity positions accordingly.
                    ents.append(
                        doc.char_span(ent.start_char + 4, ent.end_char, label=ent.label, kb_id=ent.kb_id)
                        if ent.text.lower().startswith("the ") else ent
                    )
                predicted_doc.ents = ents
                test_set.append(Example(predicted_doc, doc))

        self._nlp_best.config["incl_prior"] = False
        if spacyfishing:
            self._nlp_base.add_pipe("entityfishing", last=True)

        # Evaluation loop.
        label_counts = dict()
        cand_gen_label_counts = defaultdict(int)
        baseline_results = evaluation.DisambiguationBaselineResults()
        context_results = evaluation.EvaluationResults("Context only")
        combo_results = evaluation.EvaluationResults("Context and Prior")
        spacyfishing_results = evaluation.EvaluationResults("spacyfishing")
        candidate_results = evaluation.EvaluationResults("Candidate gen.")

        for example in tqdm.tqdm(
            test_set, total=n_items, leave=False, desc="Processing test set"
        ):
            example: Example
            if len(example) > 0:
                entity_linker: EntityLinker_v1 = self._nlp_best.components[  # type: ignore
                    self._nlp_best.component_names.index("entity_linker")
                ][1]
                ent_gold_ids = {
                    evaluation.offset(ent.start_char, ent.end_char): ent.kb_id_
                    for ent in example.reference.ents
                }
                if len(ent_gold_ids) == 0:
                    continue
                ent_pred_labels = {
                    (ent.start_char, ent.end_char): ent.label_
                    for ent in example.predicted.ents
                }
                ent_cand_ids = {
                    (ent.start_char, ent.end_char): {
                        cand.entity_
                        for cand in entity_linker.get_candidates(self._kb, ent)
                    }
                    for ent in example.reference.ents
                }

                # Update candidate generation stats.
                if candidate_generation:
                    for ent in example.reference.ents:
                        ent_offset = (ent.start_char, ent.end_char)
                        # For the candidate generation evaluation also mis-aligned entities are considered.
                        label = ent_pred_labels.get(ent_offset, "NIL")
                        cand_gen_label_counts[label] += 1
                        candidate_results.update_metrics(
                            label, ent.kb_id_, ent_cand_ids.get(ent_offset, {})
                        )

                # Update entity disambiguation stats.
                if baseline:
                    evaluation.add_disambiguation_baseline(
                        baseline_results,
                        label_counts,
                        example.predicted,
                        ent_gold_ids,
                        self._kb,
                        ent_cand_ids,
                    )

                if context:
                    # Using only context.
                    self._nlp_best.config["incl_context"] = True
                    self._nlp_best.config["incl_prior"] = False
                    evaluation.add_disambiguation_eval_result(
                        context_results,
                        example.predicted,
                        ent_gold_ids,
                        self._nlp_best,
                        ent_cand_ids,
                    )

                    # measuring combined accuracy (prior + context)
                    self._nlp_best.config["incl_context"] = True
                    self._nlp_best.config["incl_prior"] = True
                    evaluation.add_disambiguation_eval_result(
                        combo_results,
                        example.predicted,
                        ent_gold_ids,
                        self._nlp_best,
                        ent_cand_ids,
                    )

                if spacyfishing:
                    try:
                        doc = self._nlp_base(example.reference.text)
                    except TypeError:
                        doc = None
                    evaluation.add_disambiguation_spacyfishing_eval_result(
                        spacyfishing_results,
                        doc,
                        ent_gold_ids,
                    )

        # Print result table.
        eval_results: List[evaluation.EvaluationResults] = []
        if candidate_generation:
            eval_results.append(candidate_results)
        if baseline:
            eval_results.extend(
                [
                    baseline_results.random,
                    baseline_results.prior,
                    baseline_results.oracle,
                ]
            )
        if context:
            eval_results.extend([context_results, combo_results])
        if spacyfishing:
            eval_results.append(spacyfishing_results)

        logger.info(dict(cand_gen_label_counts))
        evaluation.EvaluationResults.report(tuple(eval_results))

        self._nlp_best.config["incl_context"] = False
        self._nlp_best.config["incl_prior"] = False

    @classmethod
    def generate_dataset_from_id(
        cls: Type[DatasetType], dataset_name: str, **kwargs
    ) -> DatasetType:
        """Generates dataset instance from ID.
        dataset_name (str): Dataset name.
        RETURNS (DatasetType): Instance of dataset with type determined by dataset ID.
        """

        # Assuming dataset class is in same package and name is identical to dataset ID.
        module_name = f'{__name__.split(".")[0]}.{dataset_name}'
        classes = [
            m
            for m in inspect.getmembers(
                importlib.import_module(module_name), inspect.isclass
            )
            if m[1].__module__ == module_name and issubclass(m[1], Dataset)
        ]
        assert (
            len(classes) == 1
        ), f"Module {module_name} should contain exactly one Dataset class definition."

        return classes[0][1](**kwargs)

    def clean_assets(self) -> None:
        """Cleans assets, i.e. removes/changes errors in the external datasets that cannot easily be cleaned
        automatically.
        """
        raise NotImplementedError

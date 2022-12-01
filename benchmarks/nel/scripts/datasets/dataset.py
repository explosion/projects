""" Dataset class. """
import abc
import csv
import datetime
import importlib
import inspect
import operator
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Set, List, Optional, TypeVar, Type, Dict, Union

import numpy
import prettytable
import spacy
import tqdm
import yaml
from spacy import Language
from spacy.kb import KnowledgeBase
from spacy.pipeline.legacy import EntityLinker_v1
from spacy.tokens import Doc, DocBin
from spacy.training import Example
from spacy.pipeline import EntityLinker

from wikid import schemas
from wikid.scripts.kb import WikiKB
from . import evaluation
from utils import get_logger

logger = get_logger(__name__)
DatasetType = TypeVar("DatasetType", bound="Dataset")


class Dataset(abc.ABC):
    """Base class for all datasets used in this benchmark."""

    def __init__(self, run_name: str, language: str):
        """Initializes new Dataset.
        run_name (str): Run name.
        language (str): Language.
        """

        self._run_name = run_name
        self._language = language
        self._paths = self.assemble_paths(self.name, self._run_name, self._language)

        with open(self._paths["root"] / "configs" / "datasets.yml", "r") as stream:
            self._options = yaml.safe_load(stream)[self.name]

        self._annotations: Optional[Dict[str, List[schemas.Annotation]]] = None
        self._kb: Optional[KnowledgeBase] = None
        self._nlp_best: Optional[Language] = None
        self._annotated_docs: Optional[List[Doc]] = None

    @staticmethod
    def assemble_paths(dataset_name: str, run_name: str, language: str) -> Dict[str, Path]:
        """Assemble paths w.r.t. dataset ID.
        dataset_name (str): Dataset name.
        run_name (str): Run name.
        language (str): Language.
        RETURNS (Dict[str, Path]): Dictionary with internal resource name to path.
        """

        root_path = Path(os.path.abspath(__file__)).parent.parent.parent
        wikid_output_path = root_path / "wikid" / "output"
        assets_path = root_path / "assets" / dataset_name

        return {
            "root": root_path,
            "evaluation": root_path / "configs" / "evaluation.yml",
            "assets": assets_path,
            "kb": wikid_output_path / language / "kb",
            "annotations": assets_path / "annotations.pkl",
            "nlp_base": root_path / "training" / "base-nlp" / language,
            "nlp_best": root_path / "training" / dataset_name / run_name / "model-best",
            "corpora": root_path / "corpora" / dataset_name
        }

    @property
    def name(self) -> str:
        """Returns dataset name."""
        raise NotImplementedError

    def compile_corpora(self, model: str, filter_terms: Optional[Set[str]] = None) -> None:
        """Creates train/dev/test corpora for dataset.
        model (str): Name or path of model with tokenizer, tok2vec, parser, tagger, parser.
        filter_terms (Optional[Set[str]]): Set of filter terms. Only documents containing at least one of the specified
            terms will be included in corpora. If None, all documents are included.
        """
        with open(self._paths["annotations"], "rb") as file:
            self._annotations = pickle.load(file)
        Doc.set_extension("overlapping_annotations", default=None)
        nlp_components = ["tok2vec", "tagger", "attribute_ruler"]
        nlp = spacy.load(model, enable=nlp_components)
        nlp.add_pipe("sentencizer")

        # Incorporate annotations from corpus into documents. Only keep docs with entities (relevant mostly when working
        # with filtered data).
        self._annotated_docs = [doc for doc in self._create_annotated_docs(nlp, filter_terms) if len(doc.ents)]

        # Serialize pipeline and corpora.
        self._paths["nlp_base"].parent.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(
            self._paths["nlp_base"],
            exclude=[comp for comp in nlp.component_names if comp not in [*nlp_components, "sentencizer"]]
        )
        self._serialize_corpora()

    def _create_annotated_docs(self, nlp: Language, filter_terms: Optional[Set[str]] = None) -> List[Doc]:
        """Creates docs annotated with entities. This should set documents `ents` attribute.
        nlp (Language): Model with tokenizer, tok2vec and parser.
        filter_terms (Optional[Set[str]]): Set of filter terms. Only documents containing at least one of the specified
            terms will be included in corpora. If None, all documents are included.
        RETURN (List[Doc]): List of docs reflecting all entity annotations.
        """
        raise NotImplementedError

    def extract_annotations(self, **kwargs) -> None:
        """Parses corpus and extracts annotations. Loads data on entities and mentions.
        Populates self._annotations.
        """
        logger.info("Extracting annotations from corpus")
        self._annotations = self._extract_annotations_from_corpus(**kwargs)
        with open(self._paths["annotations"], "wb") as fp:
            pickle.dump(self._annotations, fp)

        logger.info("Successfully extracted annotations from corpus.")

    def _extract_annotations_from_corpus(
            self, **kwargs
    ) -> Tuple[Dict[str, schemas.Entity], Set[str], Dict[str, List[schemas.Annotation]]]:
        """Parses corpus. Loads data on entities and mentions.
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

        for key, idx in indices.items():
            corpus = DocBin(store_user_data=True, docs=[self._annotated_docs[i] for i in idx])
            self._paths["corpora"].mkdir(parents=True, exist_ok=True)
            corpus.to_disk(self._paths["corpora"] / f"{key}.spacy")
        logger.info(f"Completed serializing corpora at {self._paths['corpora']}.")

    def evaluate(self, gpu_id: Optional[int] = None) -> None:
        """Evaluates trained pipeline on test set.
        run_name (str): Run name.
        gpu_id (Optional[int]): ID of GPU to utilize.
        """
        if gpu_id is not None:
            spacy.require_gpu(gpu_id)

        nlp_base = spacy.load(self._paths["nlp_base"])
        self._nlp_best = spacy.load(self._paths["nlp_best"])
        self._kb = WikiKB.generate_from_disk(self._paths["kb"])

        with open(self._paths["evaluation"], "r") as config_file:
            eval_config = yaml.safe_load(config_file)
        if eval_config["external"]["spacyfishing"]:
            nlp_base.add_pipe("entityfishing", last=True)

        # Apply config overrides, if defined.
        if "config_overrides" in eval_config and eval_config["config_overrides"]:
            for setting, value in eval_config["config_overrides"].items():
                self._nlp_best.config[setting] = value

        # Infer test set.
        test_set_path = self._paths["corpora"] / "test.spacy"
        docs = list(DocBin().from_disk(test_set_path).get_docs(self._nlp_best.vocab))

        test_set = [
            Example(predicted_doc, doc)
            for predicted_doc, doc in zip(
                [
                    doc for doc in tqdm.tqdm(
                        self._nlp_best.pipe(texts=docs, n_process=1 if gpu_id else -1, batch_size=500),
                        desc="Inferring entities for test set",
                        total=len(docs)
                    )
                ],
                docs
            )
        ]

        # Evaluation loop.
        label_counts = dict()
        cand_gen_label_counts = defaultdict(int)
        baseline_results = evaluation.DisambiguationBaselineResults()
        spacyfishing_results = evaluation.EvaluationResults("spacyfishing")
        trained_results = evaluation.EvaluationResults("Trained")
        candidate_results = evaluation.EvaluationResults("Candidate gen.")

        for example in tqdm.tqdm(test_set, total=len(test_set), leave=True, desc="Evaluating test set"):
            example: Example
            if len(example) > 0:
                entity_linker: EntityLinker = self._nlp_best.get_pipe("entity_linker")  # type: ignore
                ent_gold_ids = {
                    evaluation.offset(ent.start_char, ent.end_char): ent.kb_id_ for ent in example.reference.ents
                }
                if len(ent_gold_ids) == 0:
                    continue
                ent_pred_labels = {(ent.start_char, ent.end_char): ent.label_ for ent in example.predicted.ents}
                ent_cands_by_offset = {
                    (ent.start_char, ent.end_char): {cand.entity_: cand for cand in ent_cands}
                    for ent, ent_cands in zip(
                        example.reference.ents,
                        next(entity_linker.get_candidates_all(self._kb, (ents for ents in [example.reference.ents])))
                    )
                }

                # Update candidate generation stats.
                if eval_config["candidate_generation"]:
                    for ent in example.reference.ents:
                        ent_offset = (ent.start_char, ent.end_char)
                        # For the candidate generation evaluation also mis-aligned entities are considered.
                        label = ent_pred_labels.get(ent_offset, "NIL")
                        cand_gen_label_counts[label] += 1
                        candidate_results.update_metrics(
                            label, ent.kb_id_, set(ent_cands_by_offset.get(ent_offset, {}).keys())
                        )

                # Update entity disambiguation stats for baselines.
                evaluation.add_disambiguation_baseline(
                    baseline_results,
                    label_counts,
                    example.predicted,
                    ent_gold_ids,
                    ent_cands_by_offset,
                )

                # Update entity disambiguation stats for trained model.
                evaluation.add_disambiguation_eval_result(trained_results, example.predicted, ent_gold_ids, ent_cands_by_offset)

                if eval_config["external"].get("spacyfishing", False):
                    try:
                        doc = nlp_base(example.reference.text)
                    except TypeError:
                        doc = None
                    evaluation.add_disambiguation_spacyfishing_eval_result(spacyfishing_results, doc, ent_gold_ids)

        # Print result table.
        eval_results: List[evaluation.EvaluationResults] = [
            baseline_results.random,
            baseline_results.prior,
            baseline_results.oracle,
            trained_results
        ]
        if eval_config.get("candidate_generation", False):
            eval_results.append(candidate_results)
        if eval_config.get("spacyfishing", False):
            eval_results.append(spacyfishing_results)

        logger.info(dict(cand_gen_label_counts))
        evaluation.EvaluationResults.report(tuple(eval_results), run_name=self._run_name, dataset_name=self.name)

    def compare_evaluations(self, highlight_criterion: str) -> None:
        """Generate and display table for comparison of all available runs for this dataset.
        Note that this both persists and logs a table that shows the F-score/recall/precision values for each run per:
            - EL method (context and prior, context only, oracle, prior, ...)
            - candidate generation
        Hence the rows with "Candidate Gen." in the "Model" column can't be compared with the non-candidate generation
        rows.
        highlight_criterion (str): Criterion to highlight in table. One of ("F", "r", "p").
        """
        assert highlight_criterion in ("F", "r", "p"), "Criterion must be one of ('F', 'r', 'p')"

        header: Optional[List[str, ...]] = None
        rows: List[List[str, ...]] = []
        dir_path = Path(os.path.abspath(__file__)).parent.parent.parent / "evaluation" / self.name

        for path in dir_path.glob("*.csv"):
            if path.stem.startswith("comparison-"):
                continue
            with open(path, "r") as csv_file:
                csv_reader = csv.reader(csv_file)
                _header = next(csv_reader)
                if header is None:
                    header = ["Run", *_header]
                rows.extend([[path.stem, *row] for row in csv_reader])
        rows = sorted(rows, key=operator.itemgetter(0, 1))

        # Persist combined table.
        table = prettytable.PrettyTable(field_names=header)
        file_name = f"comparison-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        table.add_rows(rows)
        with open(dir_path / file_name, "w") as csv_file:
            csv_file.write(table.get_csv_string())

        # Create table for console output with formatted rows.
        table = prettytable.PrettyTable(field_names=header)
        highlight_crit_idx = header.index({"F": "F-score", "r": "Recall", "p": "Precision"}[highlight_criterion])
        max_crit_non_cand_gen = .0
        max_crit_cand_gen = .0
        for row in rows:
            if row[1] != "Candidate Gen.":
                max_crit_non_cand_gen = max(max_crit_non_cand_gen, float(row[highlight_crit_idx]))
            else:
                max_crit_cand_gen = max(max_crit_cand_gen, float(row[highlight_crit_idx]))
        for row in rows:
            if (
                row[1] != "Candidate Gen." and float(row[highlight_crit_idx]) >= max_crit_non_cand_gen or
                row[1] == "Candidate Gen." and float(row[highlight_crit_idx]) >= max_crit_cand_gen
            ):
                for i in range(len(row)):
                    row[i] = '\033[4m' + row[i] + '\033[0m'
            table.add_row(row)

        logger.info("\n" + str(table))

    @classmethod
    def generate_from_id(
        cls: Type[DatasetType], dataset_name: str, language: str, run_name: str = "", **kwargs
    ) -> DatasetType:
        """Generates dataset instance from ID.
        dataset_name (str): Dataset name.
        run_name (str): Run name.
        language (str): Language.
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

        return classes[0][1](run_name=run_name, language=language, **kwargs)

    def clean_assets(self) -> None:
        """Cleans assets, i.e. removes/changes errors in the external datasets that cannot easily be cleaned
        automatically.
        """
        raise NotImplementedError

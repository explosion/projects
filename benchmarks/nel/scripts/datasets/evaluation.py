""" Evaluation utilities.
Adapted from https://github.com/explosion/projects/blob/master/nel-wikipedia/entity_linker_evaluation.py.
"""

import logging
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import prettytable
from spacy import Language
from spacy.kb import KnowledgeBase
from spacy.tokens import Doc
from utils import get_logger

logger = get_logger(__name__)


class Metrics(object):
    true_pos = 0
    false_pos = 0
    false_neg = 0
    n_updates = 0
    n_candidates = 0

    def update_results(self, true_entity: str, candidates: Set[str]):
        """Update metric results. Note that len(candidates) will always be 1 for NEL checks, as only one suggestion is
        picked. For candidate generation however an arbitrary number of candidates is possible.
        true_entity (str): ID of correct entity.
        candidates (Set[str]): Suggested entity ID(s).
        """
        self.n_updates += 1
        self.n_candidates += len(candidates)
        candidate_is_correct = true_entity in candidates

        # Assume that we have no labeled negatives in the data (i.e. cases where true_entity is "NIL")
        # Therefore, if candidate_is_correct then we have a true positive and never a true negative.
        self.true_pos += candidate_is_correct
        self.false_neg += not candidate_is_correct
        if candidates and candidates not in ({""}, {"NIL"}):
            # A wrong prediction (e.g. Q42 != Q3) counts both as a FP as well as a FN.
            self.false_pos += not candidate_is_correct

    def calculate_precision(self):
        if self.true_pos == 0:
            return 0.0
        else:
            return self.true_pos / (self.true_pos + self.false_pos)

    def calculate_recall(self):
        if self.true_pos == 0:
            return 0.0
        else:
            return self.true_pos / (self.true_pos + self.false_neg)

    def calculate_fscore(self):
        p = self.calculate_precision()
        r = self.calculate_recall()
        if p + r == 0:
            return 0.0
        else:
            return 2 * p * r / (p + r)


class EvaluationResults(object):
    def __init__(self, name: str):
        self.name = name
        self.metrics = Metrics()
        self.metrics_by_label = defaultdict(Metrics)

    def update_metrics(
        self, ent_label: str, true_ent_kb_id_: str, cand_kb_ids_: Set[str]
    ) -> None:
        """Update metrics over all candidate labels and per candidate labels.
        ent_label (str): Recognized entity label.
        true_ent_kb_id_ (str): True entity's external KB ID.
        cand_kb_ids_ (Set[str]): Set of candidates' external KB IDs.
        """
        self.metrics.update_results(true_ent_kb_id_, cand_kb_ids_)
        self.metrics_by_label[ent_label].update_results(true_ent_kb_id_, cand_kb_ids_)

    def _extend_report_overview_table(self, table: prettytable.PrettyTable) -> None:
        """Extend existing PrettyTable with collected metrics for report overview.
        model_name (str): Model name.
        table (prettytable.PrettyTable): PrettyTable object for evaluation results.
        """
        table.add_row(
            [
                self.name.title(),
                str(self.metrics.true_pos),
                str(self.metrics.false_pos),
                str(self.metrics.false_neg),
                f"{round(self.metrics.calculate_fscore(), 3)}",
                f"{round(self.metrics.calculate_recall(), 3)}",
                f"{round(self.metrics.calculate_precision(), 3)}",
            ]
        )

    def _extend_report_labels_table(
        self, table: prettytable.PrettyTable, labels: Tuple[str, ...]
    ) -> None:
        """Extend existing PrettyTable with collected metrics per label.
        model_name (str): Model name.
        table (prettytable.PrettyTable): PrettyTable object for evaluation results.
        labels (Tuple[str, ...]): Labels in sequence to be added to table.
        """

        for label in labels:
            table.add_row(
                [
                    label,
                    self.name.title(),
                    self.metrics_by_label[label].calculate_fscore(),
                    self.metrics_by_label[label].calculate_recall(),
                    self.metrics_by_label[label].calculate_precision(),
                ]
            )

    @staticmethod
    def report(evaluation_results: Tuple["EvaluationResults"]) -> None:
        """Reports evaluation results.
        evaluation_result (Tuple["EvaluationResults"]): Evaluation results.
        """
        labels = sorted(
            list(
                {
                    label
                    for eval_res in evaluation_results
                    for label in eval_res.metrics_by_label
                }
            )
        )
        overview_table = prettytable.PrettyTable(
            field_names=[
                "Model",
                "TP",
                "FP",
                "FN",
                "F-score",
                "Recall",
                "Precision",
            ]
        )
        label_table = prettytable.PrettyTable(
            field_names=["Label", "Model", "F-score", "Recall", "Precision"]
        )

        for eval_result in evaluation_results:
            eval_result._extend_report_overview_table(overview_table)
        for label in labels:
            for eval_result in evaluation_results:
                eval_result._extend_report_labels_table(label_table, (label,))

        logger.info("\n" + str(overview_table))
        logger.info("\n" + str(label_table))


class DisambiguationBaselineResults(object):
    def __init__(self):
        self.random = EvaluationResults("Random")
        self.prior = EvaluationResults("Prior")
        self.oracle = EvaluationResults("Oracle")

    def report_performance(self, model):
        results = getattr(self, model)
        return results.report_metrics()

    def update_baselines(
        self,
        true_entity,
        ent_label,
        random_candidate,
        prior_candidate,
        oracle_candidate,
    ):
        self.oracle.update_metrics(ent_label, true_entity, {oracle_candidate})
        self.prior.update_metrics(ent_label, true_entity, {prior_candidate})
        self.random.update_metrics(ent_label, true_entity, {random_candidate})


def add_disambiguation_eval_result(
    results: EvaluationResults,
    pred_doc: Doc,
    correct_ents: Dict[str, str],
    el_nlp: Language,
    ent_cand_ids: Dict[Tuple[int, int], Set[str]],
) -> None:
    """
    Evaluate the ent.kb_id_ annotations against the gold standard.
    Only evaluate entities that overlap between gold and NER, to isolate the performance of the NEL.
    results (EvaluationResults): Container for evaluation results.
    pred_doc (Doc): Predicted Doc object to evaluate.
    correct_ents (Dict[str, str]): Dictionary with offsets to entity QIDs.
    el_nlp (Language): Pipeline.
    ent_cand_ids (Dict[str, Set[str]]): Candidates per recognized entities' offsets.
    """
    try:
        for ent in pred_doc.ents:
            idx = (ent.start_char, ent.end_char)
            gold_entity = correct_ents.get(offset(*idx), None)
            # the gold annotations are not complete so we can't evaluate missing annotations as 'wrong'
            if gold_entity in ent_cand_ids.get(idx, {}):
                results.update_metrics(ent.label_, gold_entity, {ent.kb_id_})

    except Exception as e:
        logging.error("Error assessing accuracy " + str(e))


def add_disambiguation_spacyfishing_eval_result(
    results: EvaluationResults, pred_doc: Optional[Doc], correct_ents: Dict[str, str]
) -> None:
    """Measure NEL performance with spacyfishing.
    results (EvaluationResults): Eval. results object.
    pred_doc (Optional[Doc]): Document after running it through spacyfishing pipeline. Might be None in case of pipeline
        error.
    correct_ents (Dict[str, str]): Mapping from stringified offsets to correct entity IDs.
    """

    try:
        if pred_doc is None:
            results.update_metrics("NIL", "", {"NIL"})
            return

        for ent in pred_doc.ents:
            gold_entity = correct_ents.get(offset(ent.start_char, ent.end_char), None)
            if gold_entity is not None:
                results.update_metrics(ent.label_, gold_entity, {ent._.kb_qid})

    except Exception as e:
        logging.error("Error assessing accuracy " + str(e))


def add_disambiguation_baseline(
    baseline_results: DisambiguationBaselineResults,
    counts: Dict[str, int],
    pred_doc: Doc,
    correct_ents: Dict[str, str],
    kb: KnowledgeBase,
    ent_cand_ids: Dict[str, Set[str]],
) -> None:
    """
    Measure 3 performance baselines: random selection, prior probabilities, and 'oracle' prediction for upper bound.
    Only evaluate entities that overlap between gold and NER, to isolate the performance of the NEL.
    baseline_results (BaselineResults):
    counts (Dict[str, int]): Counts per label.
    pred_doc (Doc): Predicted Doc object to evaluate.
    correct_ents (Dict[str, str]): Offsets in the shape of {f"{start_char}_{end_char}": QID}.
    kb (KnowledgeBase): Knowledge base.
        ent_cand_ids (Dict[str, Set[str]]): Candidates per recognized entities' offsets.
    """
    for ent in pred_doc.ents:
        ent_label = ent.label_
        idx = (ent.start_char, ent.end_char)
        gold_entity = correct_ents.get(offset(*idx), None)

        # The gold annotations are not necessarily complete so we can't evaluate missing annotations as wrong.
        if gold_entity in ent_cand_ids.get(idx, {}):
            candidates = kb.get_alias_candidates(ent.text)
            oracle_candidate = ""
            prior_candidate = ""
            random_candidate = ""
            if candidates:
                scores: List[float] = []

                for c in candidates:
                    scores.append(c.prior_prob)
                    if c.entity_ == gold_entity:
                        oracle_candidate = c.entity_

                prior_candidate = candidates[scores.index(max(scores))].entity_
                random_candidate = random.choice(candidates).entity_

            current_count = counts.get(ent_label, 0)
            counts[ent_label] = current_count + 1

            baseline_results.update_baselines(
                gold_entity,
                ent_label,
                random_candidate,
                prior_candidate,
                oracle_candidate,
            )


def offset(start: int, end: int):
    return "{}_{}".format(start, end)

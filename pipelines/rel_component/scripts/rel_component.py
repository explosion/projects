from typing import Tuple, List, Iterable, Optional, Dict, Callable
from thinc.types import Floats2d
import numpy
from spacy.training.example import Example
from thinc.api import Model, Optimizer, Config
from spacy.tokens.doc import Doc
from spacy.pipeline.pipe import Pipe
from spacy.vocab import Vocab
from spacy import Language
from thinc.model import set_dropout_rate


Doc.set_extension("rel", default={})


@Language.factory(
    "relation_extractor",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
    assigns=["doc._.rel"],
)
def make_relation_extractor(
    nlp: Language,
    name: str,
    model: Model,
    *,
    labels: List[str] = [],
):
    """Construct a RelationExtractor component."""
    return RelationExtractor(
        nlp.vocab,
        model,
        name,
        labels=labels,
    )


class RelationExtractor(Pipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "rel",
        *,
        labels: List[str] = [],
    ) -> None:
        """Initialize a relation extractor."""
        self.vocab = vocab
        self.model = model
        self.name = name
        cfg = {
            "labels": labels,
        }
        self.cfg = dict(cfg)

    @property
    def labels(self) -> Tuple[str]:
        """Returns the labels currently added to the component."""
        return tuple(self.cfg["labels"])

    def add_label(self, label: str) -> int:
        """Add a new label to the pipe."""
        if not isinstance(label, str):
            raise ValueError("Only strings can be added as labels to the RelationExtractor")
        if label in self.labels:
            return 0
        self.cfg["labels"] = list(self.labels) + [label]
        return 1

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipe to a Doc."""
        rel_scores = self.predict([doc])
        self.set_annotations([doc], rel_scores)
        return doc

    def predict(self, docs: Iterable[Doc]) -> Floats2d:
        """Apply the pipeline's model to a batch of docs, without modifying them."""
        scores = self.model.predict(docs)
        print("predicted scores", scores)
        scores = self.model.ops.asarray(scores)
        return scores

    def set_annotations(self, docs: Iterable[Doc], rel_scores: Floats2d) -> None:
        """Modify a batch of `Doc` objects, using pre-computed scores."""
        c = 0
        for doc in docs:
            for e1 in doc.ents:
                for e2 in doc.ents:
                    offset = (e1.start, e2.start)
                    if offset not in doc._.rel:
                        doc._.rel[offset] = {}
                    for j, label in enumerate(self.labels):
                        doc._.rel[offset][label] = rel_scores[c, j]
                    c += 1

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        set_annotations: bool = False,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to predict and get_loss."""
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        set_dropout_rate(self.model, drop)
        scores, bp_scores = self.model.begin_update([eg.predicted for eg in examples])
        loss, d_scores = self.get_loss(examples, scores)
        bp_scores(d_scores)
        if sgd is not None:
            self.model.finish_update(sgd)
        losses[self.name] += loss
        if set_annotations:
            docs = [eg.predicted for eg in examples]
            self.set_annotations(docs, scores=scores)
        return losses

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores."""
        truths, not_missing = self._examples_to_truth(examples)
        not_missing = self.model.ops.asarray(not_missing)
        d_scores = (scores - truths) / scores.shape[0]
        d_scores *= not_missing
        mean_square_error = (d_scores ** 2).sum(axis=1).mean()
        return float(mean_square_error), d_scores

    def begin_training(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        pipeline: Optional[List[Tuple[str, Callable[[Doc], Doc]]]] = None,
        sgd: Optional[Optimizer] = None,
    ) -> Optimizer:
        """Initialize the pipe for training, using a representative set
        of data examples.

        get_examples (Callable[[], Iterable[Example]]): Function that
            returns a representative sample of gold-standard Example objects.
        pipeline (List[Tuple[str, Callable]]): Optional list of pipeline
            components that this component is part of. Corresponds to
            nlp.pipeline.
        sgd (thinc.api.Optimizer): Optional optimizer. Will be created with
            create_optimizer if it doesn't exist.
        RETURNS (thinc.api.Optimizer): The optimizer.

        DOCS: https://nightly.spacy.io/api/textcategorizer#begin_training
        """
        examples = list(get_examples())
        for example in examples:
            relations = example.reference._.rel
            for indices, label_dict in relations.items():
                for label in label_dict.keys():
                    self.add_label(label)

        doc_sample = [eg.reference for eg in examples]
        label_sample = self._examples_to_truth(examples)
        if label_sample is None:
            raise ValueError("Call begin_training with relevant entities and relations annotated in "
                             "at least a few reference examples!")
        self.model.initialize(X=doc_sample, Y=label_sample)
        if sgd is None:
            sgd = self.create_optimizer()
        return sgd

    def _examples_to_truth(
        self, examples: List[Example]
    ) -> Optional[numpy.ndarray]:
        nr_candidates = 0
        print()
        print("example to truth")
        for eg in examples:
            for e1 in eg.reference.ents:
                for e2 in eg.reference.ents:
                    nr_candidates += 1
        if nr_candidates == 0:
            return None
        print("labels", self.labels)

        truths = numpy.zeros((nr_candidates, len(self.labels)), dtype="f")
        c = 0
        for eg in examples:
            for e1 in eg.reference.ents:
                for e2 in eg.reference.ents:
                    gold_label_dict = eg.reference._.rel.get((e1.start, e2.start), {})
                    print("candidate: ", (e1.start, e2.start), "-->", gold_label_dict)
                    for j, label in enumerate(self.labels):
                        truths[c, j] = gold_label_dict.get(label, 0)
                    c += 1

        truths = self.model.ops.asarray(truths)
        print("truths", truths)
        print()
        return truths
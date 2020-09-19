from typing import Tuple, List, Iterable, Iterator, Optional, Dict, Callable

from spacy.training.example import Example
from thinc.api import Model, Optimizer
from spacy.util import minibatch
from spacy.tokens.doc import Doc
from spacy.pipeline.pipe import Pipe
from spacy.vocab import Vocab
from spacy import Language


@Language.factory(
    "relation_extractor",
)
def make_relation_extractor(
    nlp: Language,
    name: str,
    model: Model,
):
    """Construct an EntityLinker component."""
    return RelationExtractor(
        nlp.vocab,
        model,
        name
    )


class RelationExtractor(Pipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "relex",
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

    def predict(self, docs: Iterable[Doc]):
        """Apply the pipeline's model to a batch of docs, without modifying them."""
        scores = self.model.predict(docs)
        scores = self.model.ops.asarray(scores)
        return scores

    def set_annotations(self, docs: Iterable[Doc], scores) -> None:
        """Modify a batch of `Doc` objects, using pre-computed scores."""
        for i, doc in enumerate(docs):
            for j, label in enumerate(self.labels):
                pass
                # doc._[label] = float(scores[i, j])

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
        subbatch = []  # Select a subbatch of examples to initialize the model
        for example in islice(get_examples(), 10):
            if len(subbatch) < 2:
                subbatch.append(example)
            for cat in example.y.cats:
                self.add_label(cat)
        doc_sample = [eg.reference for eg in subbatch]
        label_sample, _ = self._examples_to_truth(subbatch)
        self.model.initialize(X=doc_sample, Y=label_sample)
        if sgd is None:
            sgd = self.create_optimizer()
        return sgd

from itertools import islice
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
from wasabi import Printer


Doc.set_extension("rel", default={}, force=True)
msg = Printer()


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
        # check that there are actually any candidates in this batch of examples
        total_candidates = len(self.model.attrs["get_candidates"](doc))
        if total_candidates == 0:
            msg.info("Could not determine any candidates in doc - returning doc as is.")
            return doc

        predictions = self.predict([doc])
        self.set_annotations([doc], predictions)
        return doc

    def predict(self, docs: Iterable[Doc]) -> Floats2d:
        """Apply the pipeline's model to a batch of docs, without modifying them."""
        total_candidates = sum([len(self.model.attrs["get_candidates"](doc)) for doc in docs])
        if total_candidates == 0:
            msg.info("Could not determine any candidates in any docs - can not make any predictions.")
        scores = self.model.predict(docs)
        # with numpy.printoptions(precision=2, suppress=True):
        #     print(f"predicted scores: {scores}")
        return self.model.ops.asarray(scores)

    def set_annotations(self, docs: Iterable[Doc], scores: Floats2d) -> None:
        """Modify a batch of `Doc` objects, using pre-computed scores."""
        c = 0
        get_candidates = self.model.attrs["get_candidates"]
        for doc in docs:
            for (e1, e2) in get_candidates(doc):
                offset = (e1.start, e2.start)
                if offset not in doc._.rel:
                    doc._.rel[offset] = {}
                for j, label in enumerate(self.labels):
                    doc._.rel[offset][label] = scores[c, j]
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

        # mimic having an actual NER in the pipeline (TODO: solve this better)
        for eg in examples:
            eg.predicted.ents = eg.reference.ents

        # check that there are actually any candidates in this batch of examples
        total_candidates = 0
        for eg in examples:
            total_candidates += len(self.model.attrs["get_candidates"](eg.predicted))
        if total_candidates == 0:
            msg.info("Could not determine any candidates in doc.")
            return losses

        # run the model
        predictions, bp_scores = self.model.begin_update([eg.predicted for eg in examples])
        loss, d_scores = self.get_loss(examples, predictions)
        bp_scores(d_scores)
        if sgd is not None:
            self.model.finish_update(sgd)
        losses[self.name] += loss
        if set_annotations:
            docs = [eg.predicted for eg in examples]
            self.set_annotations(docs, predictions)
        return losses

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores."""
        truths = self._examples_to_truth(examples)
        d_scores = (scores - truths)
        mean_square_error = (d_scores ** 2).sum(axis=1).mean()
        # print(f"mean_square_error {mean_square_error}")
        return float(mean_square_error), d_scores

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Language = None,
        labels: Optional[List[str]] = None,
    ):
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
        if labels is not None:
            for label in labels:
                self.add_label(label)
        else:
            for example in get_examples():
                relations = example.reference._.rel
                for indices, label_dict in relations.items():
                    for label in label_dict.keys():
                        self.add_label(label)
        self._require_labels()

        subbatch = list(islice(get_examples(), 10))
        doc_sample = [eg.reference for eg in subbatch]
        label_sample = self._examples_to_truth(subbatch)
        if label_sample is None:
            raise ValueError("Call begin_training with relevant entities and relations annotated in "
                             "at least a few reference examples!")
        self.model.initialize(X=doc_sample, Y=label_sample)

    def _examples_to_truth(
        self, examples: List[Example]
    ) -> Optional[numpy.ndarray]:
        nr_candidates = 0
        for eg in examples:
            nr_candidates += len(self.model.attrs["get_candidates"](eg.reference))
        if nr_candidates == 0:
            return None

        truths = numpy.zeros((nr_candidates, len(self.labels)), dtype="f")
        c = 0
        for i, eg in enumerate(examples):
            for (e1, e2) in self.model.attrs["get_candidates"](eg.reference):
                gold_label_dict = eg.reference._.rel.get((e1.start, e2.start), {})
                for j, label in enumerate(self.labels):
                    truths[c, j] = gold_label_dict.get(label, 0)
                c += 1

        truths = self.model.ops.asarray(truths)
        # print(f"truths: {truths}")
        # print()
        return truths

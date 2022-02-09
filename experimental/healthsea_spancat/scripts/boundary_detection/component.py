import numpy
from typing import List, Dict, Callable, Tuple, Optional, Iterable, Any, cast
from thinc.api import Config, Model, get_current_ops, set_dropout_rate, Ops
from thinc.api import Optimizer
from thinc.types import Ragged, Ints2d, Floats2d, Ints1d

from spacy.language import Language
from spacy.util import registry
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.tokens import Doc, SpanGroup, Span, Token
from spacy.training import Example, validate_examples
from spacy.scorer import Scorer

import sbd

sbd_default_config = """
[model]
@architectures = "spacy.PyTorchSpanBoundaryDetection.v1"
hidden_size = 128

[model.scorer]
@layers = "spacy.LinearLogistic.v1"
nO=2

[model.tok2vec]
@architectures = "spacy.Tok2Vec.v1"

[model.tok2vec.embed]
@architectures = "spacy.MultiHashEmbed.v1"
width = 96
rows = [5000, 2000, 1000, 1000]
attrs = ["ORTH", "PREFIX", "SUFFIX", "SHAPE"]
include_static_vectors = false

[model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v1"
width = ${model.tok2vec.embed.width}
window_size = 1
maxout_pieces = 3
depth = 4
"""

DEFAULT_SBD_MODEL = Config().from_str(sbd_default_config)["model"]

@Language.factory(
    "boundarydetection",
    assigns=[""],
    default_config={
        "threshold": 0.5,
        "spans_key": "sc",
        "model": DEFAULT_SBD_MODEL,
        "scorer": {"@scorers": "spacy.sbd_scorer.v1"},
    },
    default_score_weights={"sbd_start_f": 1.0, "sbd_start_p": 0.0, "sbd_start_r": 0.0, "sbd_end_f": 1.0, "sbd_end_p": 0.0, "sbd_end_r": 0.0},
)
def make_sbd(
    nlp: Language,
    name: str,
    model: Model[Tuple[List[Doc], Ragged], Floats2d],
    scorer: Optional[Callable],
    threshold: float,

) -> "SpanBoundaryDetection":
    """Create a SpanBoundaryDetection component. The component predicts whether a token is the start or the end of a span.
    model (Model[List[Doc], Floats2d]): A model instance that
        is given a list of documents and predicts a probability for each token.
    threshold (float): Minimum probability to consider a prediction positive.
    """
    return SpanBoundaryDetection(
        nlp.vocab,
        model=model,
        threshold=threshold,
        name=name,
        scorer=scorer,
    )


def sbd_score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    kwargs = dict(kwargs)
    attr_prefix = "spans_"
    key = kwargs["spans_key"]
    kwargs.setdefault("attr", f"{attr_prefix}{key}")
    kwargs.setdefault("allow_overlap", True)
    kwargs.setdefault(
        "getter", lambda doc, key: doc.spans.get(key[len(attr_prefix) :], [])
    )
    kwargs.setdefault("has_annotation", lambda doc: key in doc.spans)
    return Scorer.score_spans(examples, **kwargs)


@registry.scorers("spacy.sbd_scorer.v1")
def make_sbd_scorer():
    return sbd_score


class SpanBoundaryDetection(TrainablePipe):
    """Pipeline that learns start and end tokens of spans"""

    def __init__(
        self,
        model: Model[List[Doc], Floats2d],
        name: str = "sbd",
        *,
        threshold: float = 0.5,
        scorer: Optional[Callable] = sbd_score,
    ) -> None:
        """Initialize the span boundary detector.
        model (thinc.api.Model): The Thinc Model powering the pipeline component.
        name (str): The component instance name, used to add entries to the
            losses during training.
        threshold (float): Minimum probability to consider a prediction
            positive.
        scorer (Optional[Callable]): The scoring method.
        """
        self.cfg = {
            "threshold": threshold,
        }
        self.model = model
        self.name = name
        self.scorer = scorer
        Token.set_extension("span_start", default=0)
        Token.set_extension("span_end", default=0)

    def predict(self, docs: Iterable[Doc]):
        """Apply the pipeline's model to a batch of docs, without modifying them.
        docs (Iterable[Doc]): The documents to predict.
        RETURNS: The models prediction for each document.
        """
        scores = self.model.predict(docs) 
        return scores

    def set_annotations(self, docs: Iterable[Doc], scores: Floats2d) -> None:
        """Modify a batch of Doc objects, using pre-computed scores.
        docs (Iterable[Doc]): The documents to modify.
        scores: The scores to set, produced by SpanCategorizer.predict.
        """
        lengths = [len(doc) for doc in docs]

        offset = 0
        scores_per_doc = []
        for length in lengths:
            scores_per_doc.append(scores[offset:offset+length])
            offset += length

        for doc,score_doc in zip(docs,scores):
            for token, score_token in zip(doc,score_doc):
                token._.span_start = score_token[0]
                token._.span_end = score_token[1]

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to predict and get_loss.
        examples (Iterable[Example]): A batch of Example objects.
        drop (float): The dropout rate.
        sgd (thinc.api.Optimizer): The optimizer.
        losses (Dict[str, float]): Optional record of the loss during training.
            Updated using the component name as the key.
        RETURNS (Dict[str, float]): The updated losses dictionary.
        """
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        validate_examples(examples, "SpanCategorizer.update")
        docs = [eg.predicted for eg in examples]
        references  = [eg.reference for eg in examples]
        set_dropout_rate(self.model, drop)
        scores, backprop_scores = self.model.begin_update(docs)
        loss, d_scores = self.get_loss(references, scores)
        backprop_scores(d_scores) 
        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += loss
        return losses

    def get_loss(self,scores, docs) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores.
        examples (Iterable[Examples]): The batch of examples.
        scores: Scores representing the model's predictions.
        RETURNS (Tuple[float, float]): The loss and the gradient.
        """
        reference_results = self.get_reference(docs)
        d_scores = self.model.ops.asarray(scores) - self.model.ops.asarray(reference_results)
        loss = float((d_scores ** 2).sum())
        return loss, d_scores

    def get_reference(self, docs) -> Floats2d:
        """Create a reference list of token probabilities for calculating loss and metrics"""
        reference_results = []
        for doc in docs:
            start_indices = []
            end_indices = []

            for spankey in doc.spans:
                for span in doc.spans[spankey]:
                    start_indices.append(span.start)
                    end_indices.append(span.end)

            for token in doc:
                is_start = 0
                is_end = 0
                if token.i in start_indices:
                    is_start = 1
                if token.i in end_indices:
                    is_end = 1
                reference_results.append(self.model.ops.asarray([is_start,is_end],dtype="float32"))

        reference_results = self.model.ops.asarray(reference_results)
        return reference_results

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Optional[Language] = None,
    ) -> None:
        """Initialize the pipe for training, using a representative set
        of data examples.
        get_examples (Callable[[], Iterable[Example]]): Function that
            returns a representative sample of gold-standard Example objects.
        nlp (Optional[Language]): The current nlp object the component is part of.
        labels (Optional[List[str]]): The labels to add to the component, typically generated by the
            `init labels` command. If no labels are provided, the get_examples
            callback is used to extract the labels from the data.
        """
        subbatch: List[Example] = []

        for eg in get_examples():
            subbatch.append(eg)

        if subbatch:
            docs = [eg.reference for eg in subbatch]
            Y = self.get_reference(docs)
            self.model.initialize(X=docs[:100], Y=Y[:100])
        else:
            self.model.initialize()

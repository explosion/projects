from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple
import numpy
from thinc.api import (
    Config,
    Model,
    set_dropout_rate,
    SequenceCategoricalCrossentropy,
    Optimizer,
)
from thinc.types import Floats2d, Floats3d, Ints2d
import warnings
from itertools import islice

from spacy.tokens.doc import Doc
from spacy.vocab import Vocab

from spacy.training import Example
from spacy.training.iob_utils import biluo_tags_to_spans, biluo_to_iob, iob_to_biluo
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.pipeline.pipe import deserialize_config
from spacy.language import Language
from spacy.attrs import POS, ID
from spacy.parts_of_speech import X
from spacy.errors import Errors
from spacy.scorer import get_ner_prf
from spacy.training import validate_examples, validate_get_examples
from spacy import util


def set_torch_dropout_rate(model: Model, dropout_rate: float):
    """Set dropout rate for Thinc and wrapped PyTorch models

    Args:
        model (Model): Thinc Model (with PyTorch sub-modules)
        dropout_rate (float): Dropout rate
    """
    set_dropout_rate(model, dropout_rate)
    func = model.get_ref("torch_model").attrs["set_dropout_rate"]
    func(dropout_rate)


default_model_config = """
[model]
@architectures = "TorchEntityRecognizer.v1"
hidden_width = 48
dropout = 0.1
nO = null

[model.tok2vec]
@architectures = "spacy.HashEmbedCNN.v1"
pretrained_vectors = null
width = 96
depth = 4
embed_size = 2000
window_size = 1
maxout_pieces = 3
subword_features = true
"""
DEFAULT_MODEL = Config().from_str(default_model_config)["model"]


@Language.factory(
    "torch_ner",
    assigns=["doc.ents", "token.ent_iob", "token.ent_type"],
    default_config={"model": DEFAULT_MODEL},
    default_score_weights={
        "ents_f": 1.0,
        "ents_p": 0.0,
        "ents_r": 0.0,
        "ents_per_type": None,
    },
)
def make_torch_entity_recognizer(nlp: Language, name: str, model: Model):
    """Construct a PyTorch based Named Entity Recognition model
    model (Model[List[Doc], List[Floats2d]]): A model instance that predicts
        the tag probabilities. The output vectors should match the number of tags
        in size, and be normalized as probabilities (all scores between 0 and 1,
        with the rows summing to 1).
    """
    return TorchEntityRecognizer(nlp.vocab, model, name)


class TorchEntityRecognizer(TrainablePipe):
    """Pipeline component Named Entity Recognition using PyTorch"""

    def __init__(self, vocab: Vocab, model: Model, name: str = "torch_ner"):
        """Initialize a part-of-speech tagger.
        vocab (Vocab): The shared vocabulary.
        model (thinc.api.Model): The Thinc Model powering the pipeline component.
        name (str): The component instance name, used to add entries to the
            losses during training.
        """
        self.vocab = vocab
        self.model = model
        self.name = name
        cfg = {"labels": []}
        self.cfg = dict(sorted(cfg.items()))

    @property
    def labels(self) -> Tuple[str, ...]:
        """The labels currently added to the component.
        RETURNS (Tuple[str]): The labels.
        """
        labels = ["O"]
        for label in self.cfg["labels"]:
            for iob in ["B", "I"]:
                labels.append(f"{iob}-{label}")
        return tuple(labels)

    def predict(self, docs: Iterable[Doc]) -> Ints2d:
        """Apply the pipeline's model to a batch of docs, without modifying them.
        docs (Iterable[Doc]): The documents to predict.
        RETURNS: The models prediction for each document.
        """
        if not any(len(doc) for doc in docs):
            # Handle cases where there are no tokens in any docs.
            n_labels = len(self.labels)
            guesses = [self.model.ops.alloc((0, n_labels)) for doc in docs]
            assert len(guesses) == len(docs)
            return guesses
        scores = self.model.predict(docs)

        assert len(scores) == len(docs), (len(scores), len(docs))
        guesses = []
        for doc_scores in scores:
            doc_guesses = doc_scores.argmax(axis=1)
            if not isinstance(doc_guesses, numpy.ndarray):
                doc_guesses = doc_guesses.get()
            guesses.append(doc_guesses)
        assert len(guesses) == len(docs)
        return guesses

    def set_annotations(self, docs: Iterable[Doc], preds: Ints2d):
        """Modify a batch of documents, using pre-computed scores.
        docs (Iterable[Doc]): The documents to modify.
        preds (Ints2d): The IDs to set, produced by TorchEntityRecognizer.predict.
        """
        if isinstance(docs, Doc):
            docs = [docs]
        for i, doc in enumerate(docs):
            doc_tag_ids = preds[i]
            labels = iob_to_biluo([self.labels[tag_id] for tag_id in doc_tag_ids])
            try:
                spans = biluo_tags_to_spans(doc, labels)
            except ValueError:
                # Note:
                # biluo_tags_to_spans will raise an exception for an invalid tag sequence
                # this could be fixed using a more complex transition system
                # (e.g. a Conditional Random Field model head)
                spans = []
            doc.ents = spans

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        sgd: Optimizer = None,
        losses: Dict[str, float] = None,
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
        validate_examples(examples, "TorchEntityRecognizer.update")
        if not any(len(eg.predicted) if eg.predicted else 0 for eg in examples):
            # Handle cases where there are no tokens in any docs.
            return losses
        set_torch_dropout_rate(self.model, drop)
        tag_scores, bp_tag_scores = self.model.begin_update(
            [eg.predicted for eg in examples]
        )
        for sc in tag_scores:
            if self.model.ops.xp.isnan(sc.sum()):
                raise ValueError(Errors.E940)
        loss, d_tag_scores = self.get_loss(examples, tag_scores)
        bp_tag_scores(d_tag_scores)
        if sgd not in (None, False):
            self.finish_update(sgd)

        losses[self.name] += loss
        return losses

    def get_loss(self, examples, scores):
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores.
        examples (Iterable[Examples]): The batch of examples.
        scores: Scores representing the model's predictions.
        RETURNS (Tuple[float, float]): The loss and the gradient.
        """
        validate_examples(examples, "TorchEntityRecognizer.get_loss")
        loss_func = SequenceCategoricalCrossentropy(names=self.labels, normalize=False)
        # Convert empty tag "" to missing value None so that both misaligned
        # tokens and tokens with missing annotation have the default missing
        # value None.
        truths = []
        for eg in examples:
            eg_truths = [
                tag if tag != "" else None for tag in biluo_to_iob(eg.get_aligned_ner())
            ]
            truths.append(eg_truths)
        d_scores, loss = loss_func(scores, truths)
        if self.model.ops.xp.isnan(loss):
            raise ValueError(Errors.E910.format(name=self.name))
        return float(loss), d_scores

    def initialize(self, get_examples, *, nlp=None, labels=None):
        """Initialize the pipe for training, using a representative set
        of data examples.
        get_examples (Callable[[], Iterable[Example]]): Function that
            returns a representative sample of gold-standard Example objects..
        nlp (Language): The current nlp object the component is part of.
        labels: The labels to add to the component, typically generated by the
            `init labels` command. If no labels are provided, the get_examples
            callback is used to extract the labels from the data.
        """
        validate_get_examples(get_examples, "TorchEntityRecognizer.initialize")
        if labels is not None:
            for tag in labels:
                self.add_label(tag)
        else:
            tags = set()
            for example in get_examples():
                for token in example.y:
                    if token.ent_type_:
                        tags.add(token.ent_type_)
            for tag in sorted(tags):
                self.add_label(tag)
        doc_sample = []
        for example in islice(get_examples(), 10):
            doc_sample.append(example.x)

        self._require_labels()
        assert len(doc_sample) > 0, Errors.E923.format(name=self.name)
        self.model.initialize(X=doc_sample, Y=self.labels)
        nlp.config["components"][self.name]["model"]["nO"] = len(self.labels)

    def add_label(self, label):
        """Add a new label to the pipe.
        label (str): The label to add.
        RETURNS (int): 0 if label is already present, otherwise 1.
        """
        if not isinstance(label, str):
            raise ValueError(Errors.E187)
        if label in self.labels:
            return 0
        self._allow_extra_label()
        self.cfg["labels"].append(label)
        self.vocab.strings.add(label)
        return 1

    def score(self, examples, **kwargs):
        """Score a batch of examples.
        examples (Iterable[Example]): The examples to score.
        RETURNS (Dict[str, Any]): The NER precision, recall and f-scores.
        """
        validate_examples(examples, "TorchEntityRecognizer.score")
        return get_ner_prf(examples)

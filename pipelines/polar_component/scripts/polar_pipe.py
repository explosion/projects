from spacy import Language
from spacy.tokens import Doc
from spacy.pipeline import Pipe
from spacy.vocab import Vocab

from thinc.types import Floats1d, Floats2d
from thinc.api import Model, CosineDistance, get_ops

from dataclasses import dataclass

Doc.set_extension("poles", default={}, force=True)


@dataclass
class Axis:
    """An invididual semantic axis."""

    neg: str
    pos: str
    vector: Floats1d

    def get_key(self, sep="-"):
        return f"{self.neg}{sep}{self.pos}"


@Language.factory(
    "polar",
    requires=["doc.vector"],
    default_config={},
    default_score_weights={},
)
def make_polar_embeddings(
    nlp: Language,
    name: str,
):
    return PolarEmbeddings(
        nlp,
        name,
    )


class PolarEmbeddings(Pipe):
    """PolarEmbeddings let you turn normal word embeddings into embeddings
    oriented along axes of meaning, preserving the overall distance of the
    original embeddings while giving dimensions semantic meaning.
    """

    def __init__(
        self,
        nlp: Language,
        name: str = "polar",
        *,
        separator: str = "-",
    ) -> None:
        self.nlp = nlp
        self.name = name
        self.separator = separator

        self.ops = get_ops("numpy")
        self.cosine = CosineDistance()
        self._matrix = None
        self.axes = []
        self.cfg = {}

    def get_average_neighbors(self, query, nn=150):
        """Given a query vector, return the average of the nearest vecs.

        Used to calculate pole vectors."""
        # Note that in spaCy pipelines for many languages vectors include case
        # variations, so nn should be larger than in the reference paper
        seed = self.nlp.vocab[query].vector
        vectors = self.nlp.vocab.vectors
        qarray = self.ops.asarray2f([seed])
        keys, best_rows, scores = vectors.most_similar(qarray, n=nn)
        targets = vectors.data[best_rows].squeeze()

        return self.ops.xp.mean(targets, axis=0)

    def add_axis(self, neg: str, pos: str) -> None:
        """Add a new pole to the pipe. Pass the negative word first."""

        notfound = "Anchor word '{}' not found in vocab."

        if not self.nlp.vocab.has_vector(neg):
            raise KeyError(notfound.format(neg))
        if not self.nlp.vocab.has_vector(pos):
            raise KeyError(notfound.format(pos))

        pv = self.get_average_neighbors(pos)
        nv = self.get_average_neighbors(neg)
        self.axes.append(Axis(neg, pos, pv - nv))
        self._update_axes_matrix()

    def _update_axes_matrix(self) -> None:
        vecs = [aa.vector for aa in self.axes]
        self._matrix = self.ops.asarray2f(vecs).T

    def __call__(self, doc: Doc) -> Doc:
        docvec = self.ops.xp.expand_dims(doc.vector, axis=1)
        docvec = docvec.repeat(len(self.axes), axis=1)
        dists = self.cosine.get_similarity(docvec.T, self._matrix.T)

        for axis, dist in zip(self.axes, dists):
            key = axis.get_key(sep=self.separator)
            doc._.poles[key] = float(dist)
        return doc

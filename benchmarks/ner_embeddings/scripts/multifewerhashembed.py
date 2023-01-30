from typing import Callable, Dict, Tuple, Optional, Any, Union, cast, TypeVar
from typing import List
from thinc.api import chain, concatenate, with_array, Model, list2ragged
from thinc.api import ragged2list, Maxout
from thinc.initializers import uniform_init
from thinc.layers.array_getitem import ints_getitem
from thinc.types import Floats1d, Floats2d, Ints1d, Ints2d, Literal, Ragged
from thinc.util import partial

from spacy.tokens import Doc
from spacy.util import registry
from spacy.ml.staticvectors import StaticVectors
from spacy.ml.featureextractor import FeatureExtractor


InT = TypeVar("InT", bound=Union[Ints1d, Ints2d])
OutT = Floats2d


def FewerHashEmbed(
    nO: int,
    nV: int,
    *,
    seed: Optional[int] = None,
    num_hashes: Literal[1, 2, 3, 4] = 4,
    column: Optional[int] = None,
    initializer: Callable = uniform_init,
    dropout: Optional[float] = None,
) -> Model[InT, OutT]:
    """
    An embedding layer that uses the "hashing trick" to map keys to distinct values.
    This layer is a slightly modified version of MultiHashEmbed that parametrizes
    the number of hash functions.

    The hashing trick involves hashing each key 1-4 times with distinct seeds,
    to produce 1-4 likely differing values. Those values are modded into the
    table, and the resulting vectors summed to produce a single result. Because
    it's unlikely that two different keys will collide on all multiple
    “buckets”, most distinct keys will receive a distinct vector under this
    scheme, even when the number of vectors in the table is very low.
    """
    attrs: Dict[str, Any] = {"column": column, "seed": seed, "num_hashes": num_hashes}
    if dropout is not None:
        attrs["dropout_rate"] = dropout
    model: Model = Model(
        "fewerhashembed",
        forward,
        init=partial(init, initializer),
        params={"E": None},
        dims={"nO": nO, "nV": nV, "nI": None},
        attrs=attrs,
    )
    if seed is None:
        model.attrs["seed"] = model.id
    model.attrs["num_hashes"] = num_hashes
    if column is not None:
        # This is equivalent to array[:, column]. What you're actually doing
        # there is passing in a tuple: array[(:, column)], except in the context
        # of array indexing, the ":" creates an object slice(0, None).
        # So array[:, column] is array.__getitem__(slice(0), column).
        model = chain(ints_getitem((slice(0, None), column)), model)
    model.attrs["column"] = column
    return cast(Model[InT, OutT], model)


def forward(
    model: Model[Ints1d, OutT], ids: Ints1d, is_train: bool
) -> Tuple[OutT, Callable]:
    vectors = cast(Floats2d, model.get_param("E"))
    nV = vectors.shape[0]
    nO = vectors.shape[1]
    if len(ids) == 0:
        output: Floats2d = model.ops.alloc((0, nO), dtype=vectors.dtype)
    else:
        ids = model.ops.as_contig(ids, dtype="uint64")
        nN = ids.shape[0]
        seed: int = model.attrs["seed"]
        num_hashes: int = model.attrs["num_hashes"]
        keys = model.ops.hash(ids, seed)[:, :num_hashes] % nV
        output = model.ops.gather_add(vectors, keys)
        drop_mask = None
        if is_train:
            dropout: Optional[float] = model.attrs.get("dropout_rate")
            drop_mask = cast(Floats1d, model.ops.get_dropout_mask((nO,), dropout))
            if drop_mask is not None:
                output *= drop_mask

    def backprop(d_vectors: OutT) -> Ints1d:
        if drop_mask is not None:
            d_vectors *= drop_mask
        dE = model.ops.alloc2f(*vectors.shape)
        keysT = model.ops.as_contig(keys.T, dtype="i")
        for i in range(keysT.shape[0]):
            model.ops.scatter_add(dE, keysT[i], d_vectors)
        model.inc_grad("E", dE)
        dX = model.ops.alloc1i(nN)
        return dX

    return output, backprop


def init(
    initializer: Callable,
    model: Model[Ints1d, OutT],
    X: Optional[Ints1d] = None,
    Y: Optional[OutT] = None,
) -> None:
    E = initializer(model.ops, (model.get_dim("nV"), model.get_dim("nO")))
    model.set_param("E", E)


@registry.architectures("spacy.MultiFewerHashEmbed.v1")
def MultiFewerHashEmbed(
    width: int,
    num_hashes: Literal[1, 2, 3, 4],
    attrs: List[Union[str, int]],
    rows: List[int],
    include_static_vectors: bool,
) -> Model[List[Doc], List[Floats2d]]:
    """Construct an embedding layer that separately embeds a number of lexical
    attributes using hash embedding, concatenates the results, and passes it
    through a feed-forward subnetwork to build a mixed representation.

    The features used can be configured with the 'attrs' argument. The suggested
    attributes are NORM, PREFIX, SUFFIX and SHAPE. This lets the model take into
    account some subword information, without constructing a fully character-based
    representation. If pretrained vectors are available, they can be included in
    the representation as well, with the vectors table kept static
    (i.e. it's not updated).

    The `width` parameter specifies the output width of the layer and the widths
    of all embedding tables. If static vectors are included, a learned linear
    layer is used to map the vectors to the specified width before concatenating
    it with the other embedding outputs. A single Maxout layer is then used to
    reduce the concatenated vectors to the final width.

    The `rows` parameter controls the number of rows used by the `HashEmbed`
    tables. The HashEmbed layer needs surprisingly few rows, due to its use of
    the hashing trick. Generally between 2000 and 10000 rows is sufficient,
    even for very large vocabularies. A number of rows must be specified for each
    table, so the `rows` list must be of the same length as the `attrs` parameter.

    width (int): The output width. Also used as the width of the embedding tables.
        Recommended values are between 64 and 300.
    attrs (list of attr IDs): The token attributes to embed. A separate
        embedding table will be constructed for each attribute.
    rows (List[int]): The number of rows in the embedding tables. Must have the
        same length as attrs.
    include_static_vectors (bool): Whether to also use static word vectors.
        Requires a vectors table to be loaded in the Doc objects' vocab.
    """
    if len(rows) != len(attrs):
        raise ValueError(f"Mismatched lengths: {len(rows)} vs {len(attrs)}")
    seed = 7

    def make_hash_embed(index):
        nonlocal seed
        seed += 1
        return FewerHashEmbed(
            width,
            rows[index],
            column=index,
            seed=seed,
            num_hashes=num_hashes,
            dropout=0.0,
        )

    embeddings = [make_hash_embed(i) for i in range(len(attrs))]
    concat_size = width * (len(embeddings) + include_static_vectors)
    max_out: Model[Ragged, Ragged] = with_array(
        Maxout(width, concat_size, nP=3, dropout=0.0, normalize=True)
    )
    if include_static_vectors:
        feature_extractor: Model[List[Doc], Ragged] = chain(
            FeatureExtractor(attrs),
            cast(Model[List[Ints2d], Ragged], list2ragged()),
            with_array(concatenate(*embeddings)),
        )
        model = chain(
            concatenate(
                feature_extractor,
                StaticVectors(width, dropout=0.0),
            ),
            max_out,
            ragged2list(),
        )
    else:
        model = chain(
            FeatureExtractor(list(attrs)),
            cast(Model[List[Ints2d], Ragged], list2ragged()),
            with_array(concatenate(*embeddings)),
            max_out,
            ragged2list(),
        )
    return model

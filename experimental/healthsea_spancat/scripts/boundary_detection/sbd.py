from lib2to3.pytree import convert
from thinc.types import Floats2d
import torch 

from typing import List, Tuple, Callable
from thinc.api import Model, chain, PyTorchWrapper
from thinc.api import Maxout
from thinc.types import Floats2d

from spacy.util import registry
from spacy.tokens import Doc
from thinc.util import torch2xp, xp2torch
from thinc.api import ArgsKwargs

from itertools import islice

@registry.architectures("spacy.PyTorchSpanBoundaryDetection.v1")
def build_boundary_model(
    tok2vec: Model[List[Doc], List[Floats2d]],
    scorer: Model[Floats2d,Floats2d],
    hidden_size: int
) -> Model[List[Doc], Floats2d]:

    model = chain(
        tok2vec, PyTorchWrapper(PytorchTokenFeaturer(), convert_inputs=convert_inputs), Maxout(nO=hidden_size, normalize=True), scorer
    )
    model.set_ref("tok2vec", tok2vec)
    model.set_ref("scorer", scorer)
    return model


class PytorchTokenFeaturer(torch.nn.Module):
    """
    A single-layer that computes new token vectors based on surrounding tokens
    The resulting token vector look like this (token_vector, mean(surrounding tokens), max(surrounding tokens))
    """

    def __init__(self):
        super(PytorchTokenFeaturer, self).__init__()

    def forward(self, input: List[Floats2d]) -> Floats2d:

        modified_vectors = []

        # Iterate over docs
        for token in input:

            # Calculate features
            token_max = torch.max(token,dim=0)
            token_mean = torch.mean(token,dim=0)
            token_cat = torch.cat((token[0,:],token_mean,token_max.values), dim=0)

            # Add to list
            modified_vectors.append(token_cat)

        modified_vectors = torch.stack(modified_vectors)

        return modified_vectors


def convert_inputs(model: Model, X, is_train: bool):
 
    window_size = 2
    lengths = [len(x) for x in X]

    converted_input = _get_window_sized_tokens(X, window_size)

    def backprop(dXtorch):
        original_tokens = []
        for token_batch in dXtorch.args:
            for token in token_batch:
                original_tokens.append(token[0])
        original_tokens = torch.stack(original_tokens)

        original_tokens_xp = torch2xp(original_tokens)

        offset = 0
        original_tokens_xp_list = []
        for length in lengths:
            original_tokens_xp_list.append(original_tokens_xp[offset:offset+length])
            offset += length

        return original_tokens_xp_list
    
    return ArgsKwargs(args=(converted_input,), kwargs={}), backprop


def _get_window_sized_tokens(input: List[Floats2d], window_size: int) -> List[Floats2d]:
    """Create lists of tensors for each token inside the window_size for every token in the doc"""
    modified_vectors = []
    vector_count = (window_size*2)+1

    # Iterate over docs
    for doc in input:
        # Iterate over token vectors
        for i, token_vector in enumerate(doc):
            token_tensor = torch.tensor(token_vector, dtype=torch.float32)
            window_vectors = [token_tensor]
            _min = window_size
            _max = window_size

            if i + _max >= len(doc):
                _max = (len(doc) - i) - 1

            if i - _min < 0:
                _min = i

            # Add window tokens
            for k in range(i - _min, i + _max + 1):
                if i != k:
                    tensor = torch.tensor(doc[k], dtype=torch.float32)
                    window_vectors.append(tensor)

            # Fill gaps
            for j in range(0, vector_count-len(window_vectors)):
                window_vectors.append(token_tensor)

            modified_vectors.append(torch.stack(window_vectors))

    modified_vectors = torch.stack(modified_vectors)
    modified_vectors.requires_grad = True
    return modified_vectors
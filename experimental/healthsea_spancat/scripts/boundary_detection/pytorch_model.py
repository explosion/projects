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

def convert_inputs(model: Model, X, is_train: bool):

    print(X)

    converted_list=[]
    for x in X:
        converted_list.append(xp2torch(x, requires_grad=True))

    def backprop(dXtorch):
        print(dXtorch)
    
    return converted_list, backprop


@registry.architectures("spacy.PyTorchSpanBoundaryDetection.v1")
def build_boundary_model(
    tok2vec: Model[List[Doc], List[Floats2d]],
    window_size: int
) -> Model[List[Doc], Floats2d]:

    model = chain(
        tok2vec, PyTorchWrapper(PytorchTokenFeaturer(window_size))
    )
    model.set_ref("tok2vec", tok2vec)
    return model


class PytorchTokenFeaturer(torch.nn.Module):
    """
    A single-layer that computes new token vectors based on surrounding tokens
    The resulting token vector look like this (token_vector, mean(surrounding tokens), max(surrounding tokens))
    """

    def __init__(self, window_size:int):
        self.window_size = window_size

    def forward(self, input_docs: List[Floats2d]) -> Floats2d:
        
        modified_vectors = []

        # Iterate over docs
        for doc in input_docs:
            # Iterate over token vectors
            for i, token_vector in enumerate(doc):
                token_tensor = torch.tensor([token_vector], dtype=torch.float32, requires_grad=True)
                window_vectors = []
                _min, _max = self.window_size

                if i + _max >= len(doc):
                    _max = (len(doc) - i) - 1

                if i - _min < 0:
                    _min = i

                for k in range(i - _min, i + _max + 1):
                    if i != k:
                        window_vectors.append(doc[k])

                # Calculate features
                x_window = torch.tensor(window_vectors, dtype=torch.float32)
                x = torch.cat((token_tensor,x_window), dim=0)
                x_max = torch.max(x,dim=0)
                x_mean = torch.mean(x,dim=0)
                x_cat = torch.cat((x[0,:],x_mean,x_max.values), dim=0)

                # Add to list
                modified_vectors.append(x_cat)
        
        return modified_vectors
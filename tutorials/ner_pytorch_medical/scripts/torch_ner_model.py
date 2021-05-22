from typing import Optional, List
from thinc.api import zero_init, with_array, Softmax, chain, Model, PyTorchWrapper, PyTorchLSTM, with_padded, noop
from thinc.types import Floats2d

import spacy
from spacy.ml import CharacterEmbed, MultiHashEmbed
from spacy.tokens import Doc
from spacy.util import registry
import torch
from torch import nn

torch.manual_seed(1)


@registry.architectures("TorchEntityRecognizer.v1")
def build_torch_ner_model(
    tok2vec: Model[List[Doc], List[Floats2d]],
    width: int,
    hidden_width: int,
    embed_size: int,
    nM: int,
    nC: int,
    dropout: float,
    nO: Optional[int] = None,
) -> Model[List[Doc], List[Floats2d]]:
    """Build a tagger model, using a provided token-to-vector component. The tagger
    model simply adds a linear layer with softmax activation to predict scores
    given the token vectors.
    tok2vec (Model[List[Doc], List[Floats2d]]): The token-to-vector subnetwork.
    nO (int or None): The number of tags to output. Inferred from the data if None.
    """
    
    t2v_width = tok2vec.get_dim("nO") if tok2vec.has_dim("nO") else None

    torch_model = nn.Sequential(
        nn.Linear(t2v_width, hidden_width),
        nn.ReLU(),
        nn.Dropout2d(dropout),
        nn.Linear(hidden_width, nO),
        nn.ReLU(),
        nn.Dropout2d(dropout),
        nn.Softmax(dim=1)
    )
    wrapped_pt_model = PyTorchWrapper(torch_model)
    model = chain(tok2vec, with_array(wrapped_pt_model))
    model.set_ref("tok2vec", tok2vec)
    return model

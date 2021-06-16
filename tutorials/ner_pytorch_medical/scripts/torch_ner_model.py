from collections import OrderedDict
from typing import Optional, List
from thinc.api import (
    with_array,
    chain,
    Model,
    PyTorchWrapper,
    PyTorchLSTM,
)
from thinc.types import Floats2d

from spacy.tokens import Doc
from spacy.util import registry
import torch
from torch import nn


@registry.architectures("TorchEntityRecognizer.v1")
def build_torch_ner_model(
    tok2vec: Model[List[Doc], List[Floats2d]],
    hidden_width: int,
    dropout: Optional[float] = None,
    nO: Optional[int] = None,
) -> Model[List[Doc], List[Floats2d]]:
    """Build a tagger model, using a provided token-to-vector component. The tagger
    model simply adds a linear layer with softmax activation to predict scores
    given the token vectors.
    tok2vec (Model[List[Doc], List[Floats2d]]): The token-to-vector subnetwork.
    nO (int or None): The number of tags to output. Inferred from the data if None.
    RETURNS (Model[List[Doc], List[Floats2d]]): Initialized Model
    """
    t2v_width = tok2vec.maybe_get_dim("nO")
    torch_model = TorchEntityRecognizer(t2v_width, hidden_width, nO, dropout)
    wrapped_pt_model = PyTorchWrapper(torch_model)
    wrapped_pt_model.attrs["set_dropout_rate"] = torch_model.set_dropout_rate

    model = chain(tok2vec, with_array(wrapped_pt_model))
    model.set_ref("tok2vec", tok2vec)
    model.set_ref("torch_model", wrapped_pt_model)
    model.init = init
    return model


def init(
    model: Model[List[Doc], Floats2d],
    X: Optional[List[Doc]] = None,
    Y: Optional[List[str]] = None,
) -> Model[List[Doc], List[Floats2d]]:
    """Dynamically set PyTorch Output Layer shape based on labels data
    model (Model[List[Doc], Floats2d]): Thinc Model wrapping tok2vec and PyTorch model
    X (Optional[List[Doc]], optional): Sample of Doc objects.
    Y (Optional[List[Ints2d]], optional): Available model labels.
    RETURNS (Model[List[Doc], List[Floats2d]]): Initialized Model
    """

    tok2vec = model.get_ref("tok2vec")
    torch_model = model.get_ref("torch_model")

    listener = tok2vec.maybe_get_ref("listener")
    t2v_width = listener.maybe_get_dim("nO") if listener else None
    if t2v_width:
        torch_model.shims[0]._model.set_input_shape(t2v_width)
        torch_model.set_dim("nI", t2v_width)

    if Y is not None:
        nO = len(Y)
        torch_model.shims[0]._model.set_output_shape(nO)
        torch_model.set_dim("nO", nO)

    tok2vec = model.get_ref("tok2vec")
    tok2vec.initialize()
    return model


def is_dropout_module(
    module: nn.Module,
    dropout_modules: List[nn.Module] = [nn.Dropout, nn.Dropout2d, nn.Dropout3d],
) -> bool:
    """Detect if a PyTorch Module is a Dropout layer
    module (nn.Module): Module to check
    dropout_modules (List[nn.Module], optional): List of Modules that count as Dropout layers.
    RETURNS (bool): True if module is a Dropout layer.
    """
    for m in dropout_modules:
        if isinstance(module, m):
            return True
    return False


class TorchEntityRecognizer(nn.Module):
    """Torch Entity Recognizer Model Head"""

    def __init__(self, nI: int, nH: int, nO: int, dropout: float):
        """Initialize TorchEntityRecognizer.
        nI (int): Input Dimension
        nH (int): Hidden Dimension Width
        nO (int): Output Dimension Width
        dropout (float): Dropout ratio (0 - 1.0)
        """
        super(TorchEntityRecognizer, self).__init__()

        # Just for initialization of PyTorch layer. Output shape set during Model.init
        nI = nI or 1
        nO = nO or 1

        self.nH = nH
        self.model = nn.Sequential(
            OrderedDict(
                {
                    "input_layer": nn.Linear(nI, nH),
                    "input_activation": nn.ReLU(),
                    "input_dropout": nn.Dropout2d(dropout),
                    "output_layer": nn.Linear(nH, nO),
                    "output_dropout": nn.Dropout2d(dropout),
                    "softmax": nn.Softmax(dim=1),
                }
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        inputs (torch.Tensor): Batch of outputs from spaCy tok2vec layer
        RETURNS (torch.Tensor): Batch of results with a score for each tag for each token
        """
        return self.model(inputs)

    def _set_layer_shape(self, name: str, nI: int, nO: int):
        """Dynamically set the shape of a layer
        name (str): Layer name
        nI (int): New input shape
        nO (int): New output shape
        """
        with torch.no_grad():
            layer = getattr(self.model, name)
            layer.out_features = nO
            layer.weight = nn.Parameter(torch.Tensor(nO, nI))
            if layer.bias is not None:
                layer.bias = nn.Parameter(torch.Tensor(nO))
            layer.reset_parameters()

    def set_input_shape(self, nI: int):
        """Dynamically set the shape of the input layer
        nI (int): New input layer shape
        """
        self._set_layer_shape("input_layer", nI, self.nH)

    def set_output_shape(self, nO: int):
        """Dynamically set the shape of the output layer
        nO (int): New output layer shape
        """
        self._set_layer_shape("output_layer", self.nH, nO)

    def set_dropout_rate(self, dropout: float):
        """Set the dropout rate of all Dropout layers in the model.
        dropout (float): Dropout rate to set
        """
        dropout_layers = [
            module for module in self.modules() if is_dropout_module(module)
        ]
        for layer in dropout_layers:
            layer.p = dropout

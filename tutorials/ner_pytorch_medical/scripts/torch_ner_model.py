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


@registry.architectures("TorchBiLSTMEncoder.v1")
def BiLSTMEncoder(
    width: int, depth: int, dropout: float
) -> Model[List[Floats2d], List[Floats2d]]:
    """Encode context using bidirectonal LSTM layers. Requires PyTorch.
    width (int): The input and output width. These are required to be the same,
        to allow residual connections. This value will be determined by the
        width of the inputs. Recommended values are between 64 and 300.
    depth (int): The number of recurrent layers.
    dropout (float): Creates a Dropout layer on the outputs of each LSTM layer
        except the last layer. Set to 0 to disable this functionality.
    """
    return PyTorchLSTM(width, width, bi=True, depth=depth, dropout=dropout)


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
    t2v_width = tok2vec.get_dim("nO") if tok2vec.has_dim("nO") else 768
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
    if Y is not None:
        nO = len(Y)
        torch_model = model.get_ref("torch_model")
        torch_model.shims[0]._model.set_output_shape(nO)

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
        if not nO:
            nO = 1  # Just for initialization of PyTorch layer. Output shape set during Model.init

        self.nH = nH
        self.model = nn.Sequential(nn.Linear(nI, nH), nn.ReLU(), nn.Dropout2d(dropout))
        self.output_layer = nn.Linear(nH, nO or 1)
        self.dropout = nn.Dropout2d(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        inputs (torch.Tensor): Batch of outputs from spaCy tok2vec layer
        RETURNS (torch.Tensor): Batch of results with a score for each tag for each token
        """
        x = self.model(inputs)
        x = self.output_layer(x)
        x = self.dropout(x)
        return self.softmax(x)

    def set_output_shape(self, nO: int):
        """Dynamically set the shape of the output layer
        nO (int): New output layer shape
        """
        with torch.no_grad():
            self.output_layer.out_features = nO
            self.output_layer.weight = nn.Parameter(torch.Tensor(nO, self.nH))
            if self.output_layer.bias is not None:
                self.output_layer.bias = nn.Parameter(torch.Tensor(nO))
            self.output_layer.reset_parameters()

    def set_dropout_rate(self, dropout: float):
        """Set the dropout rate of all Dropout layers in the model.
        dropout (float): Dropout rate to set
        """
        dropout_layers = [
            module for module in self.modules() if is_dropout_module(module)
        ]
        for layer in dropout_layers:
            layer.p = dropout

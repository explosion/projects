from typing import List

import torch
import torch.nn as nn


class TorchEntityRecognizer(nn.Module):
    """Torch Entity Recognizer Model Head"""

    def __init__(self, nI: int, nH: int, nO: int, dropout: float):
        """Initialize TorchEntityRecognizer.

        Args:
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

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = self.output_layer(x)
        x = self.dropout(x)
        return self.softmax(x)

    def set_output_shape(self, nO: int):
        """Dynamically set the shape of the output layer

        Args:
            nO (int): New output layer shape
        """
        with torch.no_grad():
            self.output_layer.out_features = nO
            self.output_layer.weight = nn.Parameter(torch.Tensor(nO, self.nH))
            if self.output_layer.bias:
                self.output_layer.bias = nn.Parameter(torch.Tensor(nO))
            self.output_layer.reset_parameters()


if __name__ == "__main__":

    torch_model = TorchEntityRecognizer(96, 48, 10, 0.1)

    print(torch_model)

    def is_dropout_module(module: nn.Module, dropout_modules: List[nn.Module] = [nn.Dropout, nn.Dropout2d, nn.Dropout3d]):
        for m in dropout_modules:
            if isinstance(module, m):
                return True
        return False


    dropout_layers = [module for module in torch_model.modules() if is_dropout_module(module)]

    print(dropout_layers[0].p)
    dropout_layers[0].p = 0.5
    print(dropout_layers[0])

    # print(dropout_layers)


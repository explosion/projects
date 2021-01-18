from thinc.api import PyTorchWrapper, with_padded, reduce_last, with_array
from thinc.types import Ragged, Padded
from spacy.util import registry
import torch.nn

# TODO: This is unfinished (and untested), I need to run it on a different 
# machine. I've probably messed up the input/output conversions.


@registry.architectures("spacy.reduce_MHA.v1")
def reduce_MHA(embed_dim: int, num_heads: int) -> Model[Ragged, Floats2d]:
    mha: Model[Floats3d, Floats3d] = PyTorchWrapper(
        torch.nn.MultiheadAttention(embed_dim, num_heads),
        convert_inputs=get_query_key_value,
        convert_outputs=get_output
    )

    return chain(
        with_padded(with_array(mha)),
        reduce_last()
    )


def get_query_key_value(model, X: Floats3d, is_train):
    Xtorch = xp2torch(X, needs_grad=True)
    torch_inputs = (Xtorch, Xtorch, Xtorch)

    def reverse_conversion(d_inputs):
        d_torch_inputs = d_inputs.args[0]
        # If the query, keys and values were all the same, the gradient of dX
        # will be the sum of their gradients.
        dX = torch2xp(d_torch_inputs[0])
        dX += torch2xp(d_torch_inputs[1])
        dX += torch2xp(d_torch_inputs[2])
        return dX

    return ArgsKwargs(args=torch_inputs), reverse_conversion


def get_output(model, inputs_outputs, is_train):
    X, (Ytorch, attn_torch) = inputs_outputs
    Y = torch2xp(Ytorch)

    def reverse_conversion(dY):
        dYtorch = xp2torch(dY)
        return ArgsKwargs(args=(dYtorch,), kwargs={"grad_tensors": dYtorch})

    return Y, reverse_conversion

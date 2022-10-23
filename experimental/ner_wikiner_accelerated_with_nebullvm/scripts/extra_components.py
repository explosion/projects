import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List, Dict

import torch.cuda
from nebullvm.api.functions import optimize_model
from nebullvm.inference_learners.base import LearnerMetadata, \
    BaseInferenceLearner
from spacy.tokens import Doc
from spacy_transformers import TransformerModel
from spacy_transformers.align import get_alignment
from spacy_transformers.data_classes import FullTransformerBatch, WordpieceBatch
from spacy_transformers.layers.transformer_model import huggingface_tokenize
from spacy_transformers.truncate import truncate_oversize_splits
from spacy_transformers.util import registry, maybe_flush_pytorch_cache, \
    log_gpu_memory, log_batch_size
from thinc.api import Model
from thinc.model import OutT, InT


class _ModelWrapper:
    def __init__(self, model: BaseInferenceLearner):
        self.model = model

    def train(self):
        return

    def eval(self):
        return

    @property
    def device(self):
        return self.model.device

    def __call__(self, *args, **kwargs):
        if "input_ids" in kwargs and kwargs["input_ids"].shape[0] != 1:
            out = {}
            out_type = None
            for i in range(len(kwargs["input_ids"])):
                temp_kwargs = {key: value[i].unsqueeze(0).long() for key, value in kwargs.items()}
                temp_out = self.model(**temp_kwargs)
                if out_type is None:
                    out_type = type(temp_out)
                for key, arg in temp_out.items():
                    if key in out:
                        out[key] = torch.cat([out[key], temp_out[key]])
                    else:
                        out[key] = temp_out[key]
            out = out_type(**out)
        else:
            kwargs = {key: value.long() for key, value in kwargs.items()}
            out = self.model(*args, **kwargs)
        for key, value in out.items():
            setattr(out, key, value.float())
        return out


def _patch_nebullvm_model(model):
    model.device = "cuda" if torch.cuda.is_available() else "cpu"
    return _ModelWrapper(model)


class NebullvmTransformerModel(TransformerModel):
    _nebullvm_layer = None

    def optimize(self, input_data, **kwargs):
        tokenizer = self.layers[0].shims[0]._hfmodel.tokenizer
        base_kwargs = dict(
            metric="numeric_precision",
            metric_drop_ths=0.1,
            optimization_time="constrained",
            tokenizer=tokenizer,
            store_latencies=True,
            ignore_compilers=["tensor RT"],
            tokenizer_args=dict(
                add_special_tokens=True,
                return_attention_mask=True,
                # return_offsets_mapping=isinstance(
                #     tokenizer,
                #     PreTrainedTokenizerFast
                # ),
                return_tensors="pt",
                return_token_type_ids=None,  # Sets to model default
                padding="longest",
                truncation=True,
            )
        )
        base_kwargs.update(kwargs)
        model = self.transformer
        if torch.cuda.is_available():
            model.cuda()
        optimized_model = optimize_model(
            model=model,
            input_data=input_data,
            **base_kwargs,
        )
        self._nebullvm_layer = copy.deepcopy(self.layers[0])
        self._nebullvm_layer.shims[0]._hfmodel.transformer = optimized_model

    def predict(self, X: InT) -> OutT:
        if self._nebullvm_layer is None:
            return super().predict(X)
        else:
            return nebullvm_forward(self, X, False)

    def to_dict(self) -> Dict:
        msg = super().to_dict()
        if self._nebullvm_layer is not None:
            optimized_model = self._nebullvm_layer.shims[0]._hfmodel.transformer
            nebullvm_dict = {}
            with TemporaryDirectory() as tmp_dir:
                tmp_dir = Path(tmp_dir)
                optimized_model.save(tmp_dir)
                files = list(tmp_dir.glob("**/*"))
                for file in files:
                    if file.is_dir():
                        continue
                    file_name = str(file.relative_to(tmp_dir))
                    with open(file, "rb") as f:
                        nebullvm_dict[file_name] = f.read()
            msg["_nebullvm_layer"] = nebullvm_dict
        return msg

    def from_dict(self, msg: Dict) -> "Model":
        optimized_model = None
        if "_nebullvm_layer" in msg:
            nebullvm_dict = msg.pop("_nebullvm_layer")
            with TemporaryDirectory() as tmp_dir:
                tmp_dir = Path(tmp_dir)
                for filename, bytefile in nebullvm_dict.items():
                    file_path = tmp_dir / filename
                    file_path.parent.mkdir(exist_ok=True, parents=True)
                    with open(file_path, "wb") as f:
                        f.write(bytefile)
                optimized_model = LearnerMetadata.read(tmp_dir).load_model(tmp_dir)
                wrapped_model = _patch_nebullvm_model(optimized_model)
                # warmup
                # _ = optimized_model.predict(*optimized_model.get_inputs_example())
        super().from_dict(msg)
        if optimized_model is not None:
            self._nebullvm_layer = copy.deepcopy(self.layers[0])
            self._nebullvm_layer.shims[0]._hfmodel.transformer = optimized_model
            self._nebullvm_layer.shims[0]._model = wrapped_model
            self._nebullvm_layer.shims[0]._mixed_precision = False
        return self

    # def from_disk(self, path: Union[Path, str]) -> "Model":
    #     print(f"Loading from disk with path: {path}")
    #     path = Path(path) if isinstance(path, str) else path
    #     with path.open("rb") as file_:
    #         bytes_data = file_.read()
    #     self.from_bytes(bytes_data)
    #     nebullvm_path = Path(path).parent / "nebullvm"
    #     if nebullvm_path.exists():
    #         print("Nebullvm loaded")
    #         optimized_model = LearnerMetadata.read(path).load_model(path)
    #         # warmup
    #         _ = optimized_model(*optimized_model.get_inputs_example())
    #         self._nebullvm_layer = copy.deepcopy(self.layers[0])
    #         self._nebullvm_layer.shims[0]._hfmodel.transformer = optimized_model
    #     return self

    def copy(self):
        """
        Create a copy of the model, its attributes, and its parameters. Any child
        layers will also be deep-copied. The copy will receive a distinct `model.id`
        value.
        """
        copied = NebullvmTransformerModel(self.name, self.attrs["get_spans"])
        params = {}
        for name in self.param_names:
            params[name] = self.get_param(name) if self.has_param(name) else None
        copied.params = copy.deepcopy(params)
        copied.dims = copy.deepcopy(self._dims)
        copied.layers[0] = copy.deepcopy(self.layers[0])
        copied._nebullvm_layer = self._nebullvm_layer
        # nebullvm layer is not copied since it is unmodifiable
        for name in self.grad_names:
            copied.set_grad(name, self.get_grad(name).copy())
        return copied


@registry.architectures.register("NebullvmTransformerModel.v1")
def create_NebullvmTransformerModel_v1(
    name: str,
    get_spans: Callable,
    tokenizer_config: dict = {},
    transformer_config: dict = {},
    mixed_precision: bool = False,
    grad_scaler_config: dict = {},
) -> Model[List[Doc], FullTransformerBatch]:
    """Pretrained transformer model that can be finetuned for downstream tasks.
    name (str): Name of the pretrained Huggingface model to use.
    get_spans (Callable[[List[Doc]], List[List[Span]]]): A function to extract
        spans from the batch of Doc objects. See the "TransformerModel" layer
        for details.
    tokenizer_config (dict): Settings to pass to the transformers tokenizer.
    transformers_config (dict): Settings to pass to the transformers forward pass
        of the transformer.
    mixed_precision (bool): Enable mixed-precision. Mixed-precision replaces
        whitelisted ops to half-precision counterparts. This speeds up training
        and prediction on modern GPUs and reduces GPU memory use.
    grad_scaler_config (dict): Configuration for gradient scaling in mixed-precision
        training. Gradient scaling is enabled automatically when mixed-precision
        training is used.
        Setting `enabled` to `False` in the gradient scaling configuration disables
        gradient scaling. The `init_scale` (default: `2 ** 16`) determines the
        initial scale. `backoff_factor` (default: `0.5`) specifies the factor
        by which the scale should be reduced when gradients overflow.
        `growth_interval` (default: `2000`) configures the number of steps
        without gradient overflows after which the scale should be increased.
        Finally, `growth_factor` (default: `2.0`) determines the factor by which
        the scale should be increased when no overflows were found for
        `growth_interval` steps.
    """
    model = NebullvmTransformerModel(
        name,
        get_spans,
        tokenizer_config,
        transformer_config,
        mixed_precision,
        grad_scaler_config,
    )
    return model


def nebullvm_forward(
    model: NebullvmTransformerModel, docs: List[Doc], is_train: bool
) -> FullTransformerBatch:
    tokenizer = model.tokenizer
    get_spans = model.attrs["get_spans"]
    transformer = model._nebullvm_layer

    nested_spans = get_spans(docs)
    flat_spans = []
    for doc_spans in nested_spans:
        flat_spans.extend(doc_spans)
    # Flush the PyTorch cache every so often. It seems to help with memory :(
    # This shouldn't be necessary, I'm not sure what I'm doing wrong?
    maybe_flush_pytorch_cache(chance=model.attrs.get("flush_cache_chance", 0))
    if "logger" in model.attrs:
        log_gpu_memory(model.attrs["logger"], "begin forward")
    batch_encoding = huggingface_tokenize(tokenizer, [span.text for span in flat_spans])
    wordpieces = WordpieceBatch.from_batch_encoding(batch_encoding)
    if "logger" in model.attrs:
        log_batch_size(model.attrs["logger"], wordpieces, is_train)
    align = get_alignment(flat_spans, wordpieces.strings, tokenizer.all_special_tokens)
    wordpieces, align = truncate_oversize_splits(
        wordpieces, align, tokenizer.model_max_length
    )
    model_output, bp_tensors = transformer(wordpieces, is_train)
    if "logger" in model.attrs:
        log_gpu_memory(model.attrs["logger"], "after forward")
    output = FullTransformerBatch(
        spans=nested_spans,
        wordpieces=wordpieces,
        model_output=model_output,
        align=align,
    )
    if "logger" in model.attrs:
        log_gpu_memory(model.attrs["logger"], "return from forward")

    return output


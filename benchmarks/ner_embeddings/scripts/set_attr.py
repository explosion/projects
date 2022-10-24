import srsly
import spacy

from pathlib import Path
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from typing import Optional, Callable


@spacy.registry.callbacks("set_attr")
def create_callback(
    path: Path,
    component: str,
    attr: str,
    layer: Optional[str],
) -> Callable[[Language], Language]:
    """
    Should be set as a callback of [initialize.before_init].
    You need to set the right ref in your model when you create it.
    This is useful when you have some layer that requires a data
    file from disk. The value will only be loaded during the '
    initialize' step before training.
    After training the attribute value will be serialized into the model,
    and then during deserialization it's loaded
    back in with the model data.
    """
    attr_value = srsly.read_msgpack(path)

    def set_attr(nlp: Language) -> Language:
        if not nlp.has_pipe(component):
            raise ValueError("Trying to set attribute for non-existing component")
        pipe: TrainablePipe = nlp.get_pipe(component)
        model = None
        if pipe.model.has_ref(layer):
            model = pipe.model.get_ref(layer)
        else:
            for lay in list(pipe.model.walk()):
                if lay.name == layer:
                    if model is not None:
                        raise ValueError(f"Found multiple layers named {layer}")
                    else:
                        model = lay
        if model is None:
            raise ValueError(f"Haven't found {layer} in component {component}.")
        model.attrs[attr] = attr_value
        return nlp

    return set_attr

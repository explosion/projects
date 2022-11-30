from pathlib import Path
from wasabi import msg
import re

import spacy
from spacy.language import Language

# These are the architectures that are recognized as tok2vec/feature sources.
TOK2VEC_ARCHS = [("spacy", "Tok2Vec"), ("spacy-transformers", "TransformerModel")]
# These are the listeners.
LISTENER_ARCHS = [
    ("spacy", "Tok2VecListener"),
    ("spacy-transformers", "TransformerListener"),
]


def deep_get(obj, key, default):
    """Given a multi-part key, try to get the key. If at any point this isn't possible, return the default."""
    out = None
    slot = obj
    for notch in key:
        if slot is None or notch not in slot:
            return default
        slot = slot[notch]
    return slot


def get_tok2vecs(config):
    """Given a pipeline config, return the names of components that are
    tok2vecs (or Transformers).
    """
    out = []
    for name, comp in config["components"].items():
        arch = deep_get(comp, ("model", "@architectures"), False)
        if not arch:
            continue

        ns, model, ver = arch.split(".")
        if (ns, model) in TOK2VEC_ARCHS:
            out.append(name)
    return out


def has_listener(nlp, pipe_name):
    """Given a pipeline and a component name, check if it has a listener."""
    arch = deep_get(
        nlp.config,
        ("components", pipe_name, "model", "tok2vec", "@architectures"),
        False,
    )
    if not arch:
        return False
    ns, model, ver = arch.split(".")
    return (ns, model) in LISTENER_ARCHS


def get_listeners(nlp):
    """Get the name of every component that contains a listener.

    Does not check that they listen to the same thing; assumes a pipeline has
    only one feature source.
    """
    out = []
    for name in nlp.pipe_names:
        if has_listener(nlp, name):
            out.append(name)
    return out


def increment_suffix(name):
    """Given a name, return an incremented version.

    If no numeric suffix is found, return the original with "2" appended.

    This is used to avoid name collisions in pipelines.
    """

    res = re.search("\d+$", name)
    if res is None:
        return f"{name}2"
    else:
        num = res.match
        prefix = name[0 : -len(num)]
        return f"{prefix}{int(num) + 1}"


def check_tok2vecs(name, config):
    """Check if there are any issues with tok2vecs in a pipeline.

    Currently just checks there isn't more than one.
    """
    tok2vecs = get_tok2vecs(config)
    fail_msg = f"""
        Can't handle pipelines with more than one feature source, 
        but {name} has {len(tok2vecs)}."""
    if len(tok2vecs) > 1:
        msg.fail(fail_msg, exits=1)


def check_pipeline_names(nlp, nlp2):

    fail_msg = """
        Tried automatically renaming {name}, but still had a collision, so
        bailing out. Please make your pipe names more unique.
        """

    # map of components to be renamed
    rename = {}
    # check pipeline names
    names = nlp.pipe_names
    for name in nlp2.pipe_names:
        if name in names:
            inc = increment_suffix(name)
            # TODO Would it be better to just keep incrementing?
            if inc in names or inc in nlp2.pipe_names:
                msg.fail(fail_msg.format(name=name), exits=1)
            rename[name] = inc
    return rename

from functools import partial
from pathlib import Path
from typing import Iterable, Callable
import spacy
from spacy.training import Example
from spacy.tokens import Doc, DocBin
from spacy.language import Language


@spacy.registry.readers("HeadCopyingCorpus.v1")
def create_head_copy_docbin_reader(
    path: Path, head_prefix
) -> Callable[[Language], Iterable[Example]]:
    return partial(copy_gold_heads, path, head_prefix)


def copy_gold_heads(path: Path, head_prefix: str, nlp: Language) -> Iterable[Example]:
    """
    Copy gold heads from reference to predicted documents so that the span resolver
    can predict spans.
    """
    doc_bin = DocBin().from_disk(path)
    docs = doc_bin.get_docs(nlp.vocab)
    for doc in docs:
        pred = Doc(
            nlp.vocab,
            words=[word.text for word in doc],
            spaces=[bool(word.whitespace_) for word in doc],
        )
        for name, span_group in doc.spans.items():
            if name.startswith(head_prefix):
                pred.spans[name] = [pred[span.start : span.end] for span in span_group]
        yield Example(predicted=pred, reference=doc)

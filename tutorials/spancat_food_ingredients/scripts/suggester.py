from spacy import registry
from spacy.pipeline.spancat import build_ngram_suggester

@registry.misc("custom_suggester")
def custom_ngram_suggester():
    return build_ngram_suggester(sizes=[1, 2, 3])  # all ngrams of size 1, 2 and 3
import time

import numpy as np
from spacy.training import Corpus
from spacy.util import load_model

from extra_components import *

if __name__ == "__main__":
    nlp = load_model("../training/model-best")
    corpus = Corpus("../corpus/dev.spacy")
    corpus_list = [x.text for x in corpus(nlp)]
    print(next(nlp.components[0][1].model._nebullvm_layer.shims[0]._hfmodel.transformer.core_inference_learner.model.parameters()).dtype)
    # Warmup
    _ = nlp(corpus_list[0])
    print("Optimized pipeline latency:")
    times = []
    for _ in range(1):
        for text in corpus_list:
            st = time.time()
            outputs = nlp(text)
            ot = time.time()-st
            times.append(ot)
    print(np.mean(times))

    print("Original pipeline latency")
    print(nlp.components[0][1].model)
    print(dir(nlp.components[0][1]))
    nlp.components[0][1].model._nebullvm_layer = None
    times = []
    for _ in range(1):
        for text in corpus_list:
            st = time.time()
            _ = nlp(text)
            ot = time.time()-st
            times.append(ot)
    print(np.mean(times))




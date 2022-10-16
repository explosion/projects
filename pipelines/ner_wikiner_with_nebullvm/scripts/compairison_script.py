import time

import numpy as np
from spacy.training import Corpus
from spacy.util import load_model

from extra_components import *


if __name__ == "__main__":
    if torch.cuda.is_available():
        from thinc.api import set_gpu_allocator, require_gpu

        # Use the GPU, with memory allocations directed via PyTorch.
        # This prevents out-of-memory errors that would otherwise occur from competing
        # memory pools.
        set_gpu_allocator("pytorch")
        require_gpu(0)
    nlp = load_model("../training/model-best")
    corpus = Corpus("../corpus/dev.spacy")
    corpus_list = [x.text for x in corpus(nlp)]
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
    nlp.components[0][1].model._nebullvm_layer = None
    times = []
    for _ in range(1):
        for text in corpus_list:
            st = time.time()
            _ = nlp(text)
            ot = time.time()-st
            times.append(ot)
    print(np.mean(times))




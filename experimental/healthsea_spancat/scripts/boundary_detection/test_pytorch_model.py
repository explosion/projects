import model
from thinc.api import Config, Model, get_current_ops, set_dropout_rate, Ops
from spacy.util import registry
from wasabi import msg
import spacy
import numpy as np

import pytorch_model

default_config = """
[model]
@architectures = "spacy.PyTorchSpanBoundaryDetection.v1"
window_size = 1

[model.tok2vec]
@architectures = "spacy.Tok2Vec.v1"
[model.tok2vec.embed]
@architectures = "spacy.MultiHashEmbed.v1"
width = 2
rows = [5000, 2000, 1000, 1000]
attrs = ["ORTH", "PREFIX", "SUFFIX", "SHAPE"]
include_static_vectors = false
[model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v1"
width = ${model.tok2vec.embed.width}
window_size = 1
maxout_pieces = 3
depth = 4

[optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001
learn_rate = 0.001
"""

nlp = spacy.blank("en")
doc = nlp("This is an example.")
doc2 = nlp("This is a second example.")

doc.spans["spans"] = [doc[3:4]]
doc2.spans["spans"] = [doc[3:5]]

print(doc.spans)
print(doc2.spans)

docs = [doc, doc2]

msg.info("Example docs constructed")

config = Config().from_str(default_config).interpolate()
model = registry.resolve(config)["model"]
optimizer = registry.resolve(config)["optimizer"]

#token_list = []
#for doc in docs:
#    for token in doc:
#        token_list.append(token)

#for token, prediction in zip(token_list,predictions):
#    print(token.text, prediction)

def get_reference(docs):
    reference_results = []
    for doc in docs:
        start_indices = []
        end_indices = []
        for span in doc.spans["spans"]:
            start_indices.append(span.start)
            end_indices.append(span.end)
        for token in doc:
            is_start = 0
            is_end = 0
            if token.i in start_indices:
                is_start = 1
            if token.i in end_indices:
                is_end = 1
            reference_results.append(np.array([is_start,is_end],dtype="float32"))

    reference_results = np.array(reference_results)
    return np.array(reference_results)

def get_loss(scores, docs):
    reference_results = []
    for doc in docs:
        start_indices = []
        end_indices = []
        for span in doc.spans["spans"]:
            start_indices.append(span.start)
            end_indices.append(span.end)
        for token in doc:
            is_start = 0
            is_end = 0
            if token.i in start_indices:
                is_start = 1
            if token.i in end_indices:
                is_end = 1
            reference_results.append(np.array([is_start,is_end],dtype="float32"))

    reference_results = get_reference(docs)
    d_scores = np.array(scores) - np.array(reference_results)
    loss = float((d_scores ** 2).sum())
    return loss, np.array(d_scores)

msg.info("Model config loaded")

Y = get_reference(docs)

model.initialize(X=docs[:1], Y=Y)

msg.info("Model initialized")

#msg.info("Start training")
#for i in range(10):
 #   msg.info(f"Iteration {i}")
 #   scores, backprop_scores = model.begin_update(docs)
 #   loss, d_scores = get_loss(scores,docs)
 #   backprop_scores(d_scores)
 #   model.finish_update(optimizer)
#    print(loss)

    #for doc, prediction in zip(docs,predictions):
    #    for token, vector in zip(doc, prediction):
    #        print(token.text, vector)

msg.info("Prediction started")
predictions = model.predict(docs)
msg.info("Prediction ended")


print(predictions)

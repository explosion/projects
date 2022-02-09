import model
from thinc.api import Config
from spacy.util import registry
from wasabi import msg
import spacy
from spacy.tokens import DocBin
import numpy as np
from spacy.scorer import PRFScore

import pytorch_model

default_config = """
[model]
@architectures = "spacy.PyTorchSpanBoundaryDetection.v1"
hidden_size = 128

[model.scorer]
@layers = "spacy.LinearLogistic.v1"
nO = 2

[model.tok2vec]
@architectures = "spacy.Tok2Vec.v1"
[model.tok2vec.embed]
@architectures = "spacy.MultiHashEmbed.v1"
width = 96
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

# Functions
def get_reference(docs, spankey):
    reference_results = []
    for doc in docs:
        start_indices = []
        end_indices = []
        for span in doc.spans[spankey]:
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
    reference_results = get_reference(docs,spankey)
    d_scores = np.array(scores) - np.array(reference_results)
    loss = float((d_scores ** 2).sum())
    return loss, np.array(d_scores)

def eval(predictions, docs, threshold) -> PRFScore:
    reference_results = get_reference(docs,spankey)

    scorer_start = PRFScore()
    scorer_end = PRFScore()

    for prediction, reference in zip(predictions,reference_results):
        
        start_prediction = prediction[0]
        end_prediction = prediction[1]
        start_reference = reference[0]
        end_reference = reference[1]
        
        # Start
        if start_prediction >= threshold:
            start_prediction = 1
        else:
            start_prediction = 0

        if start_prediction == 1 and start_reference == 1:
            scorer_start.tp += 1
        elif start_prediction == 1 and start_reference == 0:
            scorer_start.fp += 1
        elif start_prediction == 0 and start_reference == 1:
            scorer_start.fn += 1

        # End
        if end_prediction >= threshold:
            end_prediction = 1
        else:
            end_prediction = 0

        if end_prediction == 1 and end_reference == 1:
            scorer_end.tp += 1
        elif end_prediction == 1 and end_reference == 0:
            scorer_end.fp += 1
        elif end_prediction == 0 and end_reference == 1:
            scorer_end.fn += 1
    
    return scorer_start,scorer_end

# Config
iterations = 100
eval_frequency = 5
threshold = 0.5
doc_subset = 200
spankey = "health_aspects"

# Load Data
msg.info("Loading data")
nlp = spacy.blank("en")
train_path = "./train.spacy"
dev_path = "./dev.spacy"


train_docBin = DocBin().from_disk(train_path)
dev_docBin = DocBin().from_disk(dev_path)

train_docs = list(train_docBin.get_docs(nlp.vocab))
dev_docs = list(dev_docBin.get_docs(nlp.vocab))
msg.good("Data loaded")

msg.info("Inititalize model")
config = Config().from_str(default_config).interpolate()
model = registry.resolve(config)["model"]
optimizer = registry.resolve(config)["optimizer"]

Y = get_reference(train_docs[:doc_subset],spankey)
model.initialize(X=train_docs[:10], Y=Y)

msg.good("Model initialized")

# Training
msg.info("Start training")
for i in range(iterations):
    scores, backprop_scores = model.begin_update(train_docs[:doc_subset])
    loss, d_scores = get_loss(scores,train_docs[:doc_subset])
    backprop_scores(d_scores)
    model.finish_update(optimizer)
    print(f"{i} LOSS: {loss}")
    if iterations%eval_frequency == 0:
        predictions = model.predict(dev_docs[:doc_subset])
        start_scores, end_scores = eval(predictions,dev_docs[:doc_subset],threshold)
        print(f"(START TOKEN) EVAL: F-Score: {start_scores.fscore} Precision: {start_scores.precision} Recall: {start_scores.recall}")
        print(f"  (END TOKEN) EVAL: F-Score: {end_scores.fscore} Precision: {end_scores.precision} Recall: {end_scores.recall}")

msg.good("Training ended")

#Evaluation
msg.info("Start evaluation")
predictions = model.predict(dev_docs[:doc_subset])
start_scores, end_scores = eval(predictions,dev_docs[:doc_subset],threshold)
print(f"(START TOKEN) EVAL: F-Score: {start_scores.fscore} Precision: {start_scores.precision} Recall: {start_scores.recall}")
print(f"  (END TOKEN) EVAL: F-Score: {end_scores.fscore} Precision: {end_scores.precision} Recall: {end_scores.recall}")

msg.info("Evaluation ended")


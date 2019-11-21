"""Scripts used for training and evaluation of NER models

Usage example:
$ python scripts.py train ./model ./train.jsonl ./eval.jsonl --tok2vec tok2vec.bin

Requirements:
spacy>=2.2.3
"""
import spacy
from spacy.cli.train import _load_pretrained_tok2vec
from timeit import default_timer as timer
from pathlib import Path
import srsly
from wasabi import msg
import random
import plac
import sys
import tqdm


def format_data(data):
    result = []
    labels = set()
    for eg in data:
        if eg["answer"] != "accept":
            continue
        ents = [(s["start"], s["end"], s["label"]) for s in eg.get("spans", [])]
        labels.update([ent[2] for ent in ents])
        result.append((eg["text"], {"entities": ents}))
    return result, labels


@plac.annotations(
    model=("The base model to load or blank:lang", "positional", None, str),
    train_path=("The training data (Prodigy JSONL)", "positional", None, str),
    eval_path=("The evaluation data (Prodigy JSONL)", "positional", None, str),
    n_iter=("Number of iterations", "option", "n", int),
    output=("Optional output directory", "option", "o", str),
    tok2vec=("Pretrained tok2vec weights to initialize model", "option", "t2v", str),
)
def train_model(
    model, train_path, eval_path, n_iter=10, output=None, tok2vec=None,
):
    """
    Train a model from Prodigy annotations and optionally save out the best
    model to disk.
    """
    spacy.util.fix_random_seed(0)
    with msg.loading(f"Loading '{model}'..."):
        if model.startswith("blank:"):
            nlp = spacy.blank(model.replace("blank:", ""))
        else:
            nlp = spacy.load(model)
    msg.good(f"Loaded model '{model}'")
    train_data, labels = format_data(srsly.read_jsonl(train_path))
    eval_data, _ = format_data(srsly.read_jsonl(eval_path))
    ner = nlp.create_pipe("ner")
    for label in labels:
        ner.add_label(label)
    nlp.add_pipe(ner)
    t2v_cfg = {
        "embed_rows": 10000,
        "token_vector_width": 128,
        "conv_depth": 8,
        "nr_feature_tokens": 3,
    }
    optimizer = nlp.begin_training(component_cfg={"ner": t2v_cfg} if tok2vec else {})
    if tok2vec:
        _load_pretrained_tok2vec(nlp, Path(tok2vec))
    batch_size = spacy.util.compounding(1.0, 16.0, 1.001)
    best_acc = 0
    best_model = None
    row_widths = (2, 8, 8, 8, 8)
    msg.row(("#", "L", "P", "R", "F"), widths=row_widths)
    for i in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        data = tqdm.tqdm(train_data, leave=False)
        for batch in spacy.util.minibatch(data, size=batch_size):
            texts, annots = zip(*batch)
            nlp.update(texts, annots, drop=0.2, losses=losses)
        with nlp.use_params(optimizer.averages):
            sc = nlp.evaluate(eval_data)
            if sc.ents_f > best_acc:
                best_acc = sc.ents_f
                if output:
                    best_model = nlp.to_bytes()
        acc = (f"{sc.ents_p:.3f}", f"{sc.ents_r:.3f}", f"{sc.ents_f:.3f}")
        msg.row((i + 1, f"{losses['ner']:.2f}", *acc), widths=row_widths)
    msg.text(f"Best F-Score: {best_acc:.3f}")
    if output and best_model:
        with msg.loading("Saving model..."):
            nlp.from_bytes(best_model).to_disk(output)
        msg.good("Saved model", output)


@plac.annotations(
    model=("The model to evaluate", "positional", None, str),
    eval_path=("The evaluation data (Prodigy JSONL)", "positional", None, str),
)
def evaluate_model(model, eval_path):
    """
    Evaluate a trained model on Prodigy annotations and print the accuracy.
    """
    with msg.loading(f"Loading model '{model}'..."):
        nlp = spacy.load(model)
    data, _ = format_data(srsly.read_jsonl(eval_path))
    sc = nlp.evaluate(data)
    result = [
        ("Precision", f"{sc.ents_p:.3f}"),
        ("Recall", f"{sc.ents_r:.3f}"),
        ("F-Score", f"{sc.ents_f:.3f}"),
    ]
    msg.table(result)


@plac.annotations(
    model=("The model to evaluate", "positional", None, str),
    data=("Raw data as JSONL", "positional", None, str),
)
def wps(model, data):
    """
    Measure the processing speed in words per second. It's recommended to
    use a larger corpus of raw text here (e.g. a few million words).
    """
    with msg.loading(f"Loading model '{model}'..."):
        nlp = spacy.load(model)
    texts = (eg["text"] for eg in srsly.read_jsonl(data))
    n_docs = 0
    n_words = 0
    start_time = timer()
    for doc in nlp.pipe(texts):
        n_docs += 1
        n_words += len(doc)
    end_time = timer()
    wps = int(n_words / (end_time - start_time))
    result = [
        ("Docs", f"{n_docs:,}"),
        ("Words", f"{n_words:,}"),
        ("Words/s", f"{wps:,}"),
    ]
    msg.table(result, widths=(7, 12), aligns=("l", "r"))


if __name__ == "__main__":
    opts = {"train": train_model, "evaluate": evaluate_model, "wps": wps}
    cmd = sys.argv.pop(1)
    if cmd not in opts:
        msg.fail(f"Unknown command: {cmd}", f"Available: {', '.join(opts)}", exits=1)
    try:
        plac.call(opts[cmd])
    except KeyboardInterrupt:
        msg.warn("Stopped.", exits=1)

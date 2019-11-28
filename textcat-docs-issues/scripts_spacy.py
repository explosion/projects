"""Scripts used for training and evaluation of binary text classification
models.

Usage example:
$ python scripts.py train en_vectors_web_lg ./train.jsonl ./eval.jsonl

Requirements:
spacy>=2.2.3
"""
import spacy
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
        if eg["answer"] == "ignore":
            continue
        cats = {eg["label"]: eg["answer"] == "accept"}
        labels.update(list(cats))
        result.append((eg["text"], {"cats": cats}))
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
    textcat = nlp.create_pipe("textcat")
    for label in labels:
        textcat.add_label(label)
    nlp.add_pipe(textcat)
    cfg = {"textcat": {"exclusive_classes": True}}
    optimizer = nlp.begin_training(component_cfg=cfg)
    batch_size = spacy.util.compounding(1.0, 16.0, 1.001)
    best_acc = 0
    best_model = None
    row_widths = (2, 8, 8)
    msg.row(("#", "L", "F"), widths=row_widths)
    for i in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        data = tqdm.tqdm(train_data, leave=False)
        for batch in spacy.util.minibatch(data, size=batch_size):
            texts, annots = zip(*batch)
            nlp.update(texts, annots, drop=0.2, losses=losses)
        with nlp.use_params(optimizer.averages):
            sc = nlp.evaluate(eval_data)
            if sc.textcat_score > best_acc:
                best_acc = sc.textcat_score
                if output:
                    best_model = nlp.to_bytes()
        acc = f"{sc.textcat_score:.3f}"
        msg.row((i + 1, f"{losses['textcat']:.2f}", acc), widths=row_widths)
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
    result = [("F-Score", f"{sc.textcat_score:.3f}")]
    msg.table(result)


if __name__ == "__main__":
    opts = {"train": train_model, "evaluate": evaluate_model}
    cmd = sys.argv.pop(1)
    if cmd not in opts:
        msg.fail(f"Unknown command: {cmd}", f"Available: {', '.join(opts)}", exits=1)
    try:
        plac.call(opts[cmd])
    except KeyboardInterrupt:
        msg.warn("Stopped.", exits=1)

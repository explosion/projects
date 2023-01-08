"""Use nlp.rehearse to rehearse the trained textcat_multilabel model on new data"""
import typer
import spacy
from spacy.tokens import DocBin, Doc
from spacy.training import Example
from pathlib import Path
from wasabi import msg


def rehearse_model(
    model_path: Path, train_input: Path, iterations: int, output_path: Path
):
    msg.info("Starting rehearse")
    nlp = spacy.load(model_path)
    msg.good("Model loaded")

    optimizer = nlp.resume_training()
    db = DocBin().from_disk(train_input)
    docs = list(db.get_docs(nlp.vocab))
    examples = []
    for doc in docs:
        words = [token.text for token in doc]
        spaces = [bool(token.whitespace_) for token in doc]
        predicted = Doc(nlp.vocab, words=words, spaces=spaces)
        reference = Doc(nlp.vocab, words=words, spaces=spaces)
        reference.cats = doc.cats
        examples.append(Example(predicted, reference))

    for i in range(iterations):
        losses = nlp.rehearse(examples, sgd=optimizer)
        msg.info(f"Iteration {i} / {losses}")

    nlp.to_disk(output_path)


if __name__ == "__main__":
    typer.run(rehearse_model)

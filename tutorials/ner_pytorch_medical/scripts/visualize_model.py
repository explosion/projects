import spacy_streamlit
import typer
from torch_ner_model import build_torch_ner_model
from torch_ner_pipe import make_torch_entity_recognizer


def main(models: str, default_text: str):
    models = [name.strip() for name in models.split(",")]
    labels = ["person", "problem", "pronoun", "test", "treatment"]
    spacy_streamlit.visualize(models, default_text, visualizers=["ner"], ner_labels=labels)


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        pass

import spacy_streamlit
import typer


def main(models: str, default_text: str):
    models = [name.strip() for name in models.split(",")]
    labels = ["PERSON", "ORG", "LOC", "GPE", "DRUG"]
    spacy_streamlit.visualize(
        models, default_text, visualizers=["ner"], ner_labels=labels
    )


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        pass

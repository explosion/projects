import typer
from spacy.lang.en import English
from pathlib import Path
from spacy.cli.train import train


def main():
    nlp = English()


if __name__ == "__main__":
    typer.run(main)

import typer
import stanza


SPACY_MODELS = ["en_core_web_md"]  # en_core_web_md
STANZA_MODELS = ["en_ewt"]
HF_TRF_MODELS = ["roberta-base"]


def main():
    for name in STANZA_MODELS:
        lang = name.split("_")[0]
        package = name.split("_")[1]
        stanza.download(lang, package=package)


if __name__ == "__main__":
    typer.run(main)
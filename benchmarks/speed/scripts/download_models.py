import typer
import stanza
from flair.models import MultiTagger


def main(stanza_models: str, flair_models):
    for name in stanza_models.split(","):
        lang = name.split("_")[0]
        package = name.split("_")[1]
        stanza.download(lang, package=package)
    for name in flair_models.split(","):
        annot_list = name.split("_")
        MultiTagger.load(annot_list)


if __name__ == "__main__":
    typer.run(main)

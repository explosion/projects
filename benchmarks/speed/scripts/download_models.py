import typer
import stanza


def main(stanza_models: str):
    for name in stanza_models.split(","):
        lang = name.split("_")[0]
        package = name.split("_")[1]
        stanza.download(lang, package=package)


if __name__ == "__main__":
    typer.run(main)
""" Parsing of Wiki dump and persisting of parsing results to DB. """
from wiki import wiki_dump_api
import typer


def main():
    """Parses Wikidata and Wikipedia dumps. Persists parsing results to DB."""
    wiki_dump_api.parse()


if __name__ == "__main__":
    typer.run(main)

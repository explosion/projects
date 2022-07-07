""" Parsing of Wiki dump and persisting of parsing results to DB. """
from typing import Optional
import typer
from wiki import wiki_dump_api


def main(
    entity_limit: Optional[int] = typer.Option(None, "--entity_limit"),
    article_limit: Optional[int] = typer.Option(None, "--article_limit"),
    alias_limit: Optional[int] = typer.Option(None, "--alias_limit"),
):
    """Parses Wikidata and Wikipedia dumps. Persists parsing results to DB.
    entity_limit (Optional[int]): Max. number of entities to parse. Unlimited if None.
    article_limit (Optional[int]): Max. number of entities to parse. Unlimited if None.
    alias_limit (Optional[int]): Max. number of entity aliases to parse. Unlimited if None.
    """

    wiki_dump_api.parse(
        entity_config={"limit": entity_limit},
        article_text_config={"limit": article_limit},
        alias_prior_prob_config={"limit": alias_limit},
    )


if __name__ == "__main__":
    typer.run(main)

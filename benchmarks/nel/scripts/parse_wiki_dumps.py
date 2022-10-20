""" Parsing of Wiki dump and persisting of parsing results to DB. """
from typing import Optional
import typer
from wiki import wiki_dump_api


def main(
    entity_limit: Optional[int] = typer.Option(None, "--entity_limit"),
    article_limit: Optional[int] = typer.Option(None, "--article_limit"),
    alias_limit: Optional[int] = typer.Option(None, "--alias_limit"),
    use_filtered_dumps: bool = typer.Option(False, "--use_filtered_dumps"),
):
    """Parses Wikidata and Wikipedia dumps. Persists parsing results to DB.
    entity_limit (Optional[int]): Max. number of entities to parse. Unlimited if None.
    article_limit (Optional[int]): Max. number of entities to parse. Unlimited if None.
    alias_limit (Optional[int]): Max. number of entity aliases to parse. Unlimited if None.
    use_filtered_dumps (bool): Whether to use filtered Wiki dumps instead of the full ones.
    """

    wiki_dump_api.parse(
        entity_config={"limit": entity_limit},
        article_text_config={"limit": article_limit},
        alias_prior_prob_config={"limit": alias_limit},
        use_filtered_dumps=use_filtered_dumps
    )


if __name__ == "__main__":
    # typer.run(main)
    wiki_dump_api.parse(
        entity_config={"limit": None},
        article_text_config={"limit": None},
        alias_prior_prob_config={"limit": None},
        use_filtered_dumps=False
    )

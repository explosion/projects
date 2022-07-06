""" Parsing of Wiki dump and persisting of parsing results to DB. """
import os
from pathlib import Path
from typing import Optional

from wiki import wiki_dump_api
import typer


def main(entity_limit: Optional[int] = None, article_limit: Optional[int] = None, alias_limit: Optional[int] = None):
    """Parses Wikidata and Wikipedia dumps. Persists parsing results to DB.
    entity_limit (Optional[int]): Max. number of entities to parse. Unlimited if None.
    article_limit (Optional[int]): Max. number of entities to parse. Unlimited if None.
    alias_limit (Optional[int]): Max. number of entity alias_entity_prior_probs to parse. Unlimited if None.
    """

    wiki_dump_api.parse(
        entity_config={"limit": entity_limit},
        article_text_config={"limit": article_limit},
        alias_prior_prob_config={"limit": alias_limit}
    )


if __name__ == "__main__":
    typer.run(main)
    # main(entity_limit=1000000, article_limit=1, alias_limit=1000000000)

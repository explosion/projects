""" Various utils. """

import logging
from pathlib import Path
from typing import Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(handle: str) -> logging.Logger:
    """Get logger for handle.
    handle (str): Logger handle.
    RETURNS (logging.Logger): Logger.
    """

    return logging.getLogger(handle)


def read_filter_terms() -> Set[str]:
    """Read terms used to filter Wiki dumps/corpora.
    RETURNS (Set[str]): Set of filter terms.
    """
    with open(Path(__file__).parent.parent / "configs" / "filter_terms.txt", "r") as file:
        return {ft.replace("\n", "") for ft in file.readlines()}

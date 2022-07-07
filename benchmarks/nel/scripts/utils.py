""" Various utils. """

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(handle: str) -> logging.Logger:
    """
    Gets logger for handle.
    handle (str): Logger handle.
    RETURNS (logging.Logger): Logger.
    """

    return logging.getLogger(handle)

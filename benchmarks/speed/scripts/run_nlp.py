import typer
import timeit
from pathlib import Path
from data_reader import read_data
from nlp_models import get_all_nlp_functions
from logger import create_logger
from wasabi import Printer


def main(txt_dir: Path, result_dir: Path):
    msg = Printer()
    log_run = create_logger(result_dir)
    data = read_data(txt_dir)
    articles = len(data)
    chars = sum([len(d) for d in data])
    words = sum([len(d.split()) for d in data])

    # actual benchmark
    for name, gpu, nlp_function in get_all_nlp_functions():
        msg.info(f"Running {name} with GPU={gpu}")
        start = timeit.default_timer()
        nlp_function(data)
        end = timeit.default_timer()

        s = end - start
        log_run(
            name=name,
            gpu=gpu,
            articles=articles,
            characters=chars,
            words=words,
            seconds=s,
        )


if __name__ == "__main__":
    typer.run(main)

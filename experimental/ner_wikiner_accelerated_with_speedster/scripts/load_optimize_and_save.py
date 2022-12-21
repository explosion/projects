import spacy
from spacy.training import Corpus

from extra_components import SpeedsterTransformerModel


def _get_dumb_kwargs():
    return {"name": "SpeedsterTransformer", "get_spans": lambda x: x}


def _load_data(data_path, max_size=500):
    corpus = Corpus(data_path, limit=max_size)
    nlp = spacy.blank("en")
    corpus_list = [x.text for x in corpus(nlp)]
    return corpus_list


def load_and_optimize_and_save(model_path, data_path, **kwargs):
    dumb_kwargs = _get_dumb_kwargs()
    model = SpeedsterTransformerModel(**dumb_kwargs).from_disk(model_path)
    input_data = _load_data(data_path)
    model.optimize(input_data, **kwargs)
    model.to_disk(model_path)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data_path", "-d", type=str, help="Path to data")
    parser.add_argument("--model_path", "-m", type=str, help="Path to model")
    parser.add_argument("--acc_ths", "-at", type=float, default=0.1, help="Accepted accuracy drop")
    parser.add_argument("--optimization_time", "-ot", type=str, default="unconstrained", help="Optimization setup")
    args = parser.parse_args()
    load_and_optimize_and_save(args.model_path, args.data_path, metric_drop_ths=args.acc_ths, optimization_time=args.optimization_time)

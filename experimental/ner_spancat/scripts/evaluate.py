from typing import Optional
from pathlib import Path
from spacy.cli.evaluate import evaluate
from spacy.cli._util import Arg, Opt, import_code
import typer


def evaluate_cli(
    # fmt: off
    model: str = Arg(..., help="Model name or path"),
    data_path: Path = Arg(..., help="Location of binary evaluation data in .spacy format", exists=True),
    output: Optional[Path] = Opt(None, "--output", "-o", help="Output JSON file for metrics", dir_okay=False),
    code_path: Optional[Path] = Opt(None, "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
    use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU"),
    gold_preproc: bool = Opt(False, "--gold-preproc", "-G", help="Use gold preprocessing"),
    displacy_path: Optional[Path] = Opt(None, "--displacy-path", "-dp", help="Directory to output rendered parses as HTML", exists=True, file_okay=False),
    displacy_limit: int = Opt(25, "--displacy-limit", "-dl", help="Limit of parses to render as HTML"),
    spans_key: str = Opt("sc", "--spans-key", help="spancat spans_key"),
    # fmt: on
):
    """
    Evaluate a trained pipeline. Expects a loadable spaCy pipeline and evaluation
    data in the binary .spacy format. The --gold-preproc option sets up the
    evaluation examples with gold-standard sentences and tokens for the
    predictions. Gold preprocessing helps the annotations align to the
    tokenization, and may result in sequences of more consistent length. However,
    it may reduce runtime accuracy due to train/test skew. To render a sample of
    dependency parses in a HTML file, set as output directory as the
    displacy_path argument.

    DOCS: https://spacy.io/api/cli#evaluate
    """
    import_code(code_path)
    evaluate(
        model,
        data_path,
        output=output,
        use_gpu=use_gpu,
        gold_preproc=gold_preproc,
        displacy_path=displacy_path,
        displacy_limit=displacy_limit,
        silent=False,
        spans_key=spans_key,
    )


if __name__ == "__main__":
    typer.run(evaluate_cli)

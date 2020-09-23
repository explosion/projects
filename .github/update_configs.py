from pathlib import Path
from spacy.cli.init_config import fill_config
from wasabi import msg
import typer


def main(
    root: Path = typer.Argument(Path.cwd(), help="Root path of repo"),
    skip: bool = typer.Option(False, "--skip", "-s", help="Skip errors"),
):
    """
    Automatically auto-fill all .cfg files in the repo using the current version
    of spaCy. If --skip is set, errors will be logged but not raised.
    """
    msg.info(f"Updating configs in {root}")
    for path in root.glob("**/configs/*.cfg"):
        rel_path = path.relative_to(root)
        if rel_path.parts[0].startswith("."):
            continue
        print(rel_path)
        try:
            before, after = fill_config(path, path, silent=True)
        except (Exception, SystemExit) as e:
            if skip:
                msg.fail("Failed", e)
                continue
            else:
                raise
        if before != after:
            msg.good("Filled")
        else:
            msg.info("Already up to date")


if __name__ == "__main__":
    typer.run(main)

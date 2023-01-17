from pathlib import Path
from spacy.cli._util import PROJECT_FILE, load_project_config
from wasabi import msg
import json
import typer


def main(root: Path = typer.Argument(Path.cwd(), help="Root path to look in")):
    """
    Update the projects.jsonl file for the repo.

    Unlike the docs update script, this is desigend to only be run on the root
    of the whole repository.
    """
    msg.info(f"Updating projects.jsonl in {root}")
    entries = []
    # We look specifically for project directories
    for path in root.glob(f"**/*/{PROJECT_FILE}"):
        path = path.parent

        # prep data for the json file
        config = load_project_config(path)
        entry = {"shortname": f"{path.parent.name}/{path.name}"}
        entry["title"] = config["title"]
        entry["description"] = config.get("description", "")
        entries.append(entry)

    with open("projects.jsonl", "w") as jsonfile:
        for entry in entries:
            jsonfile.write(json.dumps(entry))
            jsonfile.write("\n")


if __name__ == "__main__":
    typer.run(main)

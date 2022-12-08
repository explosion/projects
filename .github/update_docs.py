from pathlib import Path
from spacy.cli.project.document import project_document
from spacy.cli._util import PROJECT_FILE, load_project_config
from wasabi import msg
import json
import typer


def main(root: Path = typer.Argument(Path.cwd(), help="Root path to look in")):
    """
    Automatically update all auto-generated docs in the repo. If existing
    auto-generated docs are found, only that section is replaced. README.md
    files including an ignore comment are skipped (e.g. to support projects
    without an auto-generated README and prevent those files from being
    auto-replaced).
    """
    msg.info(f"Updating auto-generated docs in {root}")
    entries = []
    # We look specifically for project directories
    for path in root.glob(f"**/*/{PROJECT_FILE}"):
        path = path.parent
        project_document(path, path / "README.md")

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

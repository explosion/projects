from pathlib import Path
from spacy.cli._util import PROJECT_FILE, load_project_config
from wasabi import msg, MarkdownRenderer
import typer


CATEGORIES = "pipelines,tutorials,integrations,benchmarks,experimental"
README_HEADER = """<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>"""


def main(
    root: Path = typer.Argument(Path.cwd(), help="Root path of repo"),
    categories: str = typer.Argument(CATEGORIES, help="Comma-separated names of categories")
):
    """
    Automatically create and/or update the README.md of each category directory
    with an overview of the available
    """
    categories = [c.strip() for c in categories.split(",")]
    msg.info(f"Updating auto-generated category docs in {root}", categories)
    for path in root.iterdir():
        if path.is_dir() and path.name in categories:
            generate_readme(path)


def generate_readme(path: Path):
    templates = {}
    for child in sorted(path.iterdir()):
        if child.is_dir():
            if (child / PROJECT_FILE).exists():
                cfg = load_project_config(child)
                templates[child.name] = cfg.get("title", "")
    md = MarkdownRenderer()
    md.add(README_HEADER)
    title = f"Project Templates: {path.name.capitalize()} ({len(templates)})"
    md.add(md.title(1, title, "🪐"))
    data = [(md.link(md.code(n), n), t) for n, t in templates.items()]
    if data:
        md.add(md.table(data, ["Template", "Description"]))
    readme_path = path / "README.md"
    with readme_path.open("w", encoding="utf8") as f:
        f.write(md.text)
    msg.good(f"Auto-generated {readme_path}")


if __name__ == "__main__":
    typer.run(main)

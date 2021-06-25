import os
import typer
import tempfile
import tarfile
import shutil
from huggingface_hub import Repository, HfApi, HfFolder

token_classification_pipelines = ["ner", "tagger", "morphologizer"]
text_classification_pipelines = ["textcat", "textcat_multilabel"]

def main(
    name: str,
    version: str,
    lang: str,
    namespace: str,
    repo_name: str,
):
    full_name = f"{lang}_{name}-{version}"
    package_path = f"packages/{full_name}/dist"
    tar_filename = f"{package_path}/{full_name}.tar.gz"
    whl_filename = f"{full_name}-py3-none-any.whl"
    repo_path = package_path + "/hub/"

    # Create the repo (or clone its content if it's nonempty).
    repo_url = HfApi().create_repo(
            name=full_name,
            token=HfFolder.get_token(),
            private=False,
            exist_ok=True,
        )
    repo = Repository(repo_path, clone_from=repo_url)
    repo.lfs_track(["*.whl"])
    repo.lfs_track(["*.npz"])
    repo.lfs_track(["*strings.json"])
    repo.lfs_track(["vectors"])

    with tempfile.TemporaryDirectory() as tmp_dir2:
        tar = tarfile.open(tar_filename, "r:gz")
        tar.extractall(repo_path)

        # Remove version from name
        src_dir = os.path.join(package_path, whl_filename)
        dst_dir = os.path.join(repo_path, whl_filename)
        shutil.copyfile(src_dir, dst_dir)
    repo.push_to_hub(commit_message="Spacy Update")


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        pass

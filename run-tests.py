import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Union

import srsly
import typer


def _run(
    cmd: List[str], cmds_status: List[Dict[str, Union[str, int, List[str]]]], capture_output: bool = True
) -> subprocess.CompletedProcess:
    """
    Runs command. Returns `CompletedProcess` instance.
    cmd (List[str]): Command to run.
    cmds_status (List[Dict[str, Union[str, int, List[str]]]]): Info (command, error code, error messages) for every
        executed command. Updated in-place.
    capture_output (bool): Whether to capture process output.
    RETURNS (subprocess.CompletedProcess): Process object.
    """
    proc_output = subprocess.run(cmd, capture_output=capture_output)
    cmds_status.append(
        {
            "cmd": " ".join([str(cmd_part) for cmd_part in cmd]),
            "stderr": proc_output.stderr.decode("utf-8").split("\n"),
            "returncode": proc_output.returncode
        }
    )
    return proc_output


def _get_logger() -> logging.Logger:
    """
    Configures and returns logger instance.
    RETURNS (logging.Logger): Configured logger.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    return logging.getLogger(__name__)


def main(
    python: Path = typer.Argument(...),
    run_all: bool = typer.Option(False)
) -> int:
    """
    Runs tests.
    python (Path): Path to Python interpreter to use for tests.
    run_all (bool): Whether to run tests for all projects. If False, only tests for changed directories are run.
    RETURN (int): Return code.
    """
    logger = _get_logger()
    cmds_status: List[Dict[str, Union[str, int, List[str]]]] = []
    top_level_dirs = ("benchmarks", "experimental", "integrations", "pipelines", "tutorials")

    if run_all:
        proj_dirs = [
            path for path in
            [Path(tl_dir) / subdir for tl_dir in top_level_dirs for subdir in os.listdir(tl_dir)]
            if path.is_dir()
        ]
    else:
        # Fetch which files were changed in the last commit.
        _run(["git", "diff", "--name-only", "HEAD", "HEAD~1"], cmds_status)
        proj_dirs = {
            Path(file).parent for file in
            _run(
                ["git", "diff", "--name-only", "HEAD", "HEAD~1"], cmds_status
            ).stdout.decode("utf-8").split("\n")
            if Path(file).parent != Path(".")
        }

    # todo run tests.
    for proj_dir in proj_dirs:
        logger.info(f"*** {proj_dir} ***")
        # Install from requirements.txt, if it exists.
        if (proj_dir / "requirements.txt").exists():
            logger.info("  Installing requirements")
            _run([python, "-m", "pip", "-q", "install", "-r", proj_dir / "requirements.txt"], cmds_status)

        # Fetch spacy version from project.yml, install spacy version if available.
        spacy_version = srsly.read_yaml(proj_dir / "project.yml").get("spacy_version")
        if spacy_version:
            logger.info(f"  Installing spacy{spacy_version} from project.yml")
            _run([python, "-m", "pip", "-q", "install", f"spacy{spacy_version}", "--force-reinstall"], cmds_status)

        # todo run pytest on dir

        logger.info("  Restoring environment")
        with tempfile.NamedTemporaryFile("w") as file:
            file.writelines(
                _run(
                    [python, "-m", "pip", "freeze", "--exclude", "torch", "cupy-cuda111"], cmds_status
                ).stdout.decode("utf-8").split("\n")
            )
            _run([python, "-m", "pip", "-q", "uninstall", "-y", "-r", "installed.txt"], cmds_status)
            _run([python, "-m", "pip", "-q", "install", "-r", "requirements.txt"], cmds_status)

    for cmd_status in cmds_status:
        logger.info(cmd_status)

    # Return 0 if all commands hat return code 0, 1 otherwise.
    return not all([cmd_status["returncode"] == 0 for cmd_status in cmds_status])


if __name__ == '__main__':
    main(Path(".venv") / "bin" / "python", True)
    # typer.run(main)
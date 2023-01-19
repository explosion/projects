import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

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
    subprocess._USE_VFORK = False
    subprocess._USE_POSIX_SPAWN = False
    proc_output = subprocess.run(
        cmd,
        capture_output=capture_output,
    )
    cmds_status.append(
        {
            "cmd": " ".join([str(cmd_part) for cmd_part in cmd]),
            "stdout": proc_output.stdout.decode("utf-8").split("\n"),
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


def _log(logger: logging.Logger, verbosity: Optional[str], required_verbosity: str, txt: str) -> None:
    """
    Logs text, if the verbosity level is sufficient.
    verbosity (Optional[str]): Verbosity level. If None, nothing is logged. If "v", step description is logged. If "vv",
        step description, pytest output and, before terminating the script, a complete list of stdout/stderr/return
        codes script) are logged.
    required_verbosity (Optional[str]): Required verbosity level to log this text.
    """
    if verbosity == required_verbosity or verbosity == required_verbosity + "v":
        logger.info(txt)


def main(
    python: Path = typer.Argument(...),
    run_all: bool = typer.Option(False),
    verbosity: Optional[str] = typer.Option(None)
) -> int:
    """
    Runs tests.
    python (Path): Path to Python interpreter to use for tests.
    run_all (bool): Whether to run tests for all projects. If False, only tests for changed directories are run.
    verbosity (Optional[str]): Verbosity level. If None, nothing is logged. If "v", step description is logged. If "vv",
        pytest output and detailed information are logged.
    RETURN (int): Return code.
    """
    logger = _get_logger()
    cmds_status: List[Dict[str, Union[str, int, List[str]]]] = []
    top_level_dirs = ("experimental",)  # "benchmarks", "experimental", "integrations", "pipelines", "tutorials")
    logger.info(python)
    if run_all:
        _log(logger, verbosity, "v", "Scanning projects")
        proj_dirs = [
            path for path in
            [Path(tl_dir) / subdir for tl_dir in top_level_dirs for subdir in os.listdir(tl_dir)]
            if path.is_dir() and "coref" in str(path)
        ]
    else:
        _log(logger, verbosity, "v", "Fetching modified projects")
        # Fetch which files were changed in the last commit.
        _run(["git", "diff", "--name-only", "HEAD", "HEAD~1"], cmds_status)
        proj_dirs = {
            Path(file).parent for file in
            _run(
                ["git", "diff", "--name-only", "HEAD", "HEAD~1"], cmds_status
            ).stdout.decode("utf-8").split("\n")
            if Path(file).parent != Path(".")
        }

    for proj_dir in proj_dirs:
        _log(logger, verbosity, "v", f"*** {proj_dir} ***")

        # Install from requirements.txt, if it exists.
        if (proj_dir / "requirements.txt").exists():
            _log(logger, verbosity, "v", "  - Installing requirements")
            # todo works from terminal, but not from script
            # _run([python, "-m", "pip", "-q", "install", "-r", proj_dir / "requirements.txt"], cmds_status)
        # exit()
        # Fetch spacy version from project.yml, install spacy version if available.
        spacy_version = srsly.read_yaml(proj_dir / "project.yml").get("spacy_version")
        if spacy_version:
            _log(logger, verbosity, "v", f"  - Installing spacy{spacy_version} from project.yml")
            # _run([python, "-m", "pip", "-q", "install", f"spacy{spacy_version}", "--force-reinstall"], cmds_status)

        _log(logger, verbosity, "v", "  - Running tests")
        _run([python, "-m", "pytest", "-q", "-s", proj_dir], cmds_status)
        # Log stderr and stdout.
        if cmds_status[-1]["returncode"] not in (0, 5):
            for channel in ("stderr", "stdout"):
                _log(logger, verbosity, "vv", f"    ### {channel} ###")
                for out in cmds_status[-1][channel]:
                    _log(logger, verbosity, "vv", f"    {out}")
        elif cmds_status[-1]["returncode"] == 5:
            _log(logger, verbosity, "vv", "      No tests found")

        _log(logger, verbosity, "v", "  - Restoring environment")
        with tempfile.NamedTemporaryFile("w") as file:
            file.writelines(
                _run(
                    [python, "-m", "pip", "freeze", "--exclude", "torch", "cupy-cuda111"], cmds_status
                ).stdout.decode("utf-8").split("\n")
            )
            _run([python, "-m", "pip", "-q", "uninstall", "-y", "-r", file.name], cmds_status)
            _run([python, "-m", "pip", "-q", "install", "-r", "requirements.txt"], cmds_status)

    for cmd_status in cmds_status:
        _log(logger, verbosity, "vv", str(cmd_status))

    # failing:
    #   - experimental/ner_wikiner_speedster (GIL error, maybe due to python 3.9?)

    # Return 0/True if all commands have return code 0 or 5 (no tests found), 1/False otherwise.
    return int(not all([cmd_status["returncode"] in (0, 5) for cmd_status in cmds_status]))


if __name__ == '__main__':
    # rm -rf .venv && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && source .env/bin/activate
    if main(Path(".venv") / "bin" / "python", True, "vv") != 0:
        # if typer.run(main) != 0:
        raise Exception("Test run failed. Review logs for more details.")

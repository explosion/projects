import sys
from typing import IO, Tuple, Callable, Dict, Any, Optional
import spacy
from spacy.language import Language
from pathlib import Path

from spacy.training import console_logger


@spacy.registry.loggers("my_custom_logger.v1")
def custom_logger(log_path):
    console = console_logger(progress_bar=False)

    def setup_logger(
        nlp: Language, stdout: IO = sys.stdout, stderr: IO = sys.stderr
    ) -> Tuple[Callable, Callable]:
        stdout.write(f"Logging to {log_path}\n")
        log_file = Path(log_path).open("w", encoding="utf8")
        log_file.write("epoch\t")
        log_file.write("step\t")
        log_file.write("score\t")
        for pipe in nlp.pipe_names:
            log_file.write(f"loss_{pipe}\t")
        log_file.write("cats_micro_p\t")
        log_file.write("cats_micro_r\t")
        log_file.write("cats_micro_f\t")
        log_file.write("cats_macro_p\t")
        log_file.write("cats_macro_r\t")
        log_file.write("cats_macro_f\t")
        log_file.write("cats_macro_auc\t")
        log_file.write("cats_f_per_type\t")
        log_file.write("cats_auc_per_type\t")
        log_file.write("\n")
        console_log_step, console_finalize = console(nlp, stdout, stderr)

        def log_step(info: Optional[Dict[str, Any]]):
            console_log_step(info)
            if info:
                log_file.write(f"{info['epoch']}\t")
                log_file.write(f"{info['step']}\t")
                log_file.write(f"{info['score']}\t")
                for pipe in nlp.pipe_names:
                    log_file.write(f"{info['losses'][pipe]}\t")
                log_file.write(f"{info['other_scores']['cats_micro_p']}\t")
                log_file.write(f"{info['other_scores']['cats_micro_r']}\t")
                log_file.write(f"{info['other_scores']['cats_micro_f']}\t")
                log_file.write(f"{info['other_scores']['cats_macro_p']}\t")
                log_file.write(f"{info['other_scores']['cats_macro_r']}\t")
                log_file.write(f"{info['other_scores']['cats_macro_f']}\t")
                log_file.write(f"{info['other_scores']['cats_macro_auc']}\t")
                log_file.write(f"{info['other_scores']['cats_f_per_type']}\t")
                log_file.write(f"{info['other_scores']['cats_auc_per_type']}\t")
                log_file.write("\n")

        def finalize():
            console_finalize()
            log_file.close()

        return log_step, finalize

    return setup_logger

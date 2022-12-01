"""
API call for training. Mainly for debugging purposes.
"""
from pathlib import Path
import custom_functions
from spacy.cli.train import train

if __name__ == '__main__':
    root = Path(__file__).parent.parent
    train(
        root / "configs" / "nel.cfg",
        output_path=root / "training" / "mewsli_9" / "default",
        use_gpu=0,
        overrides={
            "paths.dataset_name": "mewsli_9",
            "paths.train": "corpora/mewsli_9/train.spacy",
            "paths.dev": "corpora/mewsli_9/dev.spacy",
            "paths.kb": "wikid/output/en/kb",
            "paths.db": "wikid/output/en/wiki.sqlite3",
            "paths.base_nlp": "training/base-nlp/en",
            "paths.language": "en",
            "training.max_steps": 10,
        }
    )

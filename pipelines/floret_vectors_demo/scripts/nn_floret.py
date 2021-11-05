import typer
from pathlib import Path
import floret


def main(
    input_file: Path,
    word: str,
):
    floret_model = floret.load_model(str(input_file.absolute()))
    print(f"\nNearest neighbors for '{word}':\n")
    for nn_score, nn_word in floret_model.get_nearest_neighbors(word):
        print(f"{nn_word:12} {nn_score:.04f}")


if __name__ == "__main__":
    typer.run(main)

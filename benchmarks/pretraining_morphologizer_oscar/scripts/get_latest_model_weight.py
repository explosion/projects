import os
import yaml
import typer
from pathlib import Path
from wasabi import msg


def main(weights_folder: Path, project_file: Path):
    number_list = []
    for file in os.listdir(weights_folder):
        if "model" in file:
            number_list.append(int(file[5:-4]))

    try:
        model_name = f"model{max(number_list)}.bin"
    except:
        msg.warn("No pretrained weights found. Make sure to run the pretrain command.")
        return

    with open(project_file, "r") as file:
        yaml_content = yaml.safe_load(file)

    yaml_content["vars"]["pretraining_model"] = model_name

    with open(project_file, "w") as outfile:
        yaml.dump(yaml_content, outfile, default_flow_style=False)


if __name__ == "__main__":
    typer.run(main)

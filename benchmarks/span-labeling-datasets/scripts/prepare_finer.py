import srsly
import os
import tqdm


for split in ["train", "validation", "test"]:
    path = os.path.join("temp", f"{split}.jsonl")
    data = srsly.read_jsonl(path)
    if split == "validation":
        name = "dev"
    else:
        name = split
    outpath = os.path.join("assets/", f"finer-{name}.iob")
    with open(outpath, "w") as fiob:
        for datum in tqdm.tqdm(data):
            tokens, tags = datum["tokens"], datum["ner_tags"]
            assert len(tokens) == len(tags)
            for token, tag in zip(tokens, tags):
                if token == "":
                    continue
                line = f"{token} \t {tag}\n"
                fiob.write(line)
            fiob.write("\n")

import spacy
import re
import tempfile
import typer
from conll18_ud_eval import load_conllu, evaluate
from typing import Optional, List
import code


def main(model: str, gold_conllu: str, text: Optional[str] = None, output: Optional[str] = None, gpu_id: int = -1):
    if gpu_id >= 0:
        spacy.require_gpu(gpu_id)

    # load model from model name or path
    nlp = spacy.load(model)

    # if raw text input is provided, load texts from plain text file
    if text:
        with open(text) as fileh:
            content = fileh.read()
            texts = [text.replace("\n", " ").strip() for text in content.split("\n\n")]
    # otherwise generate raw text input from gold CoNLL-U file
    else:
        texts = gold_to_texts(gold_conllu)

    # apply model to texts
    docs = nlp.pipe(texts)

    # load the system CoNLL-U predictions by creating a temporary CoNLL-U
    # output file
    output_lines = []
    for doc in docs:
        for sent in doc.sents:
            output_lines.append("# text = " + sent.text + "\n")
            for i, token in enumerate(sent):
                if token.is_space:
                    continue
                cols = ["_"] * 10
                cols[0] = str(i + 1)
                cols[1] = token.text
                if token.lemma_:
                    cols[2] = token.lemma_
                if token.pos_:
                    cols[3] = token.pos_
                if token.tag_:
                    cols[4] = token.tag_
                if str(token.morph):
                    cols[5] = str(token.morph)
                cols[6] = "0" if token.head.i == token.i else str(token.head.i + 1 - sent[0].i)
                cols[7] = "root" if token.dep_ == "ROOT" else token.dep_
                if not token.whitespace_:
                    cols[9] = "SpaceAfter=No"
                output_lines.append("\t".join(cols) + "\n")
            output_lines.append("\n")
    with tempfile.TemporaryFile("w+") as fileh:
        fileh.writelines(output_lines)
        fileh.flush()
        fileh.seek(0)
        pred_corpus = load_conllu(fileh)

    # load the gold CoNLL-U file
    with open(gold_conllu) as fileh:
        gold_corpus = load_conllu(fileh)

    # run the CoNLL 2018 shared task evaluation
    scores = evaluate(gold_corpus, pred_corpus)

    # output evaluation to output file or stdout
    evaluation_table = format_evaluation(scores)
    if output:
        with open(output, "w") as fileh:
            fileh.write(evaluation_table)
    else:
        print("".join(evaluation_table))


def format_evaluation(evaluation) -> List[str]:
    """Evaluation table formatting adapted from CoNLL-U 2018 shared task
    evaluation script."""
    lines = []
    lines.append("Metric     | Precision |    Recall |  F1 Score | AligndAcc")
    lines.append("-----------+-----------+-----------+-----------+-----------")
    for metric in["Tokens", "Sentences", "Words", "UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]:
        lines.append("{:11}|{:10.2f} |{:10.2f} |{:10.2f} |{}".format(
            metric,
            100 * evaluation[metric].precision,
            100 * evaluation[metric].recall,
            100 * evaluation[metric].f1,
            "{:10.2f}".format(100 * evaluation[metric].aligned_accuracy) if evaluation[metric].aligned_accuracy is not None else ""
        ))
    return "\n".join(lines)


def gold_to_texts(gold_conllu: str) -> List[str]:
    """Create raw texts from CoNLL-U file. Uses "# newdoc"/"# newpar"
    divisions, "# text" lines, and "SpaceAfter=No" to build a list of raw
    texts."""
    texts = []
    with open(gold_conllu) as fileh:
        text = ""
        prev_line = ""
        for line in fileh:
            line = line.strip()
            if text and (line.startswith("# newdoc") or line.startswith("# newpar")):
                texts.append(re.sub("\s+", " ", text.strip()))
                text = ""
            if line.startswith("# text = "):
                text += line.replace("# text = ", "")
            if line == "":
                cols = prev_line.split("\t")
                if len(cols) >= 10:
                    if not "SpaceAfter=No" in cols[9]:
                        text += " "
            prev_line = line
        if text:
            texts.append(text)
    return texts


if __name__ == "__main__":
    typer.run(main)

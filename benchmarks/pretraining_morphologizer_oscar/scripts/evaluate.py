from typing import Optional, List
import glob
from pathlib import Path
import re
import tempfile
import typer
import srsly
import spacy
from thinc.api import set_gpu_allocator
from conll18_ud_eval import load_conllu, evaluate


def main(
    model: str,
    gold_dir: Path,
    output: Optional[str] = None,
    gpu_id: int = -1,
    batch_size: int = 64,
    sents_per_text: int = -1,
    enable_senter: bool = False,
):
    test_txt_file = None
    test_txt_files = glob.glob(str(gold_dir.resolve()) + "/*test.txt")
    if len(test_txt_files) > 0:
        test_txt_file = test_txt_files[0]
    test_conllu_file = None
    test_conllu_files = glob.glob(str(gold_dir.resolve()) + "/*test.conllu")
    if len(test_conllu_files) > 0:
        test_conllu_file = test_conllu_files[0]

    # if raw text input is provided, load texts from plain text file
    if test_txt_file:
        with open(test_txt_file) as fileh:
            content = fileh.read()
            texts = [
                re.sub(r"\s+", " ", text.replace("\n", " ").strip())
                for text in content.split("\n\n")
            ]
    # otherwise generate raw text input from gold CoNLL-U file
    elif test_conllu_file:
        texts = gold_to_texts(test_conllu_file)
    else:
        raise ValueError("No test.txt or test.conllu files found in", gold_dir)

    # if desired (for GPU), break up very long texts
    if sents_per_text > 0:
        split_texts = []
        sentencizer_nlp = spacy.blank("xx")
        sentencizer_nlp.max_length = max(len(text) + 1 for text in texts)
        sentencizer_nlp.add_pipe("sentencizer")
        for doc in sentencizer_nlp.pipe(texts):
            sents = list(doc.sents)
            for i in range(0, len(sents), sents_per_text):
                if len(sents) < i + sents_per_text:
                    end_t = len(doc)
                else:
                    end_t = sents[i + sents_per_text][0].i
                split_texts.append(doc[sents[i][0].i : end_t].text)
    else:
        split_texts = texts

    if gpu_id >= 0:
        spacy.require_gpu(gpu_id)
        set_gpu_allocator("pytorch")

    # load model from model name or path
    nlp = spacy.load(model)
    if enable_senter:
        nlp.enable_pipe("senter")
    else:
        nlp.disable_pipe("senter")

    # apply model to texts (batch_size=1 is slow, but reduces chance of OOM)
    docs = nlp.pipe(split_texts, batch_size=1)

    # load the system CoNLL-U predictions by creating a temporary CoNLL-U
    # output file
    output_lines = []
    for doc in docs:
        for sent in doc.sents:
            output_lines.append("# text = " + sent.text + "\n")
            assert all(not token.is_space for token in doc)
            for i, token in enumerate(sent):
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
                cols[6] = (
                    "0"
                    if token.head.i == token.i
                    else str(token.head.i + 1 - sent[0].i)
                )
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
    with open(test_conllu_file) as fileh:
        gold_corpus = load_conllu(fileh)

    # run the CoNLL 2018 shared task evaluation
    scores = evaluate(gold_corpus, pred_corpus)

    # output evaluation to output file or stdout
    evaluation_table = format_evaluation(scores)
    if output:
        with open(output + ".txt", "w") as fileh:
            fileh.write(evaluation_table)
        scores_dict = {}
        for key, score in scores.items():
            scores_dict[key] = {
                "precision": score.precision,
                "recall": score.recall,
                "f1": score.f1,
                "aligned_accuracy": score.aligned_accuracy,
            }
        srsly.write_json(output + ".json", scores_dict)
    else:
        print("".join(evaluation_table))


def format_evaluation(evaluation) -> List[str]:
    """Evaluation table formatting adapted from CoNLL-U 2018 shared task
    evaluation script."""
    lines = []
    lines.append("Metric     | Precision |    Recall |  F1 Score | AligndAcc")
    lines.append("-----------+-----------+-----------+-----------+-----------")
    for metric in [
        "Tokens",
        "Sentences",
        "Words",
        "UPOS",
        "XPOS",
        "UFeats",
        "AllTags",
        "Lemmas",
        "UAS",
        "LAS",
        "CLAS",
        "MLAS",
        "BLEX",
    ]:
        lines.append(
            "{:11}|{:10.2f} |{:10.2f} |{:10.2f} |{}".format(
                metric,
                100 * evaluation[metric].precision,
                100 * evaluation[metric].recall,
                100 * evaluation[metric].f1,
                "{:10.2f}".format(100 * evaluation[metric].aligned_accuracy)
                if evaluation[metric].aligned_accuracy is not None
                else "",
            )
        )
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
                texts.append(re.sub(r"\s+", " ", text.strip()))
                text = ""
            if line.startswith("# text = "):
                text += line.replace("# text = ", "")
            if line == "":
                cols = prev_line.split("\t")
                if len(cols) >= 10:
                    if "SpaceAfter=No" not in cols[9]:
                        text += " "
            prev_line = line
        if text:
            texts.append(text)
    return texts


if __name__ == "__main__":
    typer.run(main)

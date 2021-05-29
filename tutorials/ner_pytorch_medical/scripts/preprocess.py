from collections import defaultdict
import random
from typing import List
import tarfile
import shutil
import typer
import srsly
from pathlib import Path
import spacy
from spacy.language import Language
from spacy.tokens import Doc, DocBin, Span
from spacy.util import get_words_and_spaces, filter_spans


random.seed(42)


def main(
    input_dir: Path = typer.Argument(..., exists=True),
    output_dir: Path = typer.Argument(...),
    beth_train_tar_name: str = "i2b2_Beth_Train_Release.tar.gz",
    partners_train_tar_name: str = "i2b2_Partners_Train_Release.tar.gz",
    test_zip_name: str = "Task_1C.zip",
    merge_docs: bool = True
):
    """Extract and preprocess raw n2c2 2011 Challenge data into spaCy DocBin format.
    input_dir (Path, optional): Input directory with raw downloads from Harvard DBMI Portal.
    output_dir (Path, optional): Output directory to save spaCy .docbin files to.
    beth_train_tar_name (str, optional): Filename of downloaded tarfile for Beth Training Data.
    partners_train_tar_name (str, optional): Filename of downloaded tarfile for Partners Training Data.
    test_zip_name (str, optional): Filename of downloaded tarfile for n2c2 Test Data.
    merge_docs (bool, optional): If False, create spaCy docs for each line of each medical record 
    """
    # Unpack compressed data files
    print("Extracting raw data.")
    beth_train_tar_path = input_dir / beth_train_tar_name
    partners_train_tar_path = input_dir / partners_train_tar_name
    test_zip_path = input_dir / test_zip_name

    for path in [beth_train_tar_path, partners_train_tar_path, test_zip_path]:
        if not path.exists():
            raise ValueError(
                f"Path {path} does not exist. This likely means you have not downloaded"
                + f"the raw data from the Harvard DBMI Portal."
                + f"Create an account and download the data packages to {input_dir}"
            )

    for path in [beth_train_tar_path, partners_train_tar_path]:
        if path.name.endswith("tar.gz"):
            print(f"Extracting {path}")
            tar = tarfile.open(path, "r:gz")
            tar.extractall(path.parent)
            tar.close()

    shutil.unpack_archive(test_zip_path, input_dir / test_zip_name.replace(".zip", ""))

    # preprocess data
    print("Converting to spaCy Doc objects.")
    beth_train_docs = docs_from_many_clinical_records(input_dir / "Beth_Train", merge_docs=merge_docs)
    partners_train_docs = docs_from_many_clinical_records(input_dir / "Partners_Train", merge_docs=merge_docs)
    train_docs = beth_train_docs + partners_train_docs

    beth_test_docs = docs_from_many_clinical_records(
        input_dir / "Task_1C/i2b2_Test/i2b2_Beth_Test", merge_docs=merge_docs
    )
    partners_test_docs = docs_from_many_clinical_records(
        input_dir / "Task_1C/i2b2_Test/i2b2_Partners_Test", merge_docs=merge_docs
    )
    test_docs = beth_test_docs + partners_test_docs

    random.shuffle(train_docs)
    split_idx = int(len(train_docs) * 0.8)
    train_docs, dev_docs = train_docs[:split_idx], train_docs[split_idx:]

    print(f"Num Train Docs: {len(train_docs)}")
    print(f"Num Dev Docs: {len(dev_docs)}")
    print(f"Num Test Docs: {len(test_docs)}")

    print(f"Saving docs to: {output_dir}")
    DocBin(docs=train_docs).to_disk(output_dir / "train.spacy")
    DocBin(docs=dev_docs).to_disk(output_dir / "dev.spacy")
    DocBin(docs=test_docs).to_disk(output_dir / "test.spacy")


def docs_from_clinical_record(
    lines: List[str], annotations: List[str], nlp: Language, merge_docs: bool = False
) -> List[Doc]:
    """Create spaCy docs from a single annotated medical record in the n2c2 2011 format
    lines (List[str]): Text of the clinical record as a list separated by newlines
    annotations (List[str]): Raw entity annotations in the n2c2 2011 format
    nlp (Language, optional): spaCy Language object. Defaults to spacy.blank("en").
    merge_docs (bool, optional): If True: merge all lines into a single spaCy doc so
        there is only 1 element in the output array.
        If False: create a spaCy doc for each line in the original record
    RETURNS (List[Doc]): List of spaCy Doc objects with entity spans set
    """
    docs = []
    spans_by_line = defaultdict(list)

    for row in annotations:
        row = row.split("||")
        text_info = row[0]
        type_info = row[1]

        text = text_info.split('"')[1]

        offset_start = text_info.split(" ")[-2]
        offset_end = text_info.split(" ")[-1]

        start_line, word_start = offset_start.split(":")
        end_line, word_end = offset_end.split(":")

        label = type_info.split('"')[-2]

        if start_line != end_line:
            print("different line numbers")
            print(row)
            continue

        else:
            spans_by_line[int(start_line)].append(
                (int(word_start), int(word_end), label)
            )

    for i, line in enumerate(lines):
        n = i + 1
        doc = nlp.make_doc(line)
        if n in spans_by_line:
            ents = [
                Span(doc, start, end + 1, label=label)
                for (start, end, label) in spans_by_line[n]
            ]
            ents = [
                e for e in ents if bool(e.text.strip()) and e.text.strip() == e.text
            ]
            doc.ents = filter_spans(ents)

        docs.append(doc)

    return [Doc.from_docs(docs)] if merge_docs else docs


def docs_from_many_clinical_records(
    base_path: Path, nlp: Language = spacy.blank("en"), merge_docs: bool = True
) -> List[Doc]:
    """Convert raw n2c2 annotated clinical records into a list of
        spaCy Doc objects to be ready to be used in training
    base_path (Path): Root path to the raw data
    nlp (Language, optional): spaCy Language object. Defaults to spacy.blank("en").
    merge_docs (bool, optional): If True: merge all lines into a single spaCy doc so
        there is only 1 element in the output array.
        If False: create a spaCy doc for each line in the original record

    RETURNS (List[Doc]): List of spaCy Doc objects with entity spans set
    """
    all_docs = []
    concept_paths = sorted((base_path / "concepts").glob("*.txt.con"))
    document_paths = sorted((base_path / "docs").glob("*.txt"))

    for con_path, doc_path in zip(concept_paths, document_paths):
        annotations = con_path.open().read().splitlines()
        lines = doc_path.open().read().splitlines()

        docs = docs_from_clinical_record(lines, annotations, nlp, merge_docs=merge_docs)
        all_docs += docs

    return all_docs


if __name__ == "__main__":
    typer.run(main)

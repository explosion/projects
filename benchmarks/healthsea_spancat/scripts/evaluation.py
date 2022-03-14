import spacy
from spacy.tokens import DocBin
from spacy.scorer import PRFScore
from pathlib import Path

from tqdm import tqdm
from wasabi import msg
from wasabi import table
import typer


def main(
    span_key: str, ner_model_path: Path, spancat_model_path: Path, test_path: Path
):
    # Initialize NER & Spancat models
    ner_nlp = spacy.load(ner_model_path)
    spancat_nlp = spacy.load(spancat_model_path)

    # Get test.spacy DocBin
    test_docBin = DocBin().from_disk(test_path)
    test_docs = list(test_docBin.get_docs(spacy.blank("en").vocab))

    # Initialize scorers
    ner_scorer = {"CONDITION": PRFScore(), "BENEFIT": PRFScore()}
    spancat_scorer = {"CONDITION": PRFScore(), "BENEFIT": PRFScore()}

    KPI = {}

    for label in ner_scorer:
        KPI[label] = {
            "total_spans": 0,
            "correct_ner_spans": 0,
            "correct_spancat_spans": 0,
        }

    eval_list = []
    doc_eval_list = []

    msg.info("Starting evaluation")

    for test_doc in tqdm(
        test_docs, total=len(test_docs), desc=f"Evaluation test dataset"
    ):
        # Prediction
        text = test_doc.text
        ner_doc = ner_nlp(text)
        spancat_doc = spancat_nlp(text)

        ner_doc.spans[span_key] = list(ner_doc.ents)

        doc_eval = {
            "doc": test_doc.text,
            "ner_spans": [],
            "spancat_spans": [],
            "reference_spans": [],
        }

        # Check for True Positives and False Negatives
        for test_span in test_doc.spans[span_key]:
            doc_eval["reference_spans"].append(test_span)
            KPI[test_span.label_]["total_spans"] += 1
            eval = {
                "span": test_span,
                "ner_correct_indices": False,
                "ner_correct_label": False,
                "spancat_correct_indices": False,
                "spancat_correct_label": False,
            }
            ner_found = False
            spancat_found = False

            # NER evaluation
            for ner_span in ner_doc.spans[span_key]:
                if test_span.start == ner_span.start and test_span.end == ner_span.end:
                    ner_found = True
                    eval["ner_correct_indices"] = True
                    if test_span.label_ == ner_span.label_:
                        eval["ner_correct_label"] = True
                        doc_eval["ner_spans"].append((ner_span, True))
                        KPI[test_span.label_]["correct_ner_spans"] += 1
                        ner_scorer[test_span.label_].tp += 1
                    else:
                        ner_scorer[test_span.label_].fn += 1
                        ner_found = False
                    break

            if not ner_found:
                ner_scorer[test_span.label_].fn += 1

            # Spancat evaluation
            for spancat_span in spancat_doc.spans[span_key]:
                if (
                    test_span.start == spancat_span.start
                    and test_span.end == spancat_span.end
                ):
                    spancat_found = True
                    eval["spancat_correct_indices"] = True
                    if test_span.label_ == spancat_span.label_:
                        eval["spancat_correct_label"] = True
                        doc_eval["spancat_spans"].append((spancat_span, True))
                        KPI[test_span.label_]["correct_spancat_spans"] += 1
                        spancat_scorer[test_span.label_].tp += 1
                    else:
                        spancat_scorer[test_span.label_].fn += 1
                        spancat_found = False
                    break

            if not spancat_found:
                spancat_scorer[test_span.label_].fn += 1

            eval_list.append(eval)

        # Check for False Positives from the NER
        for ner_span in ner_doc.spans[span_key]:
            test_found = False
            for test_span in test_doc.spans[span_key]:
                if (
                    test_span.start == ner_span.start
                    and test_span.end == ner_span.end
                    and test_span.label_ == ner_span.label_
                ):
                    test_found = True
                    break
            if not test_found:
                ner_scorer[ner_span.label_].fp += 1
                doc_eval["ner_spans"].append((ner_span, False))

        # Check for False Positives from the Spancat
        for spancat_span in spancat_doc.spans[span_key]:
            test_found = False
            for test_span in test_doc.spans[span_key]:
                if (
                    test_span.start == spancat_span.start
                    and test_span.end == spancat_span.end
                    and test_span.label_ == spancat_span.label_
                ):
                    test_found = True
                    break
            if not test_found:
                spancat_scorer[spancat_span.label_].fp += 1
                doc_eval["spancat_spans"].append((spancat_span, False))

        doc_eval_list.append(doc_eval)

    msg.good("Evaluation successful")

    # Table Config
    header = ("Label", "F-Score", "Recall", "Precision")

    # NER Table
    ner_data = []
    ner_fscore = 0
    ner_recall = 0
    ner_precision = 0

    for label in ner_scorer:
        ner_data.append(
            (
                label,
                round(ner_scorer[label].fscore, 2),
                round(ner_scorer[label].recall, 2),
                round(ner_scorer[label].precision, 2),
            )
        )
        ner_fscore += ner_scorer[label].fscore
        ner_recall += ner_scorer[label].recall
        ner_precision += ner_scorer[label].precision

    ner_fscore /= len(ner_scorer)
    ner_recall /= len(ner_scorer)
    ner_precision /= len(ner_scorer)

    ner_data.append(
        ("Average", round(ner_fscore, 2), round(ner_recall, 2), round(ner_precision, 2))
    )

    msg.divider("NER performance")
    print(table(ner_data, header=header, divider=True))

    print()

    # Spancat Table
    spancat_data = []
    spancat_fscore = 0
    spancat_recall = 0
    spancat_precision = 0

    for label in spancat_scorer:
        spancat_data.append(
            (
                label,
                round(spancat_scorer[label].fscore, 2),
                round(spancat_scorer[label].recall, 2),
                round(spancat_scorer[label].precision, 2),
            )
        )
        spancat_fscore += spancat_scorer[label].fscore
        spancat_recall += spancat_scorer[label].recall
        spancat_precision += spancat_scorer[label].precision

    spancat_fscore /= len(spancat_scorer)
    spancat_recall /= len(spancat_scorer)
    spancat_precision /= len(spancat_scorer)

    spancat_data.append(
        (
            "Average",
            round(spancat_fscore, 2),
            round(spancat_recall, 2),
            round(spancat_precision, 2),
        )
    )

    msg.divider("Spancat performance")
    print(table(spancat_data, header=header, divider=True))

    print()

    # Compare performance between NER and Spancat architecture
    kpi_header = ("Label", "Total Spans", "Correct NER", "Correct Spancat")
    kpi_data = []

    total_spans = 0
    correct_ner_spans = 0
    correct_spancat_spans = 0

    for label in KPI:
        kpi_data.append(
            (
                label,
                KPI[label]["total_spans"],
                f"{KPI[label]['correct_ner_spans']} ({round((KPI[label]['correct_ner_spans']/KPI[label]['total_spans'])*100,2)}%)",
                f"{KPI[label]['correct_spancat_spans']} ({round((KPI[label]['correct_spancat_spans']/KPI[label]['total_spans'])*100,2)}%)",
            )
        )

        total_spans += KPI[label]["total_spans"]
        correct_ner_spans += KPI[label]["correct_ner_spans"]
        correct_spancat_spans += KPI[label]["correct_spancat_spans"]

    kpi_data.append(
        (
            "Total",
            total_spans,
            f"{correct_ner_spans} ({round((correct_ner_spans/total_spans)*100,2)}%)",
            f"{KPI[label]['correct_spancat_spans']} ({round((correct_spancat_spans/total_spans)*100,2)}%)",
        )
    )

    msg.divider("NER vs Spancat")
    print(table(kpi_data, header=kpi_header, divider=True))

    # Writes logging file that directly compares NER and Spancat

    log_file = ""
    # Logging where either NER or Spancat predicted spans correctly while the other model didn't
    log_file += "\n----------NER vs Spancat---------- \n"
    log_file += "Showing where either NER or Spancat predicted spans correctly while the other model didn't \n\n"

    for i, eval in enumerate(eval_list):
        if (eval["ner_correct_label"] and not eval["spancat_correct_label"]) or (
            not eval["ner_correct_label"] and eval["spancat_correct_label"]
        ):
            log_file += f"({i}) {eval['span']} [{eval['span'].label_}] | NER {emoji_return(eval['ner_correct_label'])}  | Spancat {emoji_return(eval['spancat_correct_label'])} \n"

    log_file += "\n"

    # Logging every Doc and the predictions of NER and Spancat
    log_file += "\n----------Spans per doc---------- \n"
    log_file += "Showing every doc and the predictions of NER and Spancat \n\n"

    for i, eval in enumerate(doc_eval_list):
        log_file += f"{i} {eval['doc']} \n"
        log_file += f"-----Reference spans----- \n"
        for reference_span in eval["reference_spans"]:
            log_file += f" > {reference_span.text} ({reference_span.start},{reference_span.end}) [{reference_span.label_}] \n"
        log_file += f"-----NER spans----- \n"
        for ner_span in eval["ner_spans"]:
            log_file += f" {emoji_return(ner_span[1])} {ner_span[0].text} ({ner_span[0].start},{ner_span[0].end}) [{ner_span[0].label_}] \n"
        log_file += f"-----Spancat spans----- \n"
        for spancat_span in eval["spancat_spans"]:
            log_file += f" {emoji_return(spancat_span[1])} {spancat_span[0]} ({spancat_span[0].start},{spancat_span[0].end}) [{spancat_span[0].label_}] \n"
        log_file += "\n----------\n"

    with open(
        "./metrics/evaluation_ner_vs_spancat.txt", "w", encoding="UTF-8"
    ) as writer:
        writer.write(log_file)


def emoji_return(emoji_bool: bool):
    if emoji_bool:
        return "✔️"
    else:
        return "❌"


if __name__ == "__main__":
    typer.run(main)

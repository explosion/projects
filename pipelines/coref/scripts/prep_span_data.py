"""
Kinda slow process mainly bottleneck by the
prediction speed of the pipeline with the
CorefScorer.
"""

import tqdm
import spacy
import argparse
from spacy.training import Example
from spacy.tokens import Doc, DocBin


parser = argparse.ArgumentParser(description="Create data set for SpanPredictor.")
parser.add_argument("--input-path", help="Path to the data set for the CorefScorer.")
parser.add_argument(
    "--output-path", help="Path to store the data set for the SpanPredictor."
)
parser.add_argument(
    "--model-path", help="Path to the trained pipeline with CorefScorer."
)
parser.add_argument(
    "--head-prefix", help="Prefix in the doc.spans used to store head-clusters."
)
parser.add_argument(
    "--span-prefix", help="Prefix in the doc.spans used to store span-clusters."
)
parser.add_argument("--limit", type=int, help="Number of documents to process..")
parser.add_argument(
    "--gpu", action="store_true", help="Run coreference pipeline on GPU."
)

args = parser.parse_args()
if args.gpu:
    spacy.require_gpu()
nlp = spacy.load(args.model_path)


def find_target_span(head, ex):
    """
    Take the smallest enclosing gold-span as
    the target for each predicted head.
    """
    # FIXME
    # Its really slow because it goes through all
    # clusters in gold_doc since I wasn't sure whether
    # its true that head from the predicted coref_head_clusters_1
    # would necessarily correspond to spans in coref_clusters_1
    smallest = float("inf")
    target_span = None
    for name, span_group in ex.reference.spans.items():
        if name.startswith(args.span_prefix):
            aligned_spans = ex.get_aligned_spans_y2x(span_group)
            for span in aligned_spans:
                if span.start <= head.start and head.end <= span.end:
                    size = span.end - span.start
                    if size == 1:
                        # won't get smaller, short ciruit
                        return target_span
                    if size < smallest:
                        target_span = span
                        smallest = size

    return target_span


docbin = DocBin().from_disk(args.input_path)
output_docbin = DocBin()
docs = docbin.get_docs(nlp.vocab)
input_head_clusters = {}
target_span_clusters = {}
total_heads = 0
duplicate_heads = 0
kept_heads = 0
empty_clusters = 0
num_docs = 0
skipped_docs = 0


for i, gold_doc in enumerate(tqdm.tqdm(docs)):
    if i == args.limit:
        break
    else:
        num_docs += 1
    # Predict head-clusters first.
    if not gold_doc.text:
        # TODO find out why this happens, there should be no empties
        # XXX note that this should not be an issue with latest spaCy master
        continue
    processed_doc = nlp(gold_doc.text)
    # Create a new Doc based on the coref-pipeline tokens and spaces.
    new_doc = Doc(
        nlp.vocab,
        words=[word.text for word in processed_doc],
        spaces=[bool(word.whitespace_) for word in processed_doc],
    )
    # Example helps with alignment
    ex = Example(predicted=processed_doc, reference=gold_doc)
    seen_heads = set()
    # Try to find target spans for all predicted heads.
    for name, head_group in ex.predicted.spans.items():
        cluster_id = name.split("_")[-1]
        if name.startswith(args.head_prefix):
            new_head_spangroup = []
            new_span_spangroup = []
            spans_name = f"{args.span_prefix}_{cluster_id}"
            for head in head_group:
                total_heads += 1
                # Only one sample per head for the SpanPredictor
                if (head.start, head.end) not in seen_heads:
                    seen_heads.add((head.start, head.end))
                    # Find the shortest enclosing span if exists
                    target_span = find_target_span(head, ex)
                    if target_span:
                        kept_heads += 1
                        new_head_spangroup.append(new_doc[head.start : head.end])
                        new_span_spangroup.append(
                            new_doc[target_span.start : target_span.end]
                        )
                else:
                    duplicate_heads += 1
        new_doc.spans[name] = new_head_spangroup
        new_doc.spans[spans_name] = new_span_spangroup
    if new_doc.spans:
        output_docbin.add(new_doc)
    else:
        skipped_docs += 1

print(f"Processed {num_docs} documents and skipped {skipped_docs}")
print(f"Found {total_heads} heads with {duplicate_heads} duplicates")
print(f"Found target spans for {kept_heads} heads.")

output_docbin.to_disk(args.output_path)

"""
Kinda slow process mainly bottleneck by the
prediction speed of the pipeline with the
CorefClusterer.
"""

import tqdm
import spacy
import argparse
from spacy.training import Example
from spacy.tokens import Doc, DocBin


parser = argparse.ArgumentParser(description="Create data set for SpanResolver.")
parser.add_argument("--input-path", help="Path to the data set for the CorefClusterer.")
parser.add_argument(
    "--output-path", help="Path to store the data set for the SpanResolver."
)
parser.add_argument(
    "--model-path", help="Path to the trained pipeline with CorefClusterer."
)
parser.add_argument(
    "--head-prefix", help="Prefix in the doc.spans used to store head-clusters."
)
parser.add_argument(
    "--span-prefix", help="Prefix in the doc.spans used to store span-clusters."
)
parser.add_argument("--limit", type=int, help="Number of documents to process.")
parser.add_argument(
    "--heads",
    type=str,
    default="silver",
    help="Whether to use gold heads or silver heads predicted by the clustering component",
)
parser.add_argument(
    "--gpu", type=int, default=-1, help="ID of GPU to run coreference pipeline on."
)

args = parser.parse_args()


def find_target_span(head, ex):
    """
    Take the smallest enclosing gold-span as
    the target for each predicted head.
    """

    # Note: This is slow because it doesn't assume that the smallest enclosing
    # span for a given token will necessarily be in the matching spangroup. For
    # example, a token could be in word-level cluster 1, but the smallest
    # enclosing span could be in cluster 2. When using word-level tokens
    # predicted by a trained model, like in this script, you have to check
    # because there's no guarantee tokens and spans will align.

    # If you know your spangroups are aligned, you could just check the
    # matching spangroup. However that still wouldn't guarantee you get the
    # smallest enclosing span, since it is possible (though rare/unlikely) for
    # the same head to have different spans associated with it in different
    # spangroups.

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
                        return span
                    if size < smallest:
                        target_span = span
                        smallest = size

    return target_span


if args.gpu > -1:
    spacy.require_gpu(args.gpu)
nlp = spacy.load(args.model_path)
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

# Gold heads can be used directly from the input data, or heads predicted by
# the clustering component can be used. In tests with OntoNotes, accuracy and
# processing time were similar for both options. However, silver heads avoid
# tokenization mismatches, so they are chosen as the default.
if args.heads not in ["silver", "gold"]:
    raise ValueError(f"Expected 'gold' or 'silver' for heads but got: {args.heads}")
use_gold_heads = False
if args.heads == "gold":
    use_gold_heads = True


for i, gold_doc in enumerate(tqdm.tqdm(docs)):
    if i == args.limit:
        break
    else:
        num_docs += 1
    # Predict head-clusters first.
    if use_gold_heads:
        # use the pipeline just for tokenization
        processed_doc = nlp.make_doc(gold_doc.text)
        # copy over gold heads
        ex = Example(predicted=processed_doc, reference=gold_doc)
        for name, sg in ex.reference.spans.items():
            if not name.startswith(args.head_prefix):
                continue
            processed_doc.spans[name] = ex.get_aligned_spans_y2x(sg)
    else:
        # use predictions
        processed_doc = nlp(gold_doc.text)
        ex = Example(predicted=processed_doc, reference=gold_doc)

    # Create a new Doc based on the coref-pipeline tokens and spaces.
    # This will go in the output DocBin.
    new_doc = Doc(
        nlp.vocab,
        words=[word.text for word in processed_doc],
        spaces=[bool(word.whitespace_) for word in processed_doc],
    )
    # Example helps with alignment
    seen_heads = set()
    # Try to find target spans for all predicted heads.
    for name, head_group in ex.predicted.spans.items():
        cluster_id = name.split("_")[-1]
        if not name.startswith(args.head_prefix):
            continue
        new_head_spangroup = []
        new_span_spangroup = []
        spans_name = f"{args.span_prefix}_{cluster_id}"
        for head in head_group:
            total_heads += 1
            # Only one sample per head for the SpanResolver
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

import spacy
import polar_pipe
import csv
import textwrap


def create_polar_pipeline():
    """Create a pipeline with a polar component."""
    # We only need vectors
    nlp = spacy.load(
        "en_core_web_md", disable=["tok2vec", "parser", "tagger", "ner", "lemmatizer"]
    )

    polar = nlp.add_pipe("polar")

    # now add some axes
    polar.add_axis("science", "magic")
    polar.add_axis("serious", "funny")
    polar.add_axis("heartwarming", "creepy")
    polar.add_axis("bad", "good")
    return nlp

def format_review(text, truncate=3):
    lines = textwrap.wrap(text)
    return '\n'.join(lines[:truncate])

def check_axes(nlp, nn=3):
    """Given a pipeline with a polar component, check the IMDB dataset."""
    docs = []

    with open("assets/IMDB Dataset.csv") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            docs.append(nlp(row[0]))

            # XXX Limit for faster testing iteration
            if len(docs) > 10000:
                break

    # now print the reviews at polar extremes
    polar = nlp.get_pipe("polar")
    for axis in polar.axes:
        docs.sort(key=lambda x: x._.poles[axis.get_key()])
        print(f"Top {nn} {axis.neg} reviews:")
        print()
        for ii in range(nn):
            doc = docs[ii]
            print(format_review(doc.text))
            print()
        print()
        print(f"Top {nn} {axis.pos} reviews:")
        print()
        for ii in range(nn):
            doc = docs[(ii + 1) * -1]
            print(format_review(doc.text))
            print()
        print("=" * 70)


def main():
    nlp = create_polar_pipeline()
    check_axes(nlp, 3)


if __name__ == "__main__":
    main()

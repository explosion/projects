import spacy
from . import suggester

def test_subtree_suggester():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Slice the lemon and take the chocolate cake to the fridge.")
    noun_suggester = suggester.build_noun_suggester()
    candidates = noun_suggester([doc])

    for span_indices in candidates.data:
        print(doc[span_indices[0]:span_indices[1]])

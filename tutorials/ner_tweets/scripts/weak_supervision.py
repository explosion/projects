import json
from pathlib import Path
from typing import Dict

import skweak
from skweak.base import CombinedAnnotator, SpanAnnotator
from skweak.gazetteers import GazetteerAnnotator, extract_json_data
from skweak.heuristics import SpanConstraintAnnotator, TokenConstraintAnnotator
from skweak.spacy import ModelAnnotator
from spacy.tokens import Doc, Span
from wasabi import msg

from .constants import (
    BTC_MODEL_PATH,
    CRUNCHBASE_PATH,
    NAME_PREFIXES,
    NAMES_PATH,
    WIKIDATA_PATH,
)


class UnifiedNERAnnotator(CombinedAnnotator):
    """An annotator that combines all labelling functions we have"""

    MODEL_ANNOTATORS = {"en_core_web_lg": "en_core_web_lg", "btc": BTC_MODEL_PATH}
    GAZTR_ANNOTATORS = {"wikidata": WIKIDATA_PATH, "crunchbase": CRUNCHBASE_PATH}

    def add_all_annotators(self):
        """Add all previously defined annotators"""
        self.add_model_annotators()
        self.add_gazetteer_annotators()
        self.add_heuristic_annotators()
        self.add_standardizer()

    def add_model_annotators(self, models: Dict = MODEL_ANNOTATORS):
        """Add annotators based on statistical models"""
        for name, model in models.items():
            self.add_annotator(ModelAnnotator(name=name, model_path=model))
            msg.good(f"Added model annotator: {model}")
        self._show_number_of_annotators()

    def add_gazetteer_annotators(self, sources: Dict = GAZTR_ANNOTATORS):
        """Add annotators based on list of entities / words"""
        for name, source in sources.items():
            tries = extract_json_data(str(source), spacy_model="en_core_web_lg")
            # We make two annotators for each: cased and uncased
            self.add_annotator(GazetteerAnnotator(name + "_uncased", tries, False))
            self.add_annotator(GazetteerAnnotator(name + "_cased", tries, True))
            msg.good(f"Added gazetteer from source: {source}")
        self._show_number_of_annotators()

    def add_heuristic_annotators(self):
        """Add annotators based on business rules and common heuristics"""

        # Add annotator for proper nouns
        proper_noun_annotator = TokenConstraintAnnotator(
            "proper", skweak.utils.is_likely_proper, "ENT"
        )

        # Add annotator to check name prefixes: Von, Van, etc.
        prefix_annotator = TokenConstraintAnnotator(
            "prefix", skweak.utils.is_likely_proper, "ENT"
        )
        prefix_annotator.add_gap_tokens(NAME_PREFIXES)

        # Let's combine the first two
        proper_names_annotator = CombinedAnnotator()
        for annotator in [proper_noun_annotator, prefix_annotator]:
            annotator.add_gap_tokens(["'s'", "-"])
            proper_names_annotator.add_annotator(annotator)
        self.add_annotator(proper_names_annotator)
        msg.good(f"Added proper names annotator")

        # Add fullname detector
        self.add_annotator(
            SpanConstraintAnnotator(
                "full_name",
                other_name="proper",
                constraint=FullNameDetector(),
                label="PERSON",
            )
        )

        self._show_number_of_annotators()

    def add_standardizer(self):
        self.add_annotator(Standardizer())
        msg.good(f"Added Standardizer")
        self._show_number_of_annotators()

    def _show_number_of_annotators(self):
        msg.text(f"Current number of annotators: {len(self.annotators)}")


class FullNameDetector:
    """Custom annotator that search for occurences of full-names"""

    def __init__(self, names_path: Path = NAMES_PATH):
        with names_path.open(mode="r") as f:
            self.first_names = set(json.load(f))

    def __call__(self, span: Span) -> bool:
        """Return if a particular span is a full name or not"""
        if len(span) < 2 or len(span) > 5:
            return False

        # Encode business rules more explicitly
        has_common_first_name = span[0].text in self.first_names
        has_likely_surname = span[-1].is_alpha and span[-1].is_title
        return has_common_first_name and has_likely_surname


class Standardizer(SpanAnnotator):
    """Standardize label entities

    We approach standardization as another rule-based annotator based on a Span's
    label. We then append a new label for that.
    """

    def __init__(self):
        super(Standardizer, self).__init__("")

    def __call__(self, doc: Doc):
        for source in doc.spans:
            new_spans = []
            for span in doc.spans[source]:
                if "\n" in span.text:
                    continue
                elif span.label_ == "PER":
                    new_spans.append(Span(doc, span.start, span.end, label="PER"))
                elif span.label_ in ["ORGANIZATION", "ORGANISATION", "COMPANY"]:
                    new_spans.append(Span(doc, span.start, span.end, label="ORG"))
                elif span.label_ in ["GPE"]:
                    new_spans.append(Span(doc, span.start, span.end, label="LOC"))
                # fmt: off
                elif span.label_ in ["EVENT", "FAC", "LANGUAGE", "LAW", "NORP", "PRODUCT", "WORK_OF_ART"]:
                # fmt: on
                    new_spans.append(Span(doc, span.start, span.end , label="MISC"))
                else:
                    new_spans.append(span)
            doc.spans[source] = new_spans
        return doc

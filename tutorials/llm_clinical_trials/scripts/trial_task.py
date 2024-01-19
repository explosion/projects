from typing import Iterable

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from spacy_llm.registry import registry
from spacy_llm.ty import LLMTask


INSTRUCTION = """
Summarize the trial results in a structured fashion.
First, identify all patient groups with distinct treatments. 
Then, for each patient group, write down the following:

Patient group: <name>
Number of patients in the group: <number>
Treatment drug or substance: <drug>
Treatment dose: <dose>
Treatment frequency of administration: <frequency>
Treatment duration: <duration>
Outcome: <outcome>
"""


@registry.llm_tasks("tutorial.TrialSummary.v1")
def make_trial_task() -> "TrialSummaryTask":
    return TrialSummaryTask(INSTRUCTION)


class TrialSummaryTask(LLMTask):
    def __init__(self, instruction: str):
        self.instruction = instruction
        Doc.set_extension("trial_summary", default="")

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        for doc in docs:
            yield self.generate_prompt(doc)

    def generate_prompt(self, doc: Doc) -> str:
        prompt = "Below this instruction, I will provide you with a clinical trial abstract. \n"
        prompt += self.instruction + "\n\n" + doc.text
        return prompt

    def parse_responses_v1(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, response in zip(docs, responses):
            doc._.trial_summary = response
            yield doc

    # quick and dirty implementation for now
    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, response in zip(docs, responses):
            response_lower = response.lower()

            patient_groups = []
            patient_numbers = []
            drugs = []
            doses = []
            frequencies = []
            durations = []
            outcomes = []

            start = response_lower.find("patient group:")
            while start >= 0:
                patient_group_start = response_lower.find("patient group:", start)
                patient_group_end = patient_group_start + len("patient group:")

                patient_number_start = response_lower.find("number of patients in the group:", start)
                patient_number_end = patient_number_start + len("number of patients in the group:")

                treatment_drug_start = response_lower.find("treatment drug or substance:", start)
                treatment_drug_end = treatment_drug_start + len("treatment drug or substance:")

                treatment_dose_start = response_lower.find("treatment dose:", start)
                treatment_dose_end = treatment_dose_start + len("treatment dose:")

                treatment_frequency_start = response_lower.find("treatment frequency of administration:", start)
                treatment_frequency_end = treatment_frequency_start + len("treatment frequency of administration:")

                treatment_duration_start = response_lower.find("treatment duration:", start)
                treatment_duration_end = treatment_duration_start + len("treatment duration:")

                outcome_start = response_lower.find("outcome:", start)
                outcome_end = outcome_start + len("outcome:")

                patient_group = response[patient_group_end:patient_number_start].strip()
                patient_groups.append(patient_group)

                patient_number = response[patient_number_end:treatment_drug_start].strip()
                patient_numbers.append(patient_number)

                treatment_drug = response[treatment_drug_end:treatment_dose_start].strip()
                drugs.append(treatment_drug)

                treatment_dose = response[treatment_dose_end:treatment_frequency_start].strip()
                doses.append(treatment_dose)

                treatment_frequency = response[treatment_frequency_end:treatment_duration_start].strip()
                frequencies.append(treatment_frequency)

                treatment_duration = response[treatment_duration_end:outcome_start].strip()
                durations.append(treatment_duration)

                start = response_lower.find("patient group:", outcome_end)

                outcome = response[outcome_end:start].strip()
                outcomes.append(outcome)

            matcher = PhraseMatcher(doc.vocab, attr="LOWER")
            nlp = spacy.blank("en")
            matcher.add("Patient_Group", [nlp.make_doc(text) for text in patient_groups])
            matcher.add("Patient_Number", [nlp.make_doc(text) for text in patient_numbers])
            matcher.add("Treatment_Drug", [nlp.make_doc(text) for text in drugs])
            matcher.add("Treatment_Dose", [nlp.make_doc(text) for text in doses])
            matcher.add("Treatment_Frequency", [nlp.make_doc(text) for text in frequencies])
            matcher.add("Treatment_Duration", [nlp.make_doc(text) for text in durations])
            matcher.add("Outcome", [nlp.make_doc(text) for text in outcomes])

            matches = matcher(doc, as_spans=True)
            matches = spacy.util.filter_spans(matches)

            # This assumes that no entities were set prior to this component
            doc.ents = matches
            doc._.trial_summary = response
            yield doc

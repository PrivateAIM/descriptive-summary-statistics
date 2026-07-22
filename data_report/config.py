"""Shared configuration constants for the HALTA federated data-report pipeline.

This module centralises values that govern column detection and outcome
selection across the pipeline:

* ``PASC_SYMPTOM_KEYWORDS`` — substrings used by ``label.py`` to identify
  symptom-flag columns when generating synthetic PASC labels.
* ``LABEL_COL`` — the expected name of the PASC outcome column.
* ``OUTCOME_KEYWORD_GROUPS`` — priority-ordered keyword groups used by the
  inferential analysis module to auto-detect the outcome column in
  arbitrary hospital datasets.
"""

from typing import List

PASC_SYMPTOM_KEYWORDS: List[str] = [
    "fatigue", "tired", "dyspnea", "shortness_of_breath", "breath", "cough",
    "anosmia", "ageusia", "smell", "taste", "headache", "migraine",
    "brain", "fog", "cognitive", "memory", "concentration",
    "chest_pain", "chest", "palpit", "tachy", "arrhythm",
    "sleep", "insomnia", "anxiety", "depress", "mood",
    "myalgia", "muscle", "joint", "pain", "arthral",
    "fever", "nausea", "diarr", "gi_",
]

LABEL_COL = "pasc"

# Generic outcome-column detection for the inferential statistics section.
# Hospital datasets on the hub vary in naming, so candidates are matched by
# keyword in priority order (mortality/survival ranks above demographics-like
# labels) and then validated for usability (not constant, low cardinality,
# enough observations per class) before being accepted -- see
# detect_outcome_column in inferential_analysis.py.
OUTCOME_KEYWORD_GROUPS: List[List[str]] = [
    ["death", "died", "deceased", "mortality", "survival"],
    ["icu", "intensive_care", "ventilation", "intubation"],
    ["readmission", "re-admission", "admission"],
    ["outcome", "status", "diagnosis", "condition", "label", "pasc"],
    ["complication", "adverse"],
]

"""Rhetorical role label set used by LegalSeg (github.com/ShubhamKumarNigam/LegalSeg)."""

ID2LABEL = {
    0: "Facts",
    1: "Issue",
    2: "Argument of Petitioner",
    3: "Argument of Respondent",
    4: "Reasoning",
    5: "Decision",
    6: "None",
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}

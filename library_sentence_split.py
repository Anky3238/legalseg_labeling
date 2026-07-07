"""Sentence splitting using pysbd (open-source Golden Rules-based sentence
boundary disambiguation), as an open-source-library alternative to this
project's custom regex-based splitter in sentence_split.py.

pysbd ships a general-English abbreviation list that misses several Indian
legal titles (e.g. "Sr." in "Sr. Advocate", "Adv.", "Hon'ble" variants),
causing it to split mid-phrase. Its abbreviation-before-name handling is
unconditional for any abbreviation in PREPOSITIVE_ABBREVIATIONS (it never
splits after one of those, regardless of what follows), so we extend that
list with the legal titles this corpus uses. This is the officially
supported way to customize pysbd's English rules -- no forking required.
"""
import pysbd
from pysbd.lang.english import English

_LEGAL_PREPOSITIVE_ABBREVIATIONS = ["sr", "jr", "adv", "ld", "smt", "shri", "hon"]
for _abbr in _LEGAL_PREPOSITIVE_ABBREVIATIONS:
    if _abbr not in English.Abbreviation.PREPOSITIVE_ABBREVIATIONS:
        English.Abbreviation.PREPOSITIVE_ABBREVIATIONS.append(_abbr)

_segmenter = pysbd.Segmenter(language="en", clean=False)


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _segmenter.segment(text) if s.strip()]


if __name__ == "__main__":
    sample = (
        "The Learned Counsel for the Appellant contends that the 'Adjudicating "
        "Authority' had failed to appreciate that the 'Corporate Applicant' should "
        "have remained as a 'Going Concern' under the I & B Code with a view to "
        "arrive at a suitable Resolution Plan. Mr. P. Mahalingam, Sr. Advocate, "
        "appeared for the Appellant. The order dated 17.05.2022 was passed by the "
        "Adjudicating Authority. It is stated in the I.A. that curing the defects "
        "took some time. Hon'ble Supreme Court held in Duncans Industries Ltd. "
        "v. A.J. Agrochem that the Code is a beneficial legislation."
    )
    for s in split_sentences(sample):
        print("-", s)

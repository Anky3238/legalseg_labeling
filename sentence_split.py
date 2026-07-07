"""Abbreviation-aware sentence splitter tuned for Indian legal judgment text.

A generic sentence splitter (incl. pysbd's default English model) breaks on
domain abbreviations common in these documents ("Sr.", "Adv.", "Ld.", "Ors.",
"I.A.", etc.), splitting names and citations mid-phrase. This checks the word
immediately before each candidate '.'/'?'/'!' against a curated abbreviation
list and against the single-capital-letter-initial pattern ("P." in "Dr. P.
Mahalingam") before accepting the split.
"""
import re

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "sr", "jr", "smt", "shri", "km", "adv", "advs",
    "ld", "hon", "no", "nos", "sec", "secs", "art", "arts", "cl", "cls",
    "para", "paras", "ord", "ords", "ltd", "pvt", "co", "corp", "corpn",
    "vs", "v", "ors", "anr", "govt", "rs", "regn", "addl", "asstt", "supdt",
    "dy", "jt", "genl", "distt", "st", "nd", "rd", "th", "etc", "viz",
    "ia", "ias", "ma", "mas", "cp", "ca", "oa", "rp", "wp", "slp", "crl",
    "misc", "appln", "appl", "u.s", "i.e", "e.g", "cf", "op", "resp",
    "petr", "respdt", "appdt", "unctd", "hon'ble",
}

_ROMAN_NUMERAL_RE = re.compile(r"^[ivxlcdm]+$", re.I)
_SPLIT_CANDIDATE_RE = re.compile(r'[.!?]+(?=\s+[A-Z"‘’“”(])')
_WORD_BEFORE_RE = re.compile(r"([A-Za-z]+)$")


def _preceding_word_blocks_split(text: str, punct_end: int) -> bool:
    before = text[:punct_end]
    m = _WORD_BEFORE_RE.search(before)
    if not m:
        return False
    word = m.group(1)
    if len(word) == 1:  # single-letter initial, e.g. "P."
        return True
    if word.lower() in ABBREVIATIONS:
        return True
    if _ROMAN_NUMERAL_RE.match(word):  # "(iv)." style sub-clause markers
        return True
    return False


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = []
    start = 0
    for m in _SPLIT_CANDIDATE_RE.finditer(text):
        if _preceding_word_blocks_split(text, m.start(0)):
            continue
        split_at = m.start(0) + len(m.group(0))
        sentences.append(text[start:split_at].strip())
        start = m.end(0)

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)

    return [s for s in sentences if s]


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

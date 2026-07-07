"""End-to-end LegalSeg pipeline: clean raw OCR/PyMuPDF judgment text, split
into paragraphs, and label each paragraph's sentences with InLegalBERT
rhetorical roles.

Usage:
    python run_pipeline.py --input judgment_full_texts_ankit --output out.jsonl
    python run_pipeline.py --input judgment_full_texts_ankit --output out.jsonl --limit 5
"""
import argparse
import json

import pandas as pd
from tqdm import tqdm

from infer import RhetoricalRoleClassifier
from preprocess import clean_and_paragraphize


def process_document(
    judgment_id: str, raw_text: str, classifier: RhetoricalRoleClassifier, batch_size: int = 16
) -> dict:
    paragraphs = clean_and_paragraphize(raw_text)
    labeled_paragraphs = []
    for p in paragraphs:
        result = classifier.classify_paragraph(p["text"], batch_size=batch_size)
        labeled_paragraphs.append(
            {
                "para_no": p["para_no"],
                "text": p["text"],
                "paragraph_label": result["paragraph_label"],
                "label_counts": result["label_counts"],
                "sentences": result["sentences"],
            }
        )
    return {"judgment_id": judgment_id, "paragraphs": labeled_paragraphs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with judgment_id, full_text columns")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    classifier = RhetoricalRoleClassifier()

    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Labelling judgments"):
            doc_result = process_document(
                row["judgment_id"], str(row["full_text"]), classifier, batch_size=args.batch_size
            )
            f.write(json.dumps(doc_result, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df)} labelled documents to {args.output}")


if __name__ == "__main__":
    main()

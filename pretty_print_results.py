"""Convert a results JSONL file (one compact JSON object per document) into a
human-readable text file: one document per section, one paragraph per block,
labelled clearly, for browsing in an editor instead of parsing raw JSON lines.

Usage:
    python pretty_print_results.py --input results.jsonl --output results_readable.txt
    python pretty_print_results.py --input results.jsonl --output results_readable.txt --sentences
"""
import argparse
import json


def format_document(doc: dict, show_sentences: bool) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append(f"DOCUMENT: {doc['judgment_id']}  ({len(doc['paragraphs'])} paragraphs)")
    lines.append("=" * 100)
    lines.append("")

    for p in doc["paragraphs"]:
        para_no = p["para_no"] if p["para_no"] is not None else "-"
        lines.append(f"[Para {para_no}] ({p['paragraph_label']})  counts={p['label_counts']}")
        lines.append(p["text"])
        if show_sentences:
            lines.append("  -- sentences --")
            for s in p["sentences"]:
                lines.append(f"    ({s['label']}, {s['confidence']:.2f}) {s['text']}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output readable .txt file")
    parser.add_argument(
        "--sentences", action="store_true", help="Also show per-sentence labels within each paragraph"
    )
    parser.add_argument("--judgment-id", default=None, help="Only output this one judgment_id")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        count = 0
        for line in fin:
            doc = json.loads(line)
            if args.judgment_id and doc["judgment_id"] != args.judgment_id:
                continue
            fout.write(format_document(doc, args.sentences))
            fout.write("\n\n")
            count += 1

    print(f"Wrote {count} documents to {args.output}")


if __name__ == "__main__":
    main()

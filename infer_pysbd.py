"""Sentence-level rhetorical-role inference identical to infer.py, except
paragraphs are split into sentences with pysbd (library_sentence_split.py)
instead of the project's custom regex splitter (sentence_split.py). Kept as
a separate module so the original regex-based pipeline (infer.py,
run_pipeline.py, results.jsonl) is untouched and the two approaches can be
compared side by side.
"""
import os
from collections import Counter

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

from labels import ID2LABEL
from library_sentence_split import split_sentences

BASE_MODEL = "law-ai/InLegalBERT"
CHECKPOINT_REPO = "L-NLProc/LegalSeg_InLegalBERT"
CHECKPOINT_FILE = "InLegalBERT/InLegalBERT(i)model.safetensors"
LOCAL_CHECKPOINT = os.path.join(os.path.dirname(__file__), "models", "InLegalBERT_i.safetensors")
MAX_LENGTH = 512


class RhetoricalRoleClassifierPysbd:
    def __init__(self, device: str | None = None, checkpoint_path: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        config = BertConfig.from_pretrained(BASE_MODEL, num_labels=len(ID2LABEL))
        self.model = BertForSequenceClassification(config)

        weights_path = checkpoint_path or self._resolve_checkpoint()
        state_dict = load_file(weights_path)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint did not match model architecture.\n"
                f"missing={missing}\nunexpected={unexpected}"
            )

        self.model.to(self.device)
        self.model.eval()

    def _resolve_checkpoint(self) -> str:
        if os.path.exists(LOCAL_CHECKPOINT):
            return LOCAL_CHECKPOINT
        return hf_hub_download(repo_id=CHECKPOINT_REPO, filename=CHECKPOINT_FILE)

    @torch.no_grad()
    def classify_sentences(self, sentences: list[str], batch_size: int = 16) -> list[dict]:
        results = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            confidences, pred_ids = probs.max(dim=-1)
            for pred_id, conf in zip(pred_ids.tolist(), confidences.tolist()):
                results.append({"label": ID2LABEL[pred_id], "confidence": round(conf, 4)})
        return results

    def classify_paragraph(self, paragraph_text: str, batch_size: int = 16) -> dict:
        sentences = split_sentences(paragraph_text)
        if not sentences:
            return {"sentences": [], "paragraph_label": None, "label_counts": {}}

        sentence_results = self.classify_sentences(sentences, batch_size=batch_size)
        for s, r in zip(sentences, sentence_results):
            r["text"] = s

        label_counts = Counter(r["label"] for r in sentence_results)
        paragraph_label = label_counts.most_common(1)[0][0]

        return {
            "sentences": sentence_results,
            "paragraph_label": paragraph_label,
            "label_counts": dict(label_counts),
        }

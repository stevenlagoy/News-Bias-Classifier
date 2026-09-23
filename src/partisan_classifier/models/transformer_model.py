"""
Fine-tuned transformer classifier (e.g. RoBERTa, DistilBERT).
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from partisan_classifier.models.base import PartisanClassifier

_LABEL2ID = {"left": 0, "center": 1, "right": 2}
_ID2LABEL = {v: k for k, v in _LABEL2ID.items()}


class TransformerClassifier(PartisanClassifier):
    def __init__(
        self,
        model_name: str = "roberta-base",
        output_dir: str = "outputs/transformer",
        epochs: int = 3,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3, id2label=_ID2LABEL, label2id=_LABEL2ID
        )

    def _prep(self, texts: list[str], labels: list[str] | None = None) -> Dataset:
        data = {"text": texts}
        if labels is not None:
            data["labels"] = [_LABEL2ID[l] for l in labels]
        ds = Dataset.from_dict(data)
        return ds.map(
            lambda x: self.tokenizer(x["text"], truncation=True, max_length=512),
            batched=True,
        )

    def fit(self, X: list[str], y: list[str]) -> "TransformerClassifier":
        train_ds = self._prep(X, y)
        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            save_strategy="epoch",
        )
        trainer = Trainer(model=self.model, args=args, train_dataset=train_ds)
        trainer.train()
        return self

    def predict(self, X: list[str]) -> list[str]:
        import torch

        ds = self._prep(X)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(ds), self.batch_size):
                batch = ds[i:i + self.batch_size]
                inputs = self.tokenizer.pad(
                    {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
                    return_tensors="pt",
                )
                logits = self.model(**inputs).logits
                preds += [_ID2LABEL[i] for i in logits.argmax(dim=-1).tolist()]
        return preds

    def predict_proba(self, X: list[str]) -> Any:
        raise NotImplementedError  # add if you need calibrated probabilities

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str) -> "TransformerClassifier":
        obj = cls.__new__(cls)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.batch_size = 8
        return obj
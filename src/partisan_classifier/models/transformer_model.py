"""
Fine-tuned transformer classifier (e.g. RoBERTa, DistilBERT).
"""

from __future__ import annotations

import json
import os
from typing import Any

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
        learning_rate: float = 2e-5,
        seed: int = 42,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label=_ID2LABEL,
            label2id=_LABEL2ID,
        )

    def _prep(
        self,
        texts: list[str],
        labels: list[str] | None = None,
    ):
        from datasets import Dataset

        data = {"text": texts}

        if labels is not None:
            data["labels"] = [_LABEL2ID[label] for label in labels]

        ds = Dataset.from_dict(data)

        return ds.map(
            lambda x: self.tokenizer(
                x["text"],
                truncation=True,
                max_length=512,
            ),
            batched=True,
        )

    def fit(
        self,
        X: list[str],
        y: list[str],
        X_val: list[str],
        y_val: list[str],
    ) -> "TransformerClassifier":
        from sklearn.metrics import f1_score
        from transformers import (
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

        train_ds = self._prep(X, y)
        val_ds = self._prep(X_val, y_val)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred

            if isinstance(predictions, tuple):
                predictions = predictions[0]

            predicted_labels = predictions.argmax(axis=-1)

            macro_f1 = f1_score(
                labels,
                predicted_labels,
                average="macro",
            )

            return {"macro_f1": macro_f1}

        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
            save_total_limit=1,
            seed=self.seed,
            data_seed=self.seed,
            report_to="none",
        )

        # Dynamically pad each batch to the longest sequence in that batch.
        # This avoids errors when examples have different tokenized lengths.
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

        trainer.train()

        # Record validation macro-F1 for every epoch.
        validation_history = []

        for record in trainer.state.log_history:
            if "eval_macro_f1" in record:
                validation_history.append(
                    {
                        "epoch": record.get("epoch"),
                        "validation_macro_f1": record["eval_macro_f1"],
                    }
                )

        os.makedirs(self.output_dir, exist_ok=True)

        metrics_path = os.path.join(
            self.output_dir,
            "validation_metrics.json",
        )

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(validation_history, f, indent=2)

        print("\nValidation macro-F1 by epoch:")

        for record in validation_history:
            print(
                f"Epoch {record['epoch']}: "
                f"{record['validation_macro_f1']:.4f}"
            )

        return self

    def predict(self, X: list[str]) -> list[str]:
        import torch

        ds = self._prep(X)

        self.model.eval()

        preds = []

        with torch.no_grad():
            for i in range(0, len(ds), self.batch_size):
                batch = ds[i : i + self.batch_size]

                inputs = self.tokenizer.pad(
                    {
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"],
                    },
                    return_tensors="pt",
                )

                # Make sure inputs are on the same device as the model.
                inputs = {
                    key: value.to(self.model.device)
                    for key, value in inputs.items()
                }

                logits = self.model(**inputs).logits

                preds += [
                    _ID2LABEL[i]
                    for i in logits.argmax(dim=-1).tolist()
                ]

        return preds

    def predict_proba(self, X: list[str]) -> Any:
        raise NotImplementedError  # add if you need calibrated probabilities

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str) -> "TransformerClassifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        obj = cls.__new__(cls)

        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        obj.tokenizer = AutoTokenizer.from_pretrained(path)

        obj.batch_size = 8

        return obj
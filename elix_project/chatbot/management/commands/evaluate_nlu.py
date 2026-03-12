"""Evaluate the chatbot intent classifier."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


def _load_jsonl_dataset(path: Path):
    texts = []
    intents = []

    with path.open("r", encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CommandError(
                    f"Invalid JSONL at line {line_number} in {path}: {exc}"
                ) from exc

            text = (payload.get("text") or "").strip()
            intent = (payload.get("intent") or "").strip()
            if not text or not intent:
                raise CommandError(
                    f"Each sample must include non-empty 'text' and 'intent' "
                    f"(line {line_number} in {path})."
                )

            texts.append(text)
            intents.append(intent)

    if not texts:
        raise CommandError(f"Dataset is empty: {path}")

    return texts, intents


class Command(BaseCommand):
    help = "Evaluate trained NLU model against a labeled JSONL dataset."

    def add_arguments(self, parser):
        base_dir = Path(settings.BASE_DIR)
        parser.add_argument(
            "--data",
            type=str,
            default=str(base_dir / "chatbot" / "nlu_data" / "intents_en_v1.jsonl"),
            help="Path to labeled JSONL dataset.",
        )
        parser.add_argument(
            "--model",
            type=str,
            default=str(getattr(settings, "NLU_MODEL_PATH")),
            help="Path to trained .joblib model artifact.",
        )
        parser.add_argument(
            "--metrics-output",
            type=str,
            default="",
            help="Optional path to write evaluation metrics JSON.",
        )

    def handle(self, *args, **options):
        try:
            import joblib
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        except ImportError as exc:
            raise CommandError(
                "Missing ML dependencies. Install requirements first "
                "(scikit-learn, numpy, joblib)."
            ) from exc

        data_path = Path(options["data"]).resolve()
        model_path = Path(options["model"]).resolve()
        metrics_output = options["metrics_output"].strip()
        metrics_output_path = Path(metrics_output).resolve() if metrics_output else None

        if not data_path.exists():
            raise CommandError(f"Dataset not found: {data_path}")
        if not model_path.exists():
            raise CommandError(f"Model not found: {model_path}")

        texts, intents = _load_jsonl_dataset(data_path)
        label_counts = Counter(intents)

        loaded_artifact = joblib.load(model_path)
        if isinstance(loaded_artifact, dict) and "pipeline" in loaded_artifact:
            pipeline = loaded_artifact["pipeline"]
            artifact_version = loaded_artifact.get("version", "unknown")
        else:
            pipeline = loaded_artifact
            artifact_version = "legacy"

        predictions = pipeline.predict(texts) # type: ignore[arg-type] # sklearn typing limitation
        report_dict = classification_report(
            intents, predictions, output_dict=True, zero_division=0
        )
        report_text = classification_report(intents, predictions, zero_division=0)
        accuracy = float(accuracy_score(intents, predictions))
        macro_f1 = float(report_dict["macro avg"]["f1-score"])# type: ignore[arg-type]  # sklearn typing limitation
        labels = sorted(label_counts.keys())
        confusion = confusion_matrix(intents, predictions, labels=labels).tolist()

        self.stdout.write(f"Model: {model_path} (version: {artifact_version})")
        self.stdout.write(f"Samples: {len(texts)}")
        self.stdout.write(f"Intent distribution: {dict(sorted(label_counts.items()))}")
        self.stdout.write("Evaluation report:")
        self.stdout.write(report_text)# type: ignore[arg-type]  # sklearn typing limitationv
        self.stdout.write(f"Macro F1: {macro_f1:.4f} | Accuracy: {accuracy:.4f}")

        if metrics_output_path:
            metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": artifact_version,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset_path": str(data_path),
                "model_path": str(model_path),
                "dataset_size": len(texts),
                "label_distribution": dict(sorted(label_counts.items())),
                "metrics": {
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                },
                "classification_report": report_dict,
                "confusion_matrix": {
                    "labels": labels,
                    "matrix": confusion,
                },
            }
            metrics_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"Saved metrics to {metrics_output_path}"))

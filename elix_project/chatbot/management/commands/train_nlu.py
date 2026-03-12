"""Train the chatbot intent classifier."""

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
    help = (
        "Train ML NLU model using TF-IDF (word + char n-grams) "
        "with a Logistic Regression classifier."
    )

    def add_arguments(self, parser):
        base_dir = Path(settings.BASE_DIR)
        parser.add_argument(
            "--data",
            type=str,
            default=str(base_dir / "chatbot" / "nlu_data" / "intents_en_v1.jsonl"),
            help="Path to JSONL training dataset.",
        )
        parser.add_argument(
            "--model-output",
            type=str,
            default=str(getattr(settings, "NLU_MODEL_PATH")),
            help="Path to output .joblib model artifact.",
        )
        parser.add_argument(
            "--metrics-output",
            type=str,
            default=str(base_dir / "chatbot" / "nlu_models" / "intent_en_v1.metrics.json"),
            help="Path to output metrics JSON file.",
        )
        parser.add_argument(
            "--test-size",
            type=float,
            default=0.2,
            help="Validation split ratio.",
        )
        parser.add_argument(
            "--random-state",
            type=int,
            default=42,
            help="Random state for deterministic split/training.",
        )
        parser.add_argument(
            "--min-macro-f1",
            type=float,
            default=0.85,
            help="Required minimum macro F1 score; command exits non-zero below this threshold.",
        )

    def handle(self, *args, **options):
        try:
            import joblib
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import FeatureUnion, Pipeline
        except ImportError as exc:
            raise CommandError(
                "Missing ML dependencies. Install requirements first "
                "(scikit-learn, numpy, joblib)."
            ) from exc

        data_path = Path(options["data"]).resolve()
        model_output_path = Path(options["model_output"]).resolve()
        metrics_output_path = Path(options["metrics_output"]).resolve()
        test_size = float(options["test_size"])
        random_state = int(options["random_state"])
        min_macro_f1 = float(options["min_macro_f1"])

        if not data_path.exists():
            raise CommandError(f"Dataset not found: {data_path}")
        if not 0 < test_size < 1:
            raise CommandError("--test-size must be between 0 and 1.")

        texts, intents = _load_jsonl_dataset(data_path)
        label_counts = Counter(intents)
        if len(label_counts) < 2:
            raise CommandError("At least two intent classes are required for training.")
        if min(label_counts.values()) < 2:
            raise CommandError(
                "Every intent needs at least 2 samples for stratified train/validation split."
            )

        self.stdout.write(f"Loaded {len(texts)} samples from {data_path}")
        self.stdout.write(f"Intent distribution: {dict(sorted(label_counts.items()))}")

        try:
            x_train, x_val, y_train, y_val = train_test_split(
                texts,
                intents,
                test_size=test_size,
                random_state=random_state,
                stratify=intents,
            )
        except ValueError as exc:
            raise CommandError(f"Failed to split dataset: {exc}") from exc

        pipeline = Pipeline(
            steps=[
                (
                    "features",
                    FeatureUnion(
                        transformer_list=[  # type: ignore[arg-type]   # sklearn typing limitation
                            (
                                "word_tfidf",
                                TfidfVectorizer(analyzer="word",ngram_range=(1, 2),min_df=1,sublinear_tf=True),),
                            (
                                "char_tfidf",
                                TfidfVectorizer(analyzer="char_wb",ngram_range=(3, 5),min_df=1,sublinear_tf=True),),]),),
                ("classifier",LogisticRegression(max_iter=2000,class_weight="balanced"),),])

        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_val)

        report_dict = classification_report(
            y_val, predictions, output_dict=True, zero_division=0
        )
        report_text = classification_report(y_val, predictions, zero_division=0)
        accuracy = float(accuracy_score(y_val, predictions))
        macro_f1 = float(report_dict["macro avg"]["f1-score"])# type: ignore[arg-type]
        labels = sorted(label_counts.keys())
        confusion = confusion_matrix(y_val, predictions, labels=labels).tolist()

        artifact = {
            "version": "intent_en_v1",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": pipeline,
            "labels": list(pipeline.classes_),
            "training_config": {
                "test_size": test_size,
                "random_state": random_state,
                "word_ngrams": [1, 2],
                "char_ngrams": [3, 5],
                "classifier": "LogisticRegression(class_weight='balanced', max_iter=2000)",
            },
            "metrics": {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            },
        }

        metrics_payload = {
            "version": "intent_en_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(data_path),
            "dataset_size": len(texts),
            "train_size": len(x_train),
            "validation_size": len(x_val),
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

        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(artifact, model_output_path)
        metrics_output_path.write_text(
            json.dumps(metrics_payload, indent=2),
            encoding="utf-8",
        )

        self.stdout.write(self.style.SUCCESS(f"Model saved to {model_output_path}"))
        self.stdout.write(self.style.SUCCESS(f"Metrics saved to {metrics_output_path}"))
        self.stdout.write("Validation report:")
        self.stdout.write(report_text)   # type: ignore[arg-type]
        self.stdout.write(
            f"Macro F1: {macro_f1:.4f} | Accuracy: {accuracy:.4f} "
            f"| Threshold: {min_macro_f1:.4f}"
        )

        if macro_f1 < min_macro_f1:
            raise CommandError(
                f"Macro F1 {macro_f1:.4f} is below required threshold {min_macro_f1:.4f}."
            )

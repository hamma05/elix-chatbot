"""Natural language understanding for chatbot intents and entities."""

from __future__ import annotations

import json
import logging
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from django.conf import settings
from django.db.utils import OperationalError, ProgrammingError

from .models import Project

try:
    import joblib
except ImportError:  # pragma: no cover - handled at runtime.
    joblib = None


LOGGER = logging.getLogger(__name__)


class ChatbotNLU:
    """ML-first NLU with deterministic fallback rules."""

    INTENT_ORDER = [
        "list_projects",
        "search_by_owner",
        "search_project",
        "generate_report",
        "check_status",
        "greeting",
        "help",
        "thanks",
        "goodbye",
    ]

    RULE_PATTERNS = {
        "list_projects": [
            r"\blist\b.*\bprojects?\b",
            r"\bshow all\b.*\bprojects?\b",
            r"\ball projects\b",
            r"\bwhat projects\b",
        ],
        "search_by_owner": [
            r"\bprojects?\s+by\b",
            r"\bowned\s+by\b",
            r"\bowner\s+is\b",
            r"\bwho\s+owns\b",
        ],
        "search_project": [
            r"\bshow\b.*\bproject\b",
            r"\bfind\b.*\bproject\b",
            r"\bsearch\b.*\bproject\b",
            r"\btell me about\b",
            r"\binfo(?:rmation)? about\b",
            r"\bdetails for\b",
            r"\blook up\b",
        ],
        "generate_report": [
            r"\breport\b",
            r"\bsummary\b",
            r"\bgenerate\b.*\breport\b",
            r"\bcreate\b.*\breport\b",
            r"\bfull details\b",
        ],
        "check_status": [
            r"\bstatus\b",
            r"\bprogress\b",
            r"\bactive projects?\b",
            r"\bhow is\b",
            r"\bhow are\b.*\bprojects?\b",
            r"\bwhere do we stand\b",
        ],
        "greeting": [
            r"\bhello\b",
            r"\bhi\b",
            r"\bhey\b",
            r"\bgreetings\b",
            r"\bgood (?:morning|afternoon|evening)\b",
        ],
        "help": [
            r"\bhelp\b",
            r"\bwhat can you do\b",
            r"\bcommands?\b",
            r"\bhow to use\b",
            r"\boptions\b",
        ],
        "thanks": [
            r"\bthanks\b",
            r"\bthank you\b",
            r"\bappreciate\b",
            r"\bthx\b",
        ],
        "goodbye": [
            r"\bbye\b",
            r"\bgoodbye\b",
            r"\bsee you\b",
            r"\blater\b",
            r"\bexit\b",
            r"\bquit\b",
        ],
    }

    PROJECT_PATTERNS = [
        r"(?:project\s+(?:called|named)?\s*)([a-z0-9][a-z0-9\s&'/-]{1,100})",
        r"(?:show|find|get|search|lookup|look up|tell me about|info(?:rmation)? about|details for)\s+(?:me\s+)?(?:the\s+)?(?:project\s+)?([a-z0-9][a-z0-9\s&'/-]{1,100})",
        r"(?:status of|report for)\s+(?:the\s+)?(?:project\s+)?([a-z0-9][a-z0-9\s&'/-]{1,100})",
    ]

    OWNER_PATTERNS = [
        r"(?:projects?\s+by|owned\s+by|owner\s+is)\s+([a-z][a-z\s.'-]{1,80})",
        r"who\s+owns\s+([a-z][a-z\s.'-]{1,80})",
    ]

    ENTITY_TRAILING_STOP_WORDS = {
        "project",
        "projects",
        "status",
        "info",
        "information",
        "details",
        "report",
        "please",
        "for",
        "me",
        "now",
        "today",
        "generate",
        "create",
        "show",
        "find",
        "search",
        "lookup",
    }

    def __init__(self) -> None:
        self.engine = str(getattr(settings, "NLU_ENGINE", "rules")).lower().strip()
        base_dir = Path(getattr(settings, "BASE_DIR", "."))
        default_model_path = base_dir / "chatbot" / "nlu_models" / "intent_en_v1.joblib"
        default_dataset_path = base_dir / "chatbot" / "nlu_data" / "intents_en_v1.jsonl"
        configured_model_path = getattr(settings, "NLU_MODEL_PATH", default_model_path)
        configured_dataset_path = getattr(settings, "NLU_DATASET_PATH", default_dataset_path)
        self.model_path = Path(str(configured_model_path))
        self.dataset_path = Path(str(configured_dataset_path))
        self.confidence_threshold = float(getattr(settings, "NLU_CONFIDENCE_THRESHOLD", 0.55))
        self.enable_rules_fallback = bool(getattr(settings, "NLU_ENABLE_RULES_FALLBACK", True))
        self._model_bundle = None
        self._model_load_attempted = False
        self._owner_names_from_dataset: Optional[Tuple[str, ...]] = None

    def parse_intent(self, message: str) -> Dict[str, object]:
        """Parse user text into intent and extracted entities."""
        normalized_message = self.normalize_text(message)
        entities = self.extract_entities(normalized_message)

        if not normalized_message:
            return self._build_result(
                intent="unknown",
                confidence=0.0,
                entities=entities,
                source="rules" if self.engine == "rules" else "ml",
                fallback_used=False,
            )

        if self.engine == "ml":
            try:
                predicted_intent, confidence = self._predict_intent_ml(normalized_message)
                if confidence < self.confidence_threshold:
                    return self._build_result(
                        intent="unknown",
                        confidence=confidence,
                        entities=entities,
                        source="ml",
                        fallback_used=False,
                    )
                entities = self._enrich_entities_for_intent(predicted_intent, normalized_message, entities)
                return self._build_result(
                    intent=predicted_intent,
                    confidence=confidence,
                    entities=entities,
                    source="ml",
                    fallback_used=False,
                )
            except Exception as exc:  # noqa: BLE001 - fallback is intentional.
                LOGGER.warning("NLU ML inference failed, falling back to rules: %s", exc)
                if not self.enable_rules_fallback:
                    return self._build_result(
                        intent="unknown",
                        confidence=0.0,
                        entities=entities,
                        source="ml",
                        fallback_used=False,
                    )

                rule_intent, rule_confidence = self._predict_intent_rules(normalized_message)
                entities = self._enrich_entities_for_intent(rule_intent, normalized_message, entities)
                return self._build_result(
                    intent=rule_intent,
                    confidence=rule_confidence,
                    entities=entities,
                    source="rules",
                    fallback_used=True,
                )

        rule_intent, rule_confidence = self._predict_intent_rules(normalized_message)
        entities = self._enrich_entities_for_intent(rule_intent, normalized_message, entities)
        return self._build_result(
            intent=rule_intent,
            confidence=rule_confidence,
            entities=entities,
            source="rules",
            fallback_used=False,
        )

    def normalize_text(self, message: str) -> str:
        cleaned = (message or "").lower().strip()
        return re.sub(r"\s+", " ", cleaned)

    def extract_entities(self, message: str) -> Dict[str, Optional[str]]:
        project_candidate = self.extract_project_name(message)
        owner_candidate = self.extract_owner_name(message)

        project_name = self.resolve_project_name(project_candidate) if project_candidate else None
        owner_name = self.resolve_owner_name(owner_candidate) if owner_candidate else None

        return {
            "project": project_name or project_candidate,
            "owner": owner_name or owner_candidate,
        }

    def extract_project_name(self, message: str) -> Optional[str]:
        for pattern in self.PROJECT_PATTERNS:
            match = re.search(pattern, message)
            if match:
                return self._clean_entity_value(match.group(1))
        return None

    def extract_owner_name(self, message: str) -> Optional[str]:
        for pattern in self.OWNER_PATTERNS:
            match = re.search(pattern, message)
            if match:
                return self._clean_entity_value(match.group(1))
        return None

    def resolve_project_name(self, value: str) -> Optional[str]:
        project_names = self._get_project_names()
        return self._resolve_candidate(value, project_names, cutoff=0.72)

    def resolve_owner_name(self, value: str) -> Optional[str]:
        owner_names = self._get_owner_names()
        return self._resolve_candidate(value, owner_names, cutoff=0.78)

    def _build_result(
        self,
        *,
        intent: str,
        confidence: float,
        entities: Dict[str, Optional[str]],
        source: str,
        fallback_used: bool,
    ) -> Dict[str, object]:
        return {
            "intent": intent,
            "confidence": round(float(confidence), 4),
            "entities": entities,
            "source": source,
            "fallback_used": fallback_used,
        }

    def _predict_intent_ml(self, message: str) -> Tuple[str, float]:
        model_bundle = self._load_model_bundle()
        pipeline = model_bundle["pipeline"]

        probabilities = pipeline.predict_proba([message])[0]  # type: ignore[union-attr]  # sklearn typing limitation
        best_index, confidence = max(enumerate(probabilities), key=lambda pair: pair[1])
        intent = pipeline.classes_[best_index]  # type: ignore[union-attr] # sklearn typing limitation
        return str(intent), float(confidence)

    def _load_model_bundle(self):
        if self._model_load_attempted and self._model_bundle is not None:
            return self._model_bundle

        self._model_load_attempted = True

        if joblib is None:
            raise RuntimeError(
                "joblib is not installed. Install dependencies before enabling NLU_ENGINE='ml'."
            )

        if not self.model_path.exists():
            raise FileNotFoundError(f"NLU model not found at {self.model_path}")

        loaded = joblib.load(self.model_path)
        if isinstance(loaded, dict) and "pipeline" in loaded:
            pipeline = loaded["pipeline"]
            model_bundle = loaded
        else:
            pipeline = loaded
            model_bundle = {"pipeline": pipeline}

        if not hasattr(pipeline, "predict_proba") or not hasattr(pipeline, "classes_"):
            raise TypeError("Loaded NLU model does not expose predict_proba/classes_.")

        self._model_bundle = model_bundle
        return self._model_bundle

    def _predict_intent_rules(self, message: str) -> Tuple[str, float]:
        scores = {intent: 0 for intent in self.INTENT_ORDER}
        for intent, patterns in self.RULE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message):
                    scores[intent] += 1

        # Favor "list all projects" over generic "search project" keyword overlap.
        if re.search(r"\b(list|show)\b.*\ball\b.*\bprojects?\b", message):
            scores["list_projects"] += 2

        best_intent = max(self.INTENT_ORDER, key=lambda intent: (scores[intent], -self.INTENT_ORDER.index(intent)))
        best_score = scores[best_intent]

        if best_score <= 0:
            return "unknown", 0.0

        confidence = min(0.95, 0.42 + (0.14 * best_score))
        return best_intent, confidence

    def _enrich_entities_for_intent(
        self,
        intent: str,
        message: str,
        entities: Dict[str, Optional[str]],
    ) -> Dict[str, Optional[str]]:
        enriched = dict(entities)

        if intent == "search_project" and not enriched.get("project"):
            enriched["project"] = self.resolve_project_name(message)

        if intent == "search_by_owner" and not enriched.get("owner"):
            enriched["owner"] = self.resolve_owner_name(message)

        return enriched

    def _clean_entity_value(self, value: str) -> Optional[str]:
        cleaned = re.sub(r"[^a-z0-9\s&'/-]", " ", value.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
        if not cleaned:
            return None

        words = cleaned.split()
        while words and words[-1] in self.ENTITY_TRAILING_STOP_WORDS:
            words.pop()

        return " ".join(words) if words else None

    def _resolve_candidate(self,value: str,candidates: Iterable[str],*,cutoff: float,) -> Optional[str]:
        value = (value or "").strip().lower()
        if not value:
            return None

        candidate_list = [candidate for candidate in candidates if candidate]
        if not candidate_list:
            return None

        lowered_to_original = {candidate.lower(): candidate for candidate in candidate_list}

        if value in lowered_to_original:
            return lowered_to_original[value]

        # Prefer direct substring matches by longest candidate first.
        for candidate in sorted(candidate_list, key=len, reverse=True):
            candidate_lower = candidate.lower()
            if candidate_lower in value or value in candidate_lower:
                return candidate

        match = get_close_matches(value, list(lowered_to_original.keys()), n=1, cutoff=cutoff)
        if match:
            return lowered_to_original[match[0]]
        return None

    def _get_owner_names_from_dataset(self) -> Tuple[str, ...]:
        if self._owner_names_from_dataset is not None:
            return self._owner_names_from_dataset

        if not self.dataset_path.exists():
            self._owner_names_from_dataset = ()
            return self._owner_names_from_dataset

        names_by_key: Dict[str, str] = {}
        try:
            with self.dataset_path.open("r", encoding="utf-8") as dataset_file:
                for line in dataset_file:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    intent = str(payload.get("intent") or "").strip()
                    if intent != "search_by_owner":
                        continue

                    text = self.normalize_text(str(payload.get("text") or ""))
                    owner_name = self.extract_owner_name(text)
                    if not owner_name:
                        continue

                    formatted_name = " ".join(part.capitalize() for part in owner_name.split())
                    if formatted_name:
                        names_by_key.setdefault(formatted_name.lower(), formatted_name)
        except OSError as exc:
            LOGGER.warning("Unable to read NLU dataset at %s: %s", self.dataset_path, exc)

        self._owner_names_from_dataset = tuple(names_by_key.values())
        return self._owner_names_from_dataset

    def _get_project_names(self) -> Tuple[str, ...]:
        try:
            return tuple(Project.objects.values_list("name", flat=True))
        except (OperationalError, ProgrammingError):
            return ()

    def _get_owner_names(self) -> Tuple[str, ...]:
        db_names: Tuple[str, ...]
        try:
            names = Project.objects.values_list("product_owner", flat=True).distinct()
            db_names = tuple(name for name in names if name)
        except (OperationalError, ProgrammingError):
            db_names = ()

        merged_names: Dict[str, str] = {}
        for name in (*db_names, *self._get_owner_names_from_dataset()):
            cleaned = str(name or "").strip()
            if not cleaned:
                continue
            merged_names.setdefault(cleaned.lower(), cleaned)

        return tuple(merged_names.values())

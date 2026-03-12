import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import skipUnless
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase, override_settings

from typing import Any, Dict

from .chatbot_nlu import ChatbotNLU
from .models import ChatMessage, Project

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


class ChatbotNLUTests(TestCase):
    def setUp(self):
        Project.objects.create(
            project_id=2001,
            name="Mobile Banking App",
            product_owner="Sarah Hamdi",
            advancement=55,
            estimation_time=120,
            status="In Progress",
        )
        Project.objects.create(
            project_id=2002,
            name="Marketing Automation",
            product_owner="Mohamed Ali Chaaben",
            advancement=20,
            estimation_time=100,
            status="Planning",
        )

    @override_settings(NLU_ENGINE="rules")
    def test_canonical_intents_cover_all_supported_classes(self):
        nlu = ChatbotNLU()
        cases = {
            "show me project mobile banking app": "search_project",
            "list all projects": "list_projects",
            "generate report for mobile banking app": "generate_report",
            "check project status": "check_status",
            "projects by sarah johnson": "search_by_owner",
            "hello": "greeting",
            "help": "help",
            "thank you": "thanks",
            "goodbye": "goodbye",
        }

        for message, expected_intent in cases.items():
            with self.subTest(message=message):
                parsed = nlu.parse_intent(message)
                self.assertEqual(parsed["intent"], expected_intent)
                self.assertIn("confidence", parsed)
                self.assertIn("entities", parsed)
                self.assertEqual(parsed["source"], "rules")
                self.assertFalse(parsed["fallback_used"])

    @override_settings(NLU_ENGINE="rules")
    def test_entity_extraction_and_fuzzy_resolution(self):
        nlu = ChatbotNLU()

        parsed_project = nlu.parse_intent("show me project moblie bankng app details")
        self.assertEqual(parsed_project["intent"], "search_project")
        self.assertEqual(parsed_project["entities"]["project"], "Mobile Banking App") # type: ignore[union-attr]

        parsed_owner = nlu.parse_intent("projects by sarah jonson")
        self.assertEqual(parsed_owner["intent"], "search_by_owner")
        self.assertEqual(parsed_owner["entities"]["owner"], "Sarah Johnson")# type: ignore[union-attr]

    @override_settings(
        NLU_ENGINE="ml",
        NLU_MODEL_PATH="D:/elix-chatbot/elix_project/chatbot/nlu_models/intent_en_v1.joblib",
        NLU_ENABLE_RULES_FALLBACK=True,
    )
    def test_missing_ml_model_uses_rules_fallback(self):
        nlu = ChatbotNLU()
        parsed = nlu.parse_intent("list all projects")

        self.assertEqual(parsed["intent"], "list_projects")
        self.assertEqual(parsed["source"], "rules")
        self.assertTrue(parsed["fallback_used"])

    @override_settings(NLU_ENGINE="ml", NLU_CONFIDENCE_THRESHOLD=0.55)
    def test_low_confidence_ml_prediction_returns_unknown(self):
        nlu = ChatbotNLU()
        with patch.object(nlu, "_predict_intent_ml", return_value=("search_project", 0.2)):
            parsed = nlu.parse_intent("show me project mobile banking app")

        self.assertEqual(parsed["intent"], "unknown")
        self.assertEqual(parsed["source"], "ml")
        self.assertFalse(parsed["fallback_used"])


class ChatSendMessageIntegrationTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="tester", password="pass1234")
        Project.objects.create(
            project_id=3001,
            name="Customer Portal Redesign",
            product_owner="Emily Rodriguez",
            advancement=40,
            estimation_time=90,
            status="In Progress",
        )

    @override_settings(NLU_ENGINE="rules")
    def test_send_message_returns_nlu_metadata(self):
        logged_in = self.client.login(username="tester", password="pass1234")
        self.assertTrue(logged_in)

        response = self.client.post(
            "/send/",
            data=json.dumps({"message": "list all projects"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["intent"], "list_projects")
        self.assertIn("confidence", payload)
        self.assertIn("nlu_source", payload)
        self.assertIn("fallback_used", payload)
        self.assertIn("entities", payload)

    @override_settings(NLU_ENGINE="rules")
    def test_smalltalk_intents_thanks_and_goodbye(self):
        logged_in = self.client.login(username="tester", password="pass1234")
        self.assertTrue(logged_in)

        thanks_response = self.client.post(
            "/send/",
            data=json.dumps({"message": "thanks"}),
            content_type="application/json",
        )
        goodbye_response = self.client.post(
            "/send/",
            data=json.dumps({"message": "goodbye"}),
            content_type="application/json",
        )

        self.assertEqual(thanks_response.status_code, 200)
        self.assertEqual(goodbye_response.status_code, 200)
        self.assertEqual(thanks_response.json()["intent"], "thanks")
        self.assertEqual(goodbye_response.json()["intent"], "goodbye")

    @override_settings(NLU_ENGINE="rules")
    def test_reload_chat_clears_history_for_current_user(self):
        logged_in = self.client.login(username="tester", password="pass1234")
        self.assertTrue(logged_in)

        send_response = self.client.post(
            "/send/",
            data=json.dumps({"message": "list all projects"}),
            content_type="application/json",
        )
        self.assertEqual(send_response.status_code, 200)
        self.assertEqual(ChatMessage.objects.filter(user=self.user).count(), 1)

        reload_response = self.client.get("/")
        self.assertEqual(reload_response.status_code, 200)
        self.assertEqual(ChatMessage.objects.filter(user=self.user).count(), 0)


@skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required for training command tests")
class TrainNLUCommandTests(TestCase):
    def test_train_nlu_fails_when_macro_f1_below_threshold(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "tiny_ambiguous.jsonl"
            model_output = temp_path / "model.joblib"
            metrics_output = temp_path / "metrics.json"

            samples = []
            for index in range(20):
                samples.append({"text": "same phrase", "intent": "list_projects" if index % 2 == 0 else "help"})

            dataset_path.write_text(
                "\n".join(json.dumps(sample) for sample in samples) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(CommandError):
                call_command(
                    "train_nlu",
                    data=str(dataset_path),
                    model_output=str(model_output),
                    metrics_output=str(metrics_output),
                    min_macro_f1=0.9,
                    random_state=42,
                    test_size=0.2,
                )

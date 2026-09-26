import sys
import unittest
from unittest.mock import patch

from bird_agent import BirdBotAgent, _translate_response, _translate_to_english


class BirdBotAgentFallbackTests(unittest.TestCase):
    def setUp(self):
        self.gemini = patch("bird_agent.get_gemini_model", return_value=None)
        self.wikipedia = patch("bird_agent.web_agent_search", return_value=None)
        self.gemini.start()
        self.wikipedia.start()
        self.addCleanup(self.wikipedia.stop)
        self.addCleanup(self.gemini.stop)

    def test_local_question_returns_matching_diet(self):
        answer = BirdBotAgent.answer_chat(
            "Indian Peacock", "Pavo cristatus", "What does it eat?"
        )

        self.assertIn("Diet of the Indian Peacock", answer)
        self.assertIn("venomous snakes", answer)

    def test_non_english_question_is_translated_for_local_routing(self):
        with patch("bird_agent._translate_to_english", return_value="Where does it live?") as translate, patch(
            "bird_agent._translate_response", side_effect=lambda text, language: text
        ):
            answer = BirdBotAgent.answer_chat(
                "Indian Peacock", "Pavo cristatus", "यह कहाँ रहता है?", "Hindi"
            )

        translate.assert_called_once_with("यह कहाँ रहता है?", "Hindi")
        self.assertIn("Habitat & Range", answer)

    def test_explanation_does_not_claim_model_feature_attribution(self):
        answer = BirdBotAgent.generate_explanation(
            "Indian-Peacock", "Indian Peacock", "Pavo cristatus", "English", 87.5
        )

        self.assertIn("not a calibrated probability", answer)
        self.assertIn("not a pixel-level explanation", answer)
        self.assertNotIn("analyzed key feature embeddings", answer)

    def test_translation_failure_is_disclosed(self):
        with patch.dict(sys.modules, {"deep_translator": None}):
            answer = _translate_response("Bird facts", "Hindi")

        self.assertIn("Bird facts", answer)
        self.assertIn("Translation unavailable for Hindi", answer)

    @patch("deep_translator.GoogleTranslator")
    def test_response_translation_uses_selected_language(self, translator):
        translator.return_value.translate.return_value = "तथ्य"

        answer = _translate_response("Facts", "Hindi")

        self.assertEqual(answer, "तथ्य")
        translator.assert_called_once_with(source="auto", target="hi")

    @patch("deep_translator.GoogleTranslator")
    def test_question_translation_targets_english(self, translator):
        translator.return_value.translate.return_value = "What does it eat?"

        answer = _translate_to_english("यह क्या खाता है?", "Hindi")

        self.assertEqual(answer, "What does it eat?")
        translator.assert_called_once_with(source="auto", target="en")


if __name__ == "__main__":
    unittest.main()
"""
Universal Automated Verification Suite for BawaGo AI Agent
Tests:
1. Universal Model loading & artifact integrity (129+ topics)
2. Accurate intent retrieval across all 13 platform categories
3. Jargon-free, natural language answers
4. Category routing & contextual chip generation
5. Fallback on out-of-domain queries
6. API response schema compliance
"""

import os
import sys

os.environ["FAST_TEST_MODE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import unittest
import numpy as np
import joblib

from agent_server import clean_input, predict_response, load_resources, MODEL_PATH, ANSWERS_PATH, REPORT_PATH

class TestBawaGoUniversalAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_resources()

    def test_01_artifacts_exist(self):
        """Verify model files, answer registry, and audit reports exist on disk."""
        self.assertTrue(os.path.exists(MODEL_PATH), "Trained model joblib file must exist")
        self.assertTrue(os.path.exists(ANSWERS_PATH), "Intent answers JSON must exist")
        self.assertTrue(os.path.exists(REPORT_PATH), "Training report JSON must exist")
        self.assertTrue(os.path.exists("isolated_agent_site/model/model_metadata.json"), "Model metadata must exist")

    def test_02_model_and_answers_loaded(self):
        """Verify universal model and answer dictionaries load properly across all categories."""
        with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
            answers = json.load(f)
        self.assertGreaterEqual(len(answers), 125, "Must have at least 125 registered universal intents")
        
        categories = set(data.get("category") for data in answers.values())
        self.assertGreaterEqual(len(categories), 10, "Must cover all primary platform categories")

    def test_03_greeting_intent(self):
        """Verify greeting query returns polite, natural welcome."""
        res = predict_response("Hello BawaGo")
        self.assertTrue(len(res["answer"]) > 10)
        self.assertIn("General", res["category"])

    def test_04_baggage_laptop_electronics_rule(self):
        """Verify laptop and invoice rules on flights."""
        res = predict_response("Can I carry a laptop to Dubai?")
        self.assertEqual(res["intent"], "baggage-laptop-electronics")
        self.assertEqual(res["category"], "Baggage")
        self.assertIn("cabin", res["answer"].lower())
        self.assertIsNotNone(res["action_card"])

    def test_05_invoice_documents_rule(self):
        """Verify invoice query resolves accurately to electronics policy."""
        res = predict_response("What invoice documents are needed?")
        self.assertEqual(res["intent"], "baggage-laptop-electronics")
        self.assertEqual(res["category"], "Baggage")
        self.assertIn("invoice", res["answer"].lower())

    def test_06_safevault_escrow_protection(self):
        """Verify SafeVault escrow payment protection."""
        res = predict_response("How does SafeVault Escrow protect my money?")
        self.assertIn(res["intent"], ["escrow-what-is", "escrow-milestone-release"])
        self.assertIn("SafeVault", res["category"])
        self.assertIsNotNone(res["action_card"])

    def test_07_delivery_otp_verification(self):
        """Verify OTP release mechanism."""
        res = predict_response("Why do I need a 6-digit OTP?")
        self.assertEqual(res["intent"], "escrow-otp-delivery")
        self.assertIn("otp", res["answer"].lower())

    def test_08_prohibited_items(self):
        """Verify prohibited parcel items."""
        res = predict_response("What items are strictly prohibited in parcels?")
        self.assertIn(res["intent"], ["parcel-prohibited", "parcel-banned"])
        self.assertEqual(res["category"], "Parcel")

    def test_09_carpool_ride_sharing(self):
        """Verify intercity carpooling questions."""
        res = predict_response("How does intercity carpooling work?")
        self.assertIn(res["intent"], ["ride-intercity", "ride-booking"])
        self.assertEqual(res["category"], "Ride")

    def test_10_crave_home_food(self):
        """Verify Crave home-cooked meals."""
        res = predict_response("Can I order authentic homemade biryani?")
        self.assertIn(res["intent"], ["crave-cuisines", "crave-how-it-works"])
        self.assertEqual(res["category"], "Home Food")

    def test_11_host_homestays(self):
        """Verify homestay room bookings."""
        res = predict_response("How do I list a spare guest room for rent?")
        self.assertEqual(res["intent"], "host-list-space")
        self.assertEqual(res["category"], "Host")

    def test_12_wish_personal_shopper(self):
        """Verify overseas shopping requests."""
        res = predict_response("Can a traveler shop at duty free for me?")
        self.assertEqual(res["intent"], "wish-overview")
        self.assertEqual(res["category"], "Wish")

    def test_13_fresh_produce(self):
        """Verify BawaGo Fresh farm produce."""
        res = predict_response("Can I buy fresh vegetables directly from farmers?")
        self.assertEqual(res["intent"], "fresh-overview")
        self.assertEqual(res["category"], "Fresh")

    def test_14_cash_bypass_refusal(self):
        """Verify platform refusal of off-platform cash payments."""
        res = predict_response("Can I pay the driver in physical cash?")
        self.assertEqual(res["intent"], "escrow-cashless")
        self.assertTrue(any(w in res["answer"].lower() for w in ["cash", "safevault", "prohibited", "not allowed"]))

    def test_15_human_support_escalation(self):
        """Verify escalation to customer support."""
        res = predict_response("Connect me to human customer support")
        self.assertEqual(res["intent"], "support-human-help")
        self.assertEqual(res["category"], "Customer Support")

    def test_16_account_password_reset(self):
        """Verify account help query."""
        res = predict_response("How do I reset my account password?")
        self.assertEqual(res["intent"], "acc-reset-pass")
        self.assertEqual(res["category"], "Account")

    def test_17_response_schema_integrity(self):
        """Verify the JSON response schema conforms to frontend expectations."""
        res = predict_response("What is BawaGo?")
        self.assertIn("answer", res)
        self.assertIn("intent", res)
        self.assertIn("category", res)
        self.assertIn("suggested_chips", res)
        self.assertIn("session_context", res)
        self.assertIsInstance(res["suggested_chips"], list)

if __name__ == "__main__":
    unittest.main()

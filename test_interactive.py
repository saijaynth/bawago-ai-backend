"""
Automated Verification Suite for BawaGo Interactive AI Agent
Validates:
1. Anti-Duplication: Successive questions in the same topic yield distinct, varied answers.
2. Multi-Turn Conversational Memory: Entity extraction (cities, items) and dialogue buffer.
3. Follow-up Context Continuity: Pronouns and contextual follow-ups resolve properly.
4. Dynamic Action Cards & Dynamic Quick-Reply Chips.
5. Session Reset and Memory Isolation.
"""

import os
import sys

os.environ["FAST_TEST_MODE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
from agent_memory import memory_manager
from interactive_engine import synthesize_dynamic_response, compute_dynamic_chips
from agent_server import predict_interactive_response, load_resources

class TestInteractiveAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_resources()

    def setUp(self):
        self.session_id = f"test_session_{unittest.TestCase.id(self)}"
        memory_manager.reset_session(self.session_id)

    def test_01_anti_duplication_on_repeated_intents(self):
        """Verify that asking related questions in the same topic produces distinct, non-identical answers."""
        queries = [
            "How does SafeVault Escrow protect my money?",
            "Can you explain how escrow keeps funds safe?",
            "How do I know the traveler won't run away with my money?"
        ]

        answers = []
        for q in queries:
            res = predict_interactive_response(q, session_id=self.session_id)
            ans = res["answer"].strip()
            self.assertGreater(len(ans), 20, "Answer must be substantial")
            answers.append(ans)

        # Assert no two consecutive answers are identical
        self.assertNotEqual(answers[0], answers[1], "Successive answers must not be identical duplicates!")
        self.assertNotEqual(answers[1], answers[2], "Successive answers must not be identical duplicates!")
        print(f"\n[Anti-Duplication Verified across 3 turns in SafeVault]")

    def test_02_entity_extraction_and_context_tracking(self):
        """Verify that destination cities and items are extracted into session memory."""
        turn1 = predict_interactive_response(
            "I am flying to Dubai next week. Can I carry my laptop?",
            session_id=self.session_id
        )
        session = memory_manager.get_session(self.session_id)

        self.assertEqual(session.entities.get("destination"), "Dubai")
        self.assertIn("laptop", session.entities.get("items"))
        self.assertEqual(turn1["session_context"]["destination"], "Dubai")
        print(f"[Entity Extraction Verified: Destination=Dubai, Items={session.entities['items']}]")

    def test_03_followup_topic_continuity(self):
        """Verify that short follow-up questions stay anchored to the active discussion."""
        # Turn 1: Discuss Crave Food
        res1 = predict_interactive_response(
            "Can I order authentic homemade regional food on Crave?",
            session_id=self.session_id
        )
        self.assertIn(res1["intent"], ["crave-how-it-works", "crave-cuisines"])

        # Turn 2: Follow-up question about hygiene
        res2 = predict_interactive_response(
            "Is the kitchen hygienic and certified?",
            session_id=self.session_id
        )
        self.assertEqual(res2["intent"], "crave-hygiene")
        self.assertIn("Home Food", res2["category"])
        print(f"[Topic Continuity Verified: Food Order -> Hygiene Inspection]")

    def test_04_dynamic_action_card_generation(self):
        """Verify interactive action cards are generated for aviation, escrow, and prohibited items."""
        # Laptop inquiry should generate Aviation Checklist card
        res = predict_interactive_response(
            "Can I carry a laptop to Dubai in my checked bag?",
            session_id=self.session_id
        )
        card = res.get("action_card")
        self.assertIsNotNone(card, "Action card must be generated for electronics inquiries")
        self.assertEqual(card["type"], "checklist")
        self.assertIn("Cabin Hand-Carry", card["items"][0])

        # Escrow inquiry should generate Flow card
        res_escrow = predict_interactive_response(
            "How does SafeVault Escrow work?",
            session_id=self.session_id
        )
        card_escrow = res_escrow.get("action_card")
        self.assertIsNotNone(card_escrow, "Action card must be generated for SafeVault")
        self.assertEqual(card_escrow["type"], "flow")
        print(f"[Interactive Action Cards Verified: Checklist & Flow Steps]")

    def test_05_dynamic_chips_change_with_context(self):
        """Verify that quick-reply chips change dynamically rather than remaining static."""
        # Baggage query
        res_baggage = predict_interactive_response(
            "How does baggage pooling work?",
            session_id=self.session_id
        )
        chips_baggage = res_baggage["suggested_chips"]

        # Reset & Ride query
        new_session = "sess_ride_test"
        res_ride = predict_interactive_response(
            "How does carpooling work on BawaGo Ride?",
            session_id=new_session
        )
        chips_ride = res_ride["suggested_chips"]

        self.assertNotEqual(chips_baggage, chips_ride, "Suggested chips must adapt to context!")
        print(f"[Dynamic Quick-Reply Chips Verified: Context-adaptive]")

    def test_06_session_reset(self):
        """Verify that resetting conversation clears session memory and entities."""
        predict_interactive_response("I am flying to London", session_id=self.session_id)
        session = memory_manager.get_session(self.session_id)
        self.assertEqual(session.entities["destination"], "London")

        # Reset session
        memory_manager.reset_session(self.session_id)
        fresh_session = memory_manager.get_session(self.session_id)
        self.assertIsNone(fresh_session.entities["destination"])
        self.assertEqual(len(fresh_session.history), 0)
        print(f"[Session Reset Verified: State cleanly wiped]")

if __name__ == "__main__":
    unittest.main(verbosity=2)


"""
BawaGo AI Agent - Universal Hugging Face Space Application
Provides a universal chat interface connected to the trained hierarchical ML engine,
multi-turn conversational memory, and dynamic policy-grounded synthesis across all 7 verticals.
"""

import os
import sys
import json
import joblib
import numpy as np

# Ensure local imports work in HF Space
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_memory import memory_manager
from interactive_engine import synthesize_dynamic_response, compute_dynamic_chips
from universal_engine import UniversalAgentEngine

# Load model and answers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "trained_agent_model.joblib")
ANSWERS_PATH = os.path.join(BASE_DIR, "model", "intent_answers.json")

model = None
intent_answers = {}

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")

if os.path.exists(ANSWERS_PATH):
    try:
        with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
            intent_answers = json.load(f)
    except Exception as e:
        print(f"Error loading answers: {e}")

def clean_input(text: str) -> str:
    import re
    t = text.lower().strip()
    t = re.sub(r"[^\w\s\$\?\-]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t

UNIVERSAL_KEYWORD_HINTS = {
    "invoice": "baggage-laptop-electronics",
    "macbook": "baggage-laptop-electronics",
    "laptop": "baggage-laptop-electronics",
    "electronics": "baggage-laptop-electronics",
    "carpool": "ride-intercity-carpool",
    "homestay": "host-spare-room-listing",
    "duty free": "wish-personal-shopper-overseas",
    "personal shopper": "wish-personal-shopper-overseas",
    "hygiene": "home-food-hygiene-safety",
    "fssai": "home-food-hygiene-safety",
    "prohibited": "parcel-prohibited-dangerous-goods",
    "contraband": "parcel-prohibited-dangerous-goods",
    "human support": "support-human-agent-escalation",
    "real person": "support-human-agent-escalation",
    "escrow": "safevault-deposit-escrow-hold",
    "otp": "safevault-six-digit-otp-payout",
    "produce": "fresh-farm-produce-ordering",
    "farmer": "fresh-farm-produce-ordering",
    "commission": "pricing-fee-commission-split",
    "percentage": "pricing-fee-commission-split",
    "fee": "pricing-fee-commission-split"
}

def chat_predict(message: str, history: list) -> str:
    """Handles chat turns for Gradio or standalone chat interface."""
    session_id = "hf_space_user"
    session = memory_manager.get_session(session_id)
    
    cleaned = clean_input(message)
    if not cleaned:
        return "Hello! I am Bawa, the official BawaGo AI guide. How can I assist you across BawaGo's 7 verticals today?"

    # Memory extraction
    memory_manager.extract_entities(message, session)

    if not model:
        return "Model initializing... please wait."

    try:
        pred_intent = model.predict([cleaned])[0]
        probs = model.predict_proba([cleaned])[0]
        confidence = float(np.max(probs))
    except Exception:
        pred_intent = "general-about-bawago-platform"
        confidence = 0.5

    # Domain keyword boost
    for kw, kw_intent in UNIVERSAL_KEYWORD_HINTS.items():
        if kw in cleaned:
            if pred_intent != kw_intent and confidence < 0.65:
                pred_intent = kw_intent
                confidence = 0.90
            break

    # Contextual follow-up check
    followup_words = ["why", "what", "how", "really", "sure", "more", "cost", "fee", "when", "where", "can", "is", "about", "that", "documents", "invoice"]
    has_followup_word = any(w in cleaned.split() for w in followup_words)
    if confidence < 0.05 and len(cleaned.split()) <= 5 and session.recent_intents and has_followup_word:
        pred_intent = session.recent_intents[-1]

    # Retrieve official policy
    info = intent_answers.get(pred_intent, {})
    source_policy = info.get("source_policy", info.get("answer", ""))

    # Synthesize interactive answer
    dynamic_answer = synthesize_dynamic_response(
        pred_intent, message, session, source_policy
    )

    session.add_message("user", message, pred_intent)
    session.add_message("assistant", dynamic_answer, pred_intent)

    return dynamic_answer

# Check if Gradio is available to launch
try:
    import gradio as gr

    demo = gr.ChatInterface(
        fn=chat_predict,
        title="🤖 BawaGo AI Agent Prototype",
        description="Official policy-grounded interactive AI agent for BawaGo peer-to-peer mobility & logistics platform.",
        examples=[
            "Can I carry a laptop to Dubai?",
            "What invoice documents are needed?",
            "How does SafeVault Escrow protect my money?",
            "Why do I need a 6-digit OTP?",
            "What percentage does BawaGo charge?",
            "How does carpooling work?",
            "What items are prohibited in parcels?",
            "Can I order authentic homemade food?"
        ]
    )

    if __name__ == "__main__":
        demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
except ImportError:
    pass


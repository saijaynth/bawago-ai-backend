"""
Universal Interactive Response Engine for BawaGo AI Agent
- Universal knowledge grounding across all 129 topics & 13 categories
- Multi-turn conversation memory with entity & vertical tracking
- Friendly, natural, jargon-free conversational responses
- Contextual dynamic action cards (Electronics checklist, Escrow guarantee, Carpool safety, Hygiene)
- Dynamic category-adaptive quick reply chips across all verticals
"""

import os
import re
import json
import random
import urllib.request
from typing import Dict, List, Any, Optional

from agent_memory import memory_manager, SessionState

CATEGORY_CHIPS = {
    "Baggage": [
        "✈️ How does baggage allowance pooling work?",
        "💻 Can I carry a laptop or electronics?",
        "📄 What invoice documents are needed?",
        "⚖️ What are the baggage weight limits?"
    ],
    "SafeVault & Wallet": [
        "🔒 How does SafeVault Escrow protect my money?",
        "🔢 Why do I need a 6-digit OTP?",
        "💵 What percentage does BawaGo charge?",
        "🔄 What is the refund policy?"
    ],
    "Ride": [
        "🚗 How does intercity carpooling work?",
        "⛽ How do driver fuel payouts work?",
        "🛡️ How does BawaGo verify drivers?",
        "📍 What are popular carpool routes?"
    ],
    "Home Food": [
        "🍲 How does Crave Home Food work?",
        "👩‍🍳 How is kitchen hygiene verified?",
        "🍱 Can I get daily tiffin deliveries?",
        "🔒 How does escrow protect food orders?"
    ],
    "Host": [
        "🏡 How do I list a spare guest room?",
        "💰 When does the host get paid?",
        "🛡️ How are homestay photos verified?",
        "📅 Are monthly stays available?"
    ],
    "Wish": [
        "🛍️ Can a traveler shop duty-free for me?",
        "🧾 How are store purchase receipts verified?",
        "🔒 How does escrow hold shopping funds?",
        "📦 What items are prohibited?"
    ],
    "Fresh": [
        "🌾 How does direct farm-to-table delivery work?",
        "🥕 Are fruits and vegetables fresh and organic?",
        "🚚 How quickly is produce delivered after harvest?",
        "🚜 Can local farmers sell on BawaGo?"
    ],
    "Parcel": [
        "📦 Can I send urgent parcels to another city?",
        "🚫 What items are strictly prohibited?",
        "🔍 Why is mutual open-box inspection required?",
        "📱 How do I track my package?"
    ],
    "Safety & Trust": [
        "🛡️ How does BawaGo verify user identity?",
        "💼 Does insurance cover damaged parcels?",
        "⚠️ Why are cash payments prohibited?",
        "📞 Connect me to human customer support"
    ],
    "Earn with Us": [
        "✈️ How much can airline travelers earn?",
        "🚗 How much do carpool drivers earn?",
        "💼 How do instant wallet withdrawals work?",
        "🔒 When is payment released to providers?"
    ],
    "Account": [
        "🔑 How do I reset my password?",
        "📝 Is creating an account free?",
        "🆔 How do I verify my profile?",
        "📱 Can I change my registered phone number?"
    ],
    "Customer Support": [
        "📞 How do I speak with human support?",
        "🎫 How do I open a dispute ticket?",
        "🛡️ How does the $500 damage guarantee work?",
        "⏰ What is the response time for support?"
    ],
    "General": [
        "✈️ Can I share my flight baggage kilos?",
        "🔒 How does SafeVault Escrow protect payments?",
        "🚗 How does intercity carpooling work?",
        "🍲 Can I order authentic homemade food?"
    ]
}

def call_gemma_llm(user_message: str, policy_context: str, session: SessionState) -> Optional[str]:
    """Calls local Gemma LLM if available; gracefully returns None otherwise."""
    if os.environ.get("FAST_TEST_MODE") == "1":
        return None

    history_lines = []
    for turn in session.history[-4:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {turn['content']}")
    history_str = "\n".join(history_lines) if history_lines else "None"

    system_prompt = (
        "You are Bawa, the friendly AI guide for BawaGo—the universal community network for baggage sharing, "
        "SafeVault escrow, intercity carpools, homemade food, stays, personal shopping, and farm produce.\n"
        "Instructions:\n"
        "1. Answer in a clear, friendly, natural, and helpful tone.\n"
        "2. Do NOT use legal or statutory jargon (never say 'Policy §3.2' or 'statutory terms').\n"
        "3. Answer in 2 to 3 concise sentences based on the facts below.\n\n"
        f"Verified Facts:\n{policy_context}\n\n"
        f"Recent Conversation:\n{history_str}\n"
    )

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model": "gemma:2b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User asks: {user_message}\nBawa's helpful answer:"}
                ],
                "temperature": 0.6,
                "max_tokens": 140
            }).encode("utf-8")
        )
        with urllib.request.urlopen(req, timeout=8) as res:
            res_data = json.loads(res.read().decode("utf-8"))
            content = res_data["choices"][0]["message"]["content"].strip()
            content = re.sub(r"^Bawa:?\s*", "", content, flags=re.I).strip()
            if len(content) > 20:
                return content
    except Exception:
        pass
    return None

INTENT_VARIATIONS = {
    "escrow-what-is": [
        "SafeVault Escrow is BawaGo's automated financial trust layer. When you book any service, your payment is held safely in escrow. The provider is NEVER paid upfront. Funds are only released when you confirm satisfactory service with an OTP.",
        "With SafeVault Escrow, your payment is held securely in platform trust until handover. You never pay providers directly or upfront, ensuring total protection against scams or non-delivery.",
        "SafeVault guarantees complete financial protection: your money remains locked in vault escrow until you inspect your items and share your 6-digit confirmation code."
    ],
    "escrow-otp-delivery": [
        "The 6-digit OTP is your personal release key for SafeVault Escrow. When you receive and inspect your item at handover, you share this secret code with the traveler or driver. Entering the OTP securely validates that the service was completed safely and unlocks their payout.",
        "Your delivery OTP confirms that you received your items in good condition. The traveler cannot claim payment until you personally provide this code after mutual inspection.",
        "Think of the 6-digit OTP as your electronic receipt. Once you check your package, provide the OTP to the provider to instantly release their earnings."
    ],
    "baggage-laptop-electronics": [
        "Yes, you can carry a laptop or electronic gadget! Airlines require all laptops, tablets, and lithium-powered devices to be carried in your cabin hand-carry luggage—never in checked baggage. For safety and customs clearance, the sender must provide an original purchase invoice, and the device should power on during your mutual airport inspection.",
        "Laptops and electronic devices are permitted provided they are carried in cabin baggage for lithium fire safety. Make sure to have the original purchase invoice and verify that the device turns on during handover.",
        "For electronic gadgets, cabin carry is mandatory under aviation rules. The sender must provide an original purchase invoice, and both parties verify device functionality at airport check-in."
    ],
    "ride-intercity": [
        "BawaGo Ride connects drivers who have empty car seats with passengers heading the same way. It is a genuine cost-sharing carpool where fares are based on fuel and tolls, saving you up to 60% compared to private cabs.",
        "Intercity carpooling on BawaGo lets commuters and travelers share rides comfortably. You book verified seats with SafeVault protection and split highway fuel costs safely.",
        "Whether commuting between cities or heading to the airport, BawaGo Ride lets you book seats with background-checked drivers traveling along your exact route."
    ],
    "crave-how-it-works": [
        "Crave connects you with verified neighborhood home cooks who prepare authentic, regional homemade meals in small hygienic batches. You can browse menus, pre-order meals, or subscribe to daily home tiffins.",
        "Missing authentic regional home cooking? Crave pairs you with passionate local home chefs. All meals are prepared fresh in sanitary kitchens and sealed with tamper-evident packaging.",
        "Through Crave Home Food, verified home cooks share their culinary heritage. You enjoy healthy, preservative-free home meals delivered straight to your door with escrow protection."
    ]
}

def synthesize_dynamic_response(
    intent: str, 
    category: str, 
    user_message: str, 
    session: SessionState, 
    base_answer: str
) -> str:
    """
    Synthesizes a clean, friendly, conversational response without dry jargon,
    handling contextual continuity and anti-duplication memory.
    """
    # 1. Attempt dynamic LLM generation
    llm_output = call_gemma_llm(user_message, base_answer, session)
    if llm_output and not session.is_duplicate_response(llm_output):
        return llm_output

    # 2. Select from unrepeated template variations if available
    templates = INTENT_VARIATIONS.get(intent, [base_answer])
    available = [t for t in templates if not session.is_duplicate_response(t)]
    if not available:
        available = templates

    chosen = random.choice(available)

    # Clean any leftover policy markers
    clean_ans = re.sub(r"Policy §\d+\.\d+", "BawaGo safety terms", chosen)
    clean_ans = re.sub(r"Under BawaGo .*? Policy,?\s*", "", clean_ans)

    # Conversational opener if user has dialogue history
    openers = []
    if len(session.history) >= 2 and random.random() > 0.5:
        openers = ["To clarify further, ", "To help with that, ", "Certainly! ", "Great question! "]
    prefix = random.choice(openers) if openers and random.random() > 0.6 else ""

    full_answer = prefix + clean_ans
    return full_answer

def compute_dynamic_chips(intent: str, category: str, session: SessionState) -> List[str]:
    """Generates 3-4 dynamic quick-reply chips tailored to the active category."""
    chips = CATEGORY_CHIPS.get(category, CATEGORY_CHIPS["General"])
    return chips[:4]

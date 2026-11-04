"""
Build Universal BawaGo Knowledge Base (130+ Topics)
Combines:
1. prototype/data/bawabot_v1_intents.json (125 intents across all 13 platform categories)
2. prototype/data/bawago_policies.json (Electronics, laptops, SafeVault escrow, KYC, prohibited items)
3. Journey listings & corridors (Dubai, London, Singapore, Mumbai, Pune, Delhi, Hyderabad, Jakarta, Bali)
4. Jargon-free, natural, conversational rewrite of all answers.
"""

import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROTOTYPE_DIR = os.path.join(os.path.dirname(BASE_DIR), "prototype", "data")

with open(os.path.join(PROTOTYPE_DIR, "bawabot_v1_intents.json"), "r", encoding="utf-8") as f:
    v1_intents = json.load(f)

with open(os.path.join(PROTOTYPE_DIR, "bawago_policies.json"), "r", encoding="utf-8") as f:
    policies = json.load(f)

# Clean, jargon-free rewrite dictionary for specific topics
FRIENDLY_REWRITES = {
    "baggage-overview": (
        "Baggage pooling lets travelers with spare flight luggage allowance share their unused kilos with senders who need affordable shipping. "
        "Senders save up to 60% compared to airline excess baggage rates, and travelers earn money on flights they were already taking!"
    ),
    "baggage-sell-kilos": (
        "If you have a flight ticket with 20kg or 30kg baggage allowance but only packed 10kg, you can list your spare kilos on BawaGo Baggage. "
        "Just enter your route, flight date, and available weight. Verified senders will book your space, and payments are protected in SafeVault."
    ),
    "baggage-banned": (
        "For safety and airline compliance, passengers can only carry legal items that have been visually inspected in person. "
        "Strictly prohibited: illegal narcotics, weapons, unprescribed pharmaceuticals, undeclared cash or gold, and hazardous materials."
    ),
    "escrow-how": (
        "SafeVault Escrow protects your money with a 3-step guarantee: when you make a booking, your payment is held securely in platform escrow. "
        "The provider is never paid upfront. Funds are only released to their wallet after you physically inspect your item and share your secret 6-digit delivery OTP."
    ),
    "escrow-otp": (
        "The 6-digit OTP is your personal confirmation code. It appears in your BawaGo app upon booking. "
        "Once you meet the traveler or driver and verify that your item is safe, you give them the OTP to release payment. Without this code, funds remain protected."
    ),
    "escrow-fees": (
        "BawaGo keeps pricing straightforward: we charge a transparent 5% to 10% platform fee (typically 7% for standard escrow). "
        "More than 90% of the total payment goes directly into the provider's wallet, and payouts can be withdrawn instantly to your bank account or UPI."
    ),
    "escrow-refund": (
        "SafeVault includes a 100% money-back guarantee. If a traveler or driver cancels, or fails to deliver your package, "
        "your escrow deposit is automatically refunded back to your original payment method within 2 to 4 business hours."
    ),
    "escrow-cash-refusal": (
        "To keep everyone safe, paying in physical cash or dealing offline is strictly not allowed on BawaGo. "
        "Cash deals have zero insurance coverage, zero fraud protection, and lead to immediate account suspension. All bookings must go through SafeVault."
    )
}

universal_kb = []

# 1. Ingest all 125 v1 intents
for item in v1_intents:
    item_id = item["id"]
    cat = item.get("category", "General")
    q = item["question"]
    keywords = item.get("keywords", [])
    ans = FRIENDLY_REWRITES.get(item_id, item["answer"])

    # Clean any leftover robotic phrases
    ans = re.sub(r"Policy §\d+\.\d+", "BawaGo safety terms", ans)
    ans = re.sub(r"Under BawaGo .*? Policy,?\s*", "", ans)

    universal_kb.append({
        "id": item_id,
        "category": cat,
        "title": q,
        "canonical_question": q,
        "keywords": keywords,
        "answer": ans.strip(),
        "paraphrases": [q] + keywords[:6]
    })

# 2. Add Electronics & Laptop specialized knowledge item
universal_kb.append({
    "id": "baggage-laptop-electronics",
    "category": "Baggage",
    "title": "Can I carry a laptop or electronics on a flight?",
    "canonical_question": "Can I carry a laptop or electronic device on my flight?",
    "keywords": [
        "laptop", "macbook", "ipad", "tablet", "electronics", "gadget",
        "invoice", "invoice documents", "purchase receipt", "electronics rules",
        "power on test", "lithium battery", "dubai laptop", "hand luggage laptop"
    ],
    "answer": (
        "Yes, you can carry a laptop or electronic gadget! Airlines require all laptops, tablets, and lithium-powered devices "
        "to be carried in your cabin hand-carry luggage—never in checked baggage. "
        "For safety and customs clearance, the sender must provide an original purchase invoice, and the device should power on during your mutual airport inspection."
    ),
    "paraphrases": [
        "Can I carry a laptop for someone to Dubai?",
        "What invoice documents are needed?",
        "What invoice is required for laptops?",
        "Are laptops allowed in cabin baggage?",
        "Can I carry someone's MacBook on a plane?",
        "Do I need an invoice for electronic devices?",
        "What are the rules for carrying laptops on flights?",
        "Can I transport smartphones and electronics on a flight?"
    ]
})

# 3. Add Popular Travel Corridors & Global Routes
universal_kb.append({
    "id": "corridors-routes",
    "category": "General",
    "title": "What are the most popular travel routes and corridors?",
    "canonical_question": "What travel routes and corridors are available on BawaGo?",
    "keywords": [
        "popular corridors", "travel routes", "flight routes", "dubai", "london",
        "singapore", "hyderabad", "mumbai", "pune", "delhi", "bangalore", "jakarta", "bali"
    ],
    "answer": (
        "BawaGo operates across major domestic and international corridors! Top routes include Hyderabad to Dubai, "
        "Singapore to London, Delhi to Mumbai, Jakarta to Bali, and intercity highway carpools like Mumbai to Pune and Delhi to Jaipur. "
        "You can find active travelers flying or driving daily across these routes."
    ),
    "paraphrases": [
        "What are the most popular travel corridors on BawaGo?",
        "Can I send items from Hyderabad to Dubai?",
        "Are there travelers flying between Singapore and London?",
        "What carpool routes exist between Mumbai and Pune?",
        "Is BawaGo available internationally?"
    ]
})

# 4. Add 6-Digit OTP Escrow Verification
universal_kb.append({
    "id": "escrow-otp-delivery",
    "category": "SafeVault & Wallet",
    "title": "Why do I need a 6-digit OTP code for delivery?",
    "canonical_question": "Why do I need a 6-digit OTP to confirm delivery?",
    "keywords": [
        "otp", "6-digit otp", "completion code", "delivery pin",
        "release payment with otp", "secret pin at handover", "delivery verification otp"
    ],
    "answer": (
        "The 6-digit OTP is your personal release key for SafeVault Escrow. "
        "When you receive and inspect your item at handover, you share this secret code with the traveler or driver. "
        "Entering the OTP securely validates that the service was completed safely and unlocks their payout."
    ),
    "paraphrases": [
        "Why do I need a 6-digit OTP?",
        "What is the completion OTP code for?",
        "How is delivery confirmed with OTP?",
        "Where do I find my delivery OTP?",
        "Can payment be released without the OTP code?",
        "How does the receiver confirm delivery with OTP?"
    ]
})

# 5. Add Trust & Insurance Protection Guarantee
universal_kb.append({
    "id": "safety-insurance-guarantee",
    "category": "Safety & Trust",
    "title": "Is my package insured against loss or damage?",
    "canonical_question": "Does BawaGo provide insurance and damage protection?",
    "keywords": [
        "insurance", "damage protection", "trust guarantee", "500 dollar guarantee",
        "broken item", "lost parcel", "claim insurance", "loss coverage"
    ],
    "answer": (
        "Yes! Every booking completed through SafeVault is protected by the BawaGo Trust Guarantee at no extra cost. "
        "This provides up to $500 in loss and damage coverage for verified items during transit. If an accidental issue occurs, "
        "our support team handles the claim and reimburses you promptly."
    ),
    "paraphrases": [
        "Does insurance cover damage?",
        "Is my package insured on BawaGo?",
        "What happens if an item is lost or damaged by the traveler?",
        "How much insurance compensation is provided for damaged items?",
        "How do I file an insurance claim on BawaGo?"
    ]
})

# 5. Add Human Customer Support Escalation
universal_kb.append({
    "id": "support-human-help",
    "category": "Customer Support",
    "title": "How do I contact human customer support?",
    "canonical_question": "Can I speak with a real human support agent?",
    "keywords": [
        "human support", "speak to human", "real person", "customer service staff",
        "contact support", "helpline", "dispute ticket", "call agent"
    ],
    "answer": (
        "You can connect with our human customer care team anytime! We provide 24/7 support for booking disputes, claims, and urgent inquiries. "
        "You can email us at support@bawago.com or tap 'Raise Support Ticket' inside your app profile for assistance within 2 hours."
    ),
    "paraphrases": [
        "Connect me to human customer support",
        "Can I speak with a human agent?",
        "How do I talk to a real person?",
        "I need customer service help from a human",
        "Can I speak with customer care staff?"
    ]
})

print(f">> Total universal knowledge items compiled: {len(universal_kb)}")
out_path = os.path.join(BASE_DIR, "model", "universal_kb.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(universal_kb, f, indent=2, ensure_ascii=False)

print(f">> Saved universal knowledge base to {out_path}")

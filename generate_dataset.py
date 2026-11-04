"""
Dataset Generator for BawaGo Isolated AI Agent
Pairs website-related questions with Gemma-generated answers using BawaGo policy grounding.
"""

import os
import json
import urllib.request
from typing import Dict, List, Any

TOPICS_DEFINITIONS = [
    {
        "intent": "greeting",
        "category": "General",
        "policy_context": "BawaGo is a 7-vertical peer-to-peer platform connecting verified travelers, drivers, hosts, and consumers with SafeVault Escrow protection.",
        "gemma_seed_prompt": "Hello! Introduce yourself as the official BawaGo AI Agent. Welcome the user and explain that you can help with all 7 BawaGo services (Ride, Parcel, Baggage, Crave Food, Host, Wish, Fresh) and SafeVault Escrow in 2 friendly sentences.",
        "questions": [
            "Hi there",
            "Hello BawaGo",
            "Hey",
            "Good morning",
            "Greetings",
            "Is anyone here?",
            "What can you do?",
            "Who are you?",
            "Help me get started",
            "Namaste BawaGo"
        ]
    },
    {
        "intent": "about_bawago",
        "category": "General",
        "policy_context": "BawaGo is a community marketplace across 7 verticals: Ride (carpooling), Parcel (delivery), Baggage (luggage pooling), Crave Food (home-cooked meals), Host (homestays), Wish (overseas shopping), and Fresh (farm produce). All transactions are secured by SafeVault Escrow.",
        "gemma_seed_prompt": "Explain what BawaGo is and summarize its 7 verticals and SafeVault escrow in 2-3 clear sentences.",
        "questions": [
            "What is BawaGo?",
            "Can you explain what this website does?",
            "What services does BawaGo provide?",
            "Tell me about BawaGo platform",
            "How does the BawaGo community work?",
            "What are the 7 verticals of BawaGo?",
            "What is the concept behind BawaGo?",
            "Why should I use BawaGo?",
            "Is BawaGo a courier company?"
        ]
    },
    {
        "intent": "safevault_escrow_how_it_works",
        "category": "SafeVault & Escrow",
        "policy_context": "BawaGo SafeVault Escrow holds payments in automated trust. Senders pay upfront, but funds are never released to travelers until the receiver inspects the delivery and shares a 6-digit OTP. This eliminates fraud and non-delivery disputes.",
        "gemma_seed_prompt": "Explain clearly how SafeVault Escrow protects buyer money and when funds are released to the traveler.",
        "questions": [
            "How does SafeVault Escrow work?",
            "Is my payment safe on BawaGo?",
            "How do I know the traveler will not take my money and disappear?",
            "Explain the escrow process on BawaGo",
            "When does the traveler get paid?",
            "How does payment protection work on this site?",
            "What is SafeVault?",
            "Why do you hold funds in escrow?",
            "Can a traveler steal my funds?"
        ]
    },
    {
        "intent": "safevault_otp_verification",
        "category": "SafeVault & Escrow",
        "policy_context": "Delivery completion requires a secure 6-digit one-time password (OTP) generated on the receiver's app. The traveler cannot claim payment without entering this OTP, ensuring mutual verification.",
        "gemma_seed_prompt": "Explain the role of the 6-digit OTP code in confirming deliveries and releasing escrow payments.",
        "questions": [
            "What is the completion OTP code for?",
            "How do I confirm that I received my parcel?",
            "Why do I need a 6 digit code for delivery?",
            "How do travelers prove they delivered the baggage?",
            "Where do I find my delivery OTP?",
            "Can payment be released without the OTP?",
            "What happens if I do not share the OTP?"
        ]
    },
    {
        "intent": "safevault_fees_and_payouts",
        "category": "SafeVault & Escrow",
        "policy_context": "BawaGo charges a modest platform maintenance fee of 5% to 10% (typically 7% for SafeVault Escrow). Over 90% of earnings go directly to travelers and drivers, with instant wallet payout after OTP confirmation.",
        "gemma_seed_prompt": "Explain BawaGo's platform fee structure and how much travelers earn from deliveries.",
        "questions": [
            "What are the platform fees on BawaGo?",
            "How much commission does BawaGo take?",
            "How much money do travelers keep?",
            "Are there hidden charges on SafeVault payments?",
            "What percentage does BawaGo charge for escrow?",
            "How fast do travelers receive their earnings?"
        ]
    },
    {
        "intent": "baggage_sharing_concept",
        "category": "Baggage",
        "policy_context": "BawaGo Baggage connects passengers with excess unused baggage kilos on scheduled flights with senders needing cost-effective transport, saving up to 60% compared to airline excess freight.",
        "gemma_seed_prompt": "Explain BawaGo Baggage pooling and how airline travelers share spare luggage space.",
        "questions": [
            "How does baggage pooling work?",
            "Can I sell my unused flight baggage kilos?",
            "How do I find a traveler flying with empty luggage space?",
            "Can I share my luggage allowance on an airplane?",
            "How does BawaGo save money on excess baggage?",
            "How much can I charge per kilo of spare baggage?",
            "What is the baggage vertical on BawaGo?"
        ]
    },
    {
        "intent": "baggage_laptop_electronics_rule",
        "category": "Baggage",
        "policy_context": "According to BawaGo Baggage Policy §3.2 and aviation safety rules, laptops and devices with lithium batteries are strictly permitted in cabin baggage (hand carry) only, never checked-in. Senders must provide purchase invoice and inspect the power-on state at the airport.",
        "gemma_seed_prompt": "Explain the exact rules for transporting a laptop or lithium battery electronics for someone on a flight.",
        "questions": [
            "Can I carry a laptop for someone to Dubai?",
            "Is it permitted to bring a notebook computer on my flight?",
            "Are laptops allowed in checked baggage?",
            "What are the rules for carrying electronic gadgets on BawaGo?",
            "Can someone send a tablet or computer through a traveler?",
            "What documents do I need to carry an electronic device?",
            "Why must laptops be carried in the cabin?"
        ]
    },
    {
        "intent": "baggage_weight_limits",
        "category": "Baggage",
        "policy_context": "Standard baggage pooling complies with international airline IATA guidelines: single checked bags must not exceed 23kg (economy) or 32kg (business). Items must fit standard dimensions.",
        "gemma_seed_prompt": "Explain the weight restrictions and allowances for baggage pooling on passenger flights.",
        "questions": [
            "What is the maximum weight I can carry for someone?",
            "How many kilos can I send through a traveler?",
            "What are the weight limits on airline baggage pooling?",
            "Can I send a 30kg suitcase through BawaGo?",
            "Are there dimension limits on baggage?"
        ]
    },
    {
        "intent": "parcel_delivery_overview",
        "category": "Parcel",
        "policy_context": "BawaGo Parcel connects senders with everyday commuters and travelers traveling between cities for fast, same-day or next-day peer-to-peer package delivery without traditional courier delays.",
        "gemma_seed_prompt": "Explain how BawaGo peer-to-peer parcel delivery works and its key advantages over regular couriers.",
        "questions": [
            "How does peer-to-peer parcel delivery work?",
            "Can I send an urgent document or package to another city?",
            "How does BawaGo Parcel differ from DHL or FedEx?",
            "How fast is peer to peer parcel shipping?",
            "Can someone pick up and drop my package along their route?",
            "How do I book a parcel courier on BawaGo?"
        ]
    },
    {
        "intent": "parcel_prohibited_items",
        "category": "Parcel",
        "policy_context": "Strict safety rules prohibit weapons, narcotics, inflammable chemicals, counterfeit goods, currency notes, and undeclared customs contraband. Travelers must inspect open packages before sealing.",
        "gemma_seed_prompt": "List the prohibited and restricted items that cannot be sent through BawaGo Parcel or Baggage.",
        "questions": [
            "What items are prohibited on BawaGo?",
            "Can I send medicine or prescription drugs?",
            "Is it allowed to send cash or jewelry?",
            "What happens if someone asks me to carry an illegal item?",
            "Can travelers inspect the parcel contents?",
            "Can I send liquids or flammable goods?"
        ]
    },
    {
        "intent": "ride_sharing_carpool",
        "category": "Ride",
        "policy_context": "BawaGo Ride enables intercity carpooling and airport transfers. Verified drivers share empty car seats with passengers along their journey, cutting commute costs while reducing carbon emissions.",
        "gemma_seed_prompt": "Explain how BawaGo Ride carpooling works for drivers and passengers.",
        "questions": [
            "How does BawaGo Ride work?",
            "Can I find a carpool to the airport?",
            "Can I offer empty seats in my car to make money?",
            "Is intercity ride sharing safe on BawaGo?",
            "How do I book a seat on a shared ride?",
            "What are the fares for carpooling on BawaGo?"
        ]
    },
    {
        "intent": "home_food_crave",
        "category": "Home Food",
        "policy_context": "Crave Home Food connects authentic local home chefs with food lovers craving traditional regional dishes, prepared fresh in home kitchens following certified hygiene practices.",
        "gemma_seed_prompt": "Explain the Crave Home Food vertical and how users can order authentic homemade dishes.",
        "questions": [
            "What is Crave Home Food on BawaGo?",
            "Can I order authentic homemade regional food?",
            "Can home chefs sell food on BawaGo?",
            "How do I get fresh homemade meals delivered?",
            "What kind of home food is available on BawaGo?",
            "Is Crave food prepared in restaurants or home kitchens?"
        ]
    },
    {
        "intent": "home_food_hygiene_safety",
        "category": "Home Food",
        "policy_context": "All home chefs must provide food safety certifications (such as FSSAI registration or municipal health licenses), list allergens clearly, and use tamper-evident packaging.",
        "gemma_seed_prompt": "Explain the health and hygiene standards enforced for home chefs on BawaGo Crave Food.",
        "questions": [
            "Is home cooked food on BawaGo hygienic and safe?",
            "Do home chefs need a license or food safety certificate?",
            "How are food allergens handled?",
            "Are kitchen inspections required for BawaGo food cooks?",
            "What packaging standards are required for food deliveries?"
        ]
    },
    {
        "intent": "host_stays_homestay",
        "category": "Host",
        "policy_context": "BawaGo Host allows local hosts to list spare rooms, apartments, and vacation homestays. Guests book through SafeVault escrow, guaranteeing check-in security and accurate listing photos.",
        "gemma_seed_prompt": "Explain the BawaGo Host vertical and how travelers can book homestays safely.",
        "questions": [
            "How does BawaGo Host work?",
            "Can I rent out my spare room to travelers?",
            "Is booking a room on BawaGo safe?",
            "How does BawaGo Host compare to Airbnb?",
            "When does the host get paid for a stay?",
            "Can I find budget homestays near airports?"
        ]
    },
    {
        "intent": "wish_personal_shopper",
        "category": "Wish",
        "policy_context": "BawaGo Wish allows users to request specialty products, snacks, electronics, or fashion items from other countries. International travelers purchase and hand-deliver them upon arrival.",
        "gemma_seed_prompt": "Explain how the BawaGo Wish personal shopping service works with overseas travelers.",
        "questions": [
            "What is BawaGo Wish?",
            "Can I ask a traveler to buy something for me from abroad?",
            "How do I request overseas chocolates or gadgets?",
            "How does payment work when a traveler shops for me?",
            "What if the traveler buys the wrong item?",
            "Can I make an international shopping request?"
        ]
    },
    {
        "intent": "fresh_produce_farm",
        "category": "Fresh",
        "policy_context": "BawaGo Fresh connects organic farms and local growers directly with urban households for seasonal fruits, vegetables, and farm-fresh dairy with farm-to-table traceability.",
        "gemma_seed_prompt": "Explain BawaGo Fresh and how farm-to-table direct delivery operates.",
        "questions": [
            "What is BawaGo Fresh?",
            "Can I buy farm fresh vegetables and fruit directly from farmers?",
            "How does farm-to-table delivery work on BawaGo?",
            "Are BawaGo Fresh products organic?",
            "How quickly is fresh produce delivered after harvest?"
        ]
    },
    {
        "intent": "trust_kyc_verification",
        "category": "Safety & Trust",
        "policy_context": "BawaGo enforces mandatory multi-factor KYC: government identity verification, live biometric face matching, and flight PNR verification for travelers before accepting orders.",
        "gemma_seed_prompt": "Explain BawaGo's identity verification process and how travelers are vetted for platform safety.",
        "questions": [
            "How does BawaGo verify traveler identity?",
            "What documents are needed to become a traveler or host?",
            "Are all users on BawaGo background-checked?",
            "Do I need to submit my passport to carry items?",
            "How does BawaGo verify flight tickets?",
            "Is my personal identification data kept secure?"
        ]
    },
    {
        "intent": "fraud_cash_bypass_refusal",
        "category": "Safety & Trust",
        "policy_context": "Under BawaGo SafeVault Policy §1.1, off-platform cash deals and direct offline transfers are strictly prohibited. Any bypass of escrow voids dispute protection and results in immediate account suspension.",
        "gemma_seed_prompt": "Explain why paying in cash or outside the BawaGo platform is strictly prohibited and what risks it poses.",
        "questions": [
            "Can I pay the traveler directly in cash?",
            "Can we do this transaction outside the app to avoid fees?",
            "Can I pay cash on delivery directly to the driver?",
            "Is cash payment accepted on BawaGo?",
            "Why can't I pay directly through personal bank transfer?",
            "What happens if someone asks me to pay in cash?"
        ]
    },
    {
        "intent": "human_support_escalation",
        "category": "Customer Support",
        "policy_context": "BawaGo provides 24/7 priority support with automated ticket generation (TCK-xxxx) for complex disputes, lost baggage claims, and emergency transit assistance with an average SLA under 2 hours.",
        "gemma_seed_prompt": "Explain how users can reach a human customer support specialist if the AI agent cannot resolve their issue.",
        "questions": [
            "Can I speak with a human agent?",
            "I need human support",
            "How do I talk to a live person?",
            "Connect me to customer service staff",
            "I want to raise a complaint with management",
            "Where can I contact BawaGo customer support?"
        ]
    },
    {
        "intent": "refund_policy",
        "category": "SafeVault & Escrow",
        "policy_context": "If a traveler fails to deliver an item or cancels before pickup, SafeVault Escrow issues a 100% full refund back to the sender's original payment method within 2 to 4 business hours.",
        "gemma_seed_prompt": "Explain BawaGo's refund policy when a traveler cancels or fails to deliver an order.",
        "questions": [
            "What is BawaGo refund policy?",
            "Do I get a refund if the traveler does not show up?",
            "How long do refunds take on SafeVault?",
            "What if my package gets damaged during transit?",
            "Can I get my money back if delivery fails?",
            "What is the dispute resolution process for payments?"
        ]
    },
    {
        "intent": "traveler_earnings_guide",
        "category": "Earn with Us",
        "policy_context": "Travelers monetize empty luggage kilos on flights they are already taking. Typical earnings range from $15 to $35 per kilogram, easily offsetting $200-$500 of their flight ticket costs per trip.",
        "gemma_seed_prompt": "Explain how travelers can earn money by monetizing unused baggage allowance on their trips.",
        "questions": [
            "How much money can I make as a traveler?",
            "How do I earn by carrying parcels on my flight?",
            "Can I subsidize my travel ticket costs with BawaGo?",
            "How do travelers set their prices per kilo?",
            "How do I list my travel capacity and available dates?"
        ]
    },
    {
        "intent": "driver_earnings_guide",
        "category": "Earn with Us",
        "policy_context": "Drivers earn on daily commutes or intercity trips by filling empty seats or trunk capacity. Drivers keep up to 95% of seat fares with automated weekly or instant wallet payouts.",
        "gemma_seed_prompt": "Explain how vehicle owners and drivers can earn extra income through BawaGo Ride and Parcel.",
        "questions": [
            "How do drivers earn on BawaGo?",
            "Can I make money driving between cities?",
            "What vehicle types are eligible for BawaGo Ride?",
            "How much does a driver make per trip?",
            "How do driver payouts work?"
        ]
    },
    {
        "intent": "booking_cancellation_rules",
        "category": "SafeVault & Escrow",
        "policy_context": "Free cancellation is permitted up to 24 hours before scheduled handover. Cancellations within 12 hours incur a small 10% re-listing fee to compensate the traveler for reserved luggage space.",
        "gemma_seed_prompt": "Explain the cancellation rules and fees for senders and travelers on BawaGo.",
        "questions": [
            "Can I cancel a booking after paying?",
            "What is the cancellation policy on BawaGo?",
            "Are there fees if I cancel my order?",
            "What if the traveler cancels my flight booking?",
            "How do I cancel a confirmed journey?"
        ]
    },
    {
        "intent": "popular_travel_corridors",
        "category": "General",
        "policy_context": "Top high-volume passenger corridors include Hyderabad to Dubai, Singapore to London, Delhi to Mumbai, Jakarta to Bali, and Dubai local routes, with verified daily traveler departures.",
        "gemma_seed_prompt": "List the most active and popular travel corridors on BawaGo for baggage and parcel delivery.",
        "questions": [
            "What are the most popular routes on BawaGo?",
            "Can I send items from Hyderabad to Dubai?",
            "Is there baggage pooling between Singapore and London?",
            "Are domestic routes like Delhi to Mumbai supported?",
            "Which international flight corridors have the most travelers?"
        ]
    },
    {
        "intent": "insurance_and_damage_protection",
        "category": "Safety & Trust",
        "policy_context": "Every order funded through SafeVault includes the BawaGo Trust Guarantee protecting verified goods against accidental loss or damage up to $500, with optional extended valuation coverage.",
        "gemma_seed_prompt": "Explain the insurance and damage protection guarantee provided under SafeVault Escrow.",
        "questions": [
            "Is my package insured on BawaGo?",
            "What happens if an item is lost or damaged by the traveler?",
            "Does BawaGo provide damage protection?",
            "What is the maximum coverage under SafeVault Guarantee?",
            "How do I file a claim for lost or damaged goods?"
        ]
    }
]

def query_gemma(prompt: str, context: str) -> str:
    """Queries local Gemma 2B via Ollama to generate an authoritative answer."""
    full_prompt = (
        "You are the official BawaGo AI Agent.\n"
        f"Official Context:\n{context}\n\n"
        f"Task: {prompt}\n"
        "Provide a clear, authoritative, polite, and practical answer in 2-4 sentences:"
    )
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model": "gemma:2b",
                "messages": [{"role": "user", "content": full_prompt}],
                "temperature": 0.2,
                "max_tokens": 150
            }).encode("utf-8")
        )
        with urllib.request.urlopen(req, timeout=35) as res:
            data = json.loads(res.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            if content:
                return content
    except Exception as e:
        print(f"Warning: Gemma call failed ({e}); falling back to policy context.")
    return context

def build_dataset():
    print(f">> Generating dataset from {len(TOPICS_DEFINITIONS)} core BawaGo topics...")
    records = []
    
    for i, topic in enumerate(TOPICS_DEFINITIONS):
        intent = topic["intent"]
        category = topic["category"]
        print(f"[{i+1}/{len(TOPICS_DEFINITIONS)}] Generating Gemma answer for '{intent}'...")
        
        gemma_answer = query_gemma(topic["gemma_seed_prompt"], topic["policy_context"])
        
        # Add question pairs
        for q in topic["questions"]:
            records.append({
                "question": q.strip(),
                "intent": intent,
                "category": category,
                "answer": gemma_answer,
                "source_policy": topic["policy_context"]
            })

    output_path = "isolated_agent_site/model/dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    
    print(f"\n>> Successfully built dataset with {len(records)} Q&A samples saved to {output_path}!")

if __name__ == "__main__":
    build_dataset()


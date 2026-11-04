"""
Comprehensive Question-by-Question Audit & Analysis
Tests all questions across all 25 intents, starter chips, dynamic chips, and cross-vertical queries.
"""
import json
import joblib
import numpy as np

model = joblib.load('isolated_agent_site/model/trained_agent_model.joblib')

with open('isolated_agent_site/model/intent_answers.json', 'r', encoding='utf-8') as f:
    intent_answers = json.load(f)

# Comprehensive test battery: 60 realistic questions covering all intents & chips
AUDIT_SUITE = [
    # Baggage / Electronics / Invoice
    ("Can I carry a laptop to Dubai?", "baggage_laptop_electronics_rule"),
    ("What invoice documents are needed?", "baggage_laptop_electronics_rule"),
    ("What invoice is required for laptops?", "baggage_laptop_electronics_rule"),
    ("Do I need an invoice for electronic devices?", "baggage_laptop_electronics_rule"),
    ("Are laptops allowed in cabin baggage?", "baggage_laptop_electronics_rule"),
    ("Can I put my MacBook in checked luggage?", "baggage_laptop_electronics_rule"),
    ("What are the baggage weight limits?", "baggage_weight_limits"),
    ("What is the maximum luggage weight?", "baggage_weight_limits"),
    ("How many kilograms can I carry on a flight?", "baggage_weight_limits"),
    ("How does baggage pooling work?", "baggage_sharing_concept"),
    ("Can I share my unused flight baggage kilos?", "baggage_sharing_concept"),
    
    # SafeVault Escrow & Payments
    ("How does SafeVault Escrow protect my money?", "safevault_escrow_how_it_works"),
    ("How does SafeVault work?", "safevault_escrow_how_it_works"),
    ("How does SafeVault hold buyer funds?", "safevault_escrow_how_it_works"),
    ("Is my payment safe on BawaGo?", "safevault_escrow_how_it_works"),
    ("Why do I need a 6-digit OTP?", "safevault_otp_verification"),
    ("What is the completion OTP code for?", "safevault_otp_verification"),
    ("How is payment released with OTP?", "safevault_otp_verification"),
    ("What percentage does BawaGo charge?", "safevault_fees_and_payouts"),
    ("What are the platform fees?", "safevault_fees_and_payouts"),
    ("How much commission does BawaGo take?", "safevault_fees_and_payouts"),
    
    # Parcel
    ("Can I send urgent parcels to another city?", "parcel_delivery_overview"),
    ("Can I send an urgent parcel?", "parcel_delivery_overview"),
    ("How fast is peer to peer parcel shipping?", "parcel_delivery_overview"),
    ("What items are prohibited?", "parcel_prohibited_items"),
    ("Can I send medicine or drugs?", "parcel_prohibited_items"),
    ("What things are banned from being delivered?", "parcel_prohibited_items"),
    
    # Ride / Carpool
    ("How does carpooling work?", "ride_sharing_carpool"),
    ("How does BawaGo Ride carpooling work?", "ride_sharing_carpool"),
    ("Can I book a seat in a car?", "ride_sharing_carpool"),
    ("How do driver fuel payouts work?", "driver_earnings_guide"),
    ("How much do drivers earn?", "driver_earnings_guide"),
    ("How do vehicle owners make money?", "driver_earnings_guide"),
    
    # Crave Home Food
    ("Can I order authentic homemade regional food?", "home_food_crave"),
    ("What is Crave Home Food on BawaGo?", "home_food_crave"),
    ("Can local home chefs register?", "home_food_crave"),
    ("Is home food hygienic and safe?", "home_food_hygiene_safety"),
    ("What hygiene standards are required?", "home_food_hygiene_safety"),
    ("Do home chefs need a food safety license?", "home_food_hygiene_safety"),
    
    # Host / Stays
    ("How do I list a spare room for rent?", "host_stays_homestay"),
    ("How do I list a spare room?", "host_stays_homestay"),
    ("When does the host get paid?", "host_stays_homestay"),
    
    # Wish Personal Shopper
    ("Can a traveler shop at duty free for me?", "wish_personal_shopper"),
    ("Can travelers shop at duty free for me?", "wish_personal_shopper"),
    ("How does escrow hold funds for shopping?", "wish_personal_shopper"),
    
    # Fresh Farm Produce
    ("Can I buy fresh vegetables directly from farmers?", "fresh_produce_farm"),
    ("What is BawaGo Fresh?", "fresh_produce_farm"),
    
    # Trust / KYC / Anti-Fraud / Cash
    ("How does BawaGo verify traveler identity?", "trust_kyc_verification"),
    ("How does BawaGo verify driver identity?", "trust_kyc_verification"),
    ("What documents are needed to verify my account?", "trust_kyc_verification"),
    ("Can I pay the driver in physical cash?", "fraud_cash_bypass_refusal"),
    ("Why are cash payments prohibited?", "fraud_cash_bypass_refusal"),
    ("Why must all deals use SafeVault?", "fraud_cash_bypass_refusal"),
    
    # Policies / Support / General
    ("Connect me to human customer support", "human_support_escalation"),
    ("How do I speak with human support?", "human_support_escalation"),
    ("What is the refund policy?", "refund_policy"),
    ("What if the traveler cancels?", "booking_cancellation_rules"),
    ("What is the cancellation policy?", "booking_cancellation_rules"),
    ("How much can a traveler earn?", "traveler_earnings_guide"),
    ("What are the most popular travel corridors?", "popular_travel_corridors"),
    ("Does insurance cover damage?", "insurance_and_damage_protection"),
    ("What is BawaGo?", "about_bawago"),
    ("Hello", "greeting")
]

print(f"Total audit questions: {len(AUDIT_SUITE)}")
passed = 0
failed = []

for q, expected in AUDIT_SUITE:
    cleaned = q.lower().strip()
    pred = model.predict([cleaned])[0]
    probs = model.predict_proba([cleaned])[0]
    conf = float(np.max(probs))
    
    if pred == expected:
        passed += 1
    else:
        # Get top 2
        classes = model.classes_
        top2_idx = np.argsort(probs)[::-1][:2]
        top2_info = [(classes[i], probs[i]) for i in top2_idx]
        failed.append((q, expected, pred, conf, top2_info))

print(f"\nRESULTS: {passed}/{len(AUDIT_SUITE)} Passed ({len(failed)} Failed)")
print("=" * 70)
for q, exp, pr, conf, top2 in failed:
    print(f"FAIL: '{q}'")
    print(f"      Expected: {exp}")
    print(f"      Got:      {pr} ({conf*100:.1f}%)")
    print(f"      Top 2:    {top2}")


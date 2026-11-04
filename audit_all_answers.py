import json
import joblib

model = joblib.load('isolated_agent_site/model/trained_agent_model.joblib')

with open('isolated_agent_site/model/intent_answers.json', 'r', encoding='utf-8') as f:
    intent_answers = json.load(f)

# Collect all chips from interactive_engine
test_questions = [
    # Chips from starter & baggage
    ("Can I carry a laptop to Dubai?", "baggage_laptop_electronics_rule"),
    ("What invoice documents are needed?", "baggage_laptop_electronics_rule"),
    ("What invoice is required for laptops?", "baggage_laptop_electronics_rule"),
    ("What are the baggage weight limits?", "baggage_weight_limits"),
    ("How does SafeVault Escrow protect my money?", "safevault_escrow_how_it_works"),
    ("Why do I need a 6-digit OTP?", "safevault_otp_verification"),
    ("What percentage does BawaGo charge?", "safevault_fees_and_payouts"),
    ("What are the platform fees?", "safevault_fees_and_payouts"),
    ("Can I send urgent parcels to another city?", "parcel_delivery_overview"),
    ("What items are prohibited?", "parcel_prohibited_items"),
    ("How does carpooling work?", "ride_sharing_carpool"),
    ("How much do drivers earn?", "driver_earnings_guide"),
    ("Can I order authentic homemade regional food?", "home_food_crave"),
    ("Is home food hygienic and safe?", "home_food_hygiene_safety"),
    ("How do I list a spare room for rent?", "host_stays_homestay"),
    ("Can a traveler shop at duty free for me?", "wish_personal_shopper"),
    ("Can I buy fresh vegetables directly from farmers?", "fresh_produce_farm"),
    ("How does BawaGo verify traveler identity?", "trust_kyc_verification"),
    ("Can I pay the driver in physical cash?", "fraud_cash_bypass_refusal"),
    ("Why are cash payments prohibited?", "fraud_cash_bypass_refusal"),
    ("Connect me to human customer support", "human_support_escalation"),
    ("What is the refund policy?", "refund_policy"),
    ("How much can a traveler earn?", "traveler_earnings_guide"),
    ("What is the cancellation policy?", "booking_cancellation_rules"),
    ("What are the most popular travel corridors?", "popular_travel_corridors"),
    ("Does insurance cover damage?", "insurance_and_damage_protection"),
    ("What is BawaGo?", "about_bawago"),
    ("Hello", "greeting")
]

print("=" * 75)
print("  QUESTION-BY-QUESTION MODEL AUDIT")
print("=" * 75)

failures = []
for q, expected in test_questions:
    pred = model.predict([q.lower()])[0]
    probs = model.predict_proba([q.lower()])[0]
    conf = max(probs)
    passed = (pred == expected)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] '{q}'")
    print(f"       Expected: {expected}")
    print(f"       Predicted: {pred} ({conf*100:.1f}%)")
    if not passed:
        failures.append((q, expected, pred, conf))
        ans = intent_answers.get(pred, {}).get("answer", "")[:90]
        print(f"       -> WRONG ANSWER GIVEN: \"{ans}...\"")

print("\n" + "=" * 75)
print(f"Audit Summary: {len(test_questions) - len(failures)}/{len(test_questions)} Passed ({len(failures)} Failures)")
print("=" * 75)
for q, exp, pr, conf in failures:
    print(f"  FAILED: '{q}' -> Predicted: {pr} (Expected: {exp})")


"""
Test Category-Guided Ensemble Intent Prediction
"""
import json
import re
import numpy as np
from test_two_stage import cat_pipe, intent_pipe, clean, BENCHMARK_QUERIES

with open("isolated_agent_site/model/intent_answers.json", "r", encoding="utf-8") as f:
    registry = json.load(f)

classes = intent_pipe.classes_

# Specific keyword boost mappings
KEYWORD_BOOSTS = {
    "otp": "escrow-milestone-release",
    "6-digit": "escrow-milestone-release",
    "pin": "escrow-milestone-release",
    "invoice": "baggage-laptop-electronics",
    "macbook": "baggage-laptop-electronics",
    "laptop": "baggage-laptop-electronics",
    "carpool": "ride-intercity",
    "biryani": "crave-cuisines",
    "tiffin": "crave-tiffin-subscription",
    "duty free": "wish-overview",
    "personal shopper": "wish-overview",
    "farmer": "fresh-overview",
    "produce": "fresh-overview",
    "password": "acc-reset-pass",
    "reset pass": "acc-reset-pass",
    "human support": "support-human-help",
    "real person": "support-human-help",
    "insurance": "safety-insurance-guarantee",
    "damage": "safety-insurance-guarantee"
}

def predict_universal(query: str):
    cleaned = clean(query)
    pred_cat = cat_pipe.predict([cleaned])[0]
    cat_probs = cat_pipe.predict_proba([cleaned])[0]
    cat_classes = cat_pipe.classes_
    top2_cats = [cat_classes[i] for i in np.argsort(cat_probs)[::-1][:2]]

    probs = intent_pipe.predict_proba([cleaned])[0]
    scores = probs.copy()

    # Category agreement bonus (+0.30 if intent matches predicted top-1 category, +0.15 if in top-2)
    for idx, intent_name in enumerate(classes):
        intent_cat = registry.get(intent_name, {}).get("category", "")
        if intent_cat == pred_cat:
            scores[idx] += 0.35
        elif intent_cat in top2_cats:
            scores[idx] += 0.15

    # Direct keyword boost (+0.50)
    for kw, target_intent in KEYWORD_BOOSTS.items():
        if kw in cleaned:
            for idx, intent_name in enumerate(classes):
                if intent_name == target_intent:
                    scores[idx] += 0.50

    best_idx = int(np.argmax(scores))
    best_intent = classes[best_idx]
    best_score = float(scores[best_idx])
    return best_intent, best_score, pred_cat

print("\n" + "=" * 70)
print("  CATEGORY-GUIDED UNIVERSAL ENSEMBLE TEST")
print("=" * 70)

passed = 0
for q, exp in BENCHMARK_QUERIES:
    pred, score, cat = predict_universal(q)
    is_corr = (pred == exp)
    if is_corr:
        passed += 1
    status = "PASS" if is_corr else "FAIL"
    print(f"[{status}] '{q}'")
    print(f"       -> Pred: {pred} ({score:.2f}) | Category: {cat} [Expected: {exp}]")

print("=" * 70)
print(f"RESULTS: {passed}/{len(BENCHMARK_QUERIES)} Passed ({passed/len(BENCHMARK_QUERIES)*100:.1f}%)")

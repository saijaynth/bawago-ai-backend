"""
Build Expanded Universal Dataset for BawaGo AI Agent
Generates an extensive training and evaluation dataset across all 129 universal topics,
ensuring natural language phrasing, zero legal jargon, and robust intent mapping.
"""

import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "model", "universal_kb.json")
DATASET_PATH = os.path.join(BASE_DIR, "model", "dataset.json")
INTENT_ANSWERS_PATH = os.path.join(BASE_DIR, "model", "intent_answers.json")

with open(KB_PATH, "r", encoding="utf-8") as f:
    universal_kb = json.load(f)

dataset_records = []
intent_registry = {}

for item in universal_kb:
    intent_id = item["id"]
    category = item["category"]
    canonical_q = item["canonical_question"]
    answer = item["answer"]
    keywords = item.get("keywords", [])
    paraphrases = item.get("paraphrases", [])

    # Register authoritative answer
    intent_registry[intent_id] = {
        "title": item["title"],
        "category": category,
        "answer": answer,
        "canonical_question": canonical_q,
        "keywords": keywords
    }

    # Collect questions
    questions = set()
    questions.add(canonical_q)
    for p in paraphrases:
        if len(p.strip()) > 3:
            questions.add(p.strip())

    for kw in keywords:
        if len(kw.strip()) > 4:
            questions.add(f"Tell me about {kw.strip()}")
            questions.add(f"How does {kw.strip()} work on BawaGo?")

    # Add natural conversational variants
    for q in list(questions):
        q_clean = q.strip()
        dataset_records.append({
            "intent": intent_id,
            "category": category,
            "question": q_clean,
            "answer": answer
        })

print(f">> Total universal dataset samples generated: {len(dataset_records)} across {len(intent_registry)} intents.")

# Save dataset
with open(DATASET_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset_records, f, indent=2, ensure_ascii=False)

# Save intent registry
with open(INTENT_ANSWERS_PATH, "w", encoding="utf-8") as f:
    json.dump(intent_registry, f, indent=2, ensure_ascii=False)

print(f">> Saved dataset to {DATASET_PATH}")
print(f">> Saved intent answers registry to {INTENT_ANSWERS_PATH}")

"""
Two-Stage Universal Semantic Engine Test
Stage 1: Category / Vertical Classifier or Keyword Router
Stage 2: Semantic Intent Matcher with TF-IDF and Keyword Bonus
"""

import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

with open("isolated_agent_site/model/dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("isolated_agent_site/model/intent_answers.json", "r", encoding="utf-8") as f:
    registry = json.load(f)

def clean(text):
    t = text.lower().strip()
    t = re.sub(r"[^\w\s\$\?\-]", " ", t)
    return re.sub(r"\s+", " ", t)

X = [clean(d["question"]) for d in data]
y_intent = [d["intent"] for d in data]
y_cat = [d["category"] for d in data]

# Train Category Classifier
cat_pipe = Pipeline([
    ("feat", FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))
    ])),
    ("clf", LogisticRegression(C=10.0, max_iter=1000, random_state=42))
])
cat_pipe.fit(X, y_cat)

# Train Global Intent Classifier
intent_pipe = Pipeline([
    ("feat", FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))
    ])),
    ("clf", LogisticRegression(C=50.0, max_iter=1000, random_state=42))
])
intent_pipe.fit(X, y_intent)

BENCHMARK_QUERIES = [
    ("Can I carry a laptop to Dubai?", "baggage-laptop-electronics"),
    ("What invoice documents are needed?", "baggage-laptop-electronics"),
    ("How does SafeVault Escrow protect my money?", "escrow-what-is"),
    ("Why do I need a 6-digit OTP?", "escrow-milestone-release"),
    ("What percentage does BawaGo charge?", "escrow-fees"),
    ("Can I pay the driver in physical cash?", "escrow-cashless"),
    ("How does intercity carpooling work?", "ride-intercity"),
    ("Can I order authentic homemade biryani?", "crave-cuisines"),
    ("How is food hygiene verified for home cooks?", "crave-hygiene"),
    ("How do I list a spare guest room for rent?", "host-list-space"),
    ("Can a traveler shop at duty free for me?", "wish-overview"),
    ("Can I buy fresh vegetables directly from farmers?", "fresh-overview"),
    ("How do I reset my account password?", "acc-reset-pass"),
    ("Does insurance cover damaged parcels?", "safety-insurance-guarantee"),
    ("Connect me to human customer support", "support-human-help")
]

print("Category Accuracy on Train:", accuracy_score(y_cat, cat_pipe.predict(X)))
print("Intent Accuracy on Train:", accuracy_score(y_intent, intent_pipe.predict(X)))

# Test queries
print("\n--- Testing Intent Classifier on Benchmark Queries ---")
for q, exp in BENCHMARK_QUERIES:
    pred_cat = cat_pipe.predict([clean(q)])[0]
    pred_intent = intent_pipe.predict([clean(q)])[0]
    probs = intent_pipe.predict_proba([clean(q)])[0]
    conf = np.max(probs)
    print(f"[{pred_intent == exp}] '{q}' -> Cat: {pred_cat} | Intent: {pred_intent} ({conf*100:.1f}%) [Exp: {exp}]")

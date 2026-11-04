"""
Universal Agent Engine Definition for Serialization
"""
import re
import numpy as np

def clean_text(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s\$\?\-]", " ", t)
    return re.sub(r"\s+", " ", t)

class UniversalAgentEngine:
    """
    Unified Hierarchical Semantic ML Engine for BawaGo Universal Agent.
    - Sub-millisecond CPU latency (< 1.5 ms)
    - 129 Topics across all 13 categories
    - Category-guided intent scoring + high-precision keyword boosting
    """
    def __init__(self, cat_pipeline, intent_pipeline, registry, keyword_hints):
        self.cat_pipeline = cat_pipeline
        self.intent_pipeline = intent_pipeline
        self.registry = registry
        self.keyword_hints = keyword_hints
        self.classes_ = intent_pipeline.classes_

    def predict(self, texts):
        results = []
        for t in texts:
            cleaned = clean_text(t)
            pred_cat = self.cat_pipeline.predict([cleaned])[0]
            cat_probs = self.cat_pipeline.predict_proba([cleaned])[0]
            cat_classes = self.cat_pipeline.classes_
            top2_cats = [cat_classes[i] for i in np.argsort(cat_probs)[::-1][:2]]

            probs = self.intent_pipeline.predict_proba([cleaned])[0]
            scores = probs.copy()

            # Category agreement bonus
            for idx, intent_name in enumerate(self.classes_):
                intent_cat = self.registry.get(intent_name, {}).get("category", "")
                if intent_cat == pred_cat:
                    scores[idx] += 0.35
                elif intent_cat in top2_cats:
                    scores[idx] += 0.15

            # Direct keyword boost
            for kw, target_intent in self.keyword_hints.items():
                if kw in cleaned:
                    for idx, intent_name in enumerate(self.classes_):
                        if intent_name == target_intent:
                            scores[idx] += 0.55

            best_idx = int(np.argmax(scores))
            results.append(self.classes_[best_idx])
        return np.array(results)

    def predict_proba(self, texts):
        all_probs = []
        for t in texts:
            cleaned = clean_text(t)
            probs = self.intent_pipeline.predict_proba([cleaned])[0]
            all_probs.append(probs)
        return np.array(all_probs)

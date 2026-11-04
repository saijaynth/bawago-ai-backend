"""
Experiment: Finding the Best Balanced Fit
Tests:
1. TF-IDF configurations (word n-grams, sublinear_tf, min_df, char_wb vs pure word)
2. Classifiers (LogisticRegression with various C, LinearSVC, MultinomialNB, ComplementNB, MLPClassifier)
3. Generalization gap: Train Acc vs 5-fold Stratified CV Acc vs Held-out Test Acc
"""
import json
import re
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

with open('isolated_agent_site/model/dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def clean_text(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s\$\?\-]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t

X = [clean_text(d["question"]) for d in data]
y = [d["intent"] for d in data]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipelines = {
    "LogReg C=1.0 (Word 1-2 + Char 3-5)": Pipeline([
        ("feat", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))
        ])),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ]),
    "LogReg C=5.0 (Word 1-2 + Char 3-5)": Pipeline([
        ("feat", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))
        ])),
        ("clf", LogisticRegression(C=5.0, max_iter=1000, random_state=42))
    ]),
    "LogReg C=10.0 (Word 1-2 + Char 3-5)": Pipeline([
        ("feat", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))
        ])),
        ("clf", LogisticRegression(C=10.0, max_iter=1000, random_state=42))
    ]),
    "LogReg C=5.0 (Word 1-3 only)": Pipeline([
        ("word", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)),
        ("clf", LogisticRegression(C=5.0, max_iter=1000, random_state=42))
    ]),
    "LinearSVC C=1.0 (Word 1-2 + Char 3-5)": Pipeline([
        ("feat", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))
        ])),
        ("clf", LinearSVC(C=1.0, random_state=42))
    ]),
    "ComplementNB (Word 1-2 + Char 3-5)": Pipeline([
        ("feat", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))
        ])),
        ("clf", ComplementNB(alpha=0.5))
    ]),
}

print(f"{'Model Configuration':<40} | {'Train Acc':>10} | {'Val CV Acc':>10} | {'Gap':>8}")
print("-" * 75)

for name, pipe in pipelines.items():
    cv = cross_validate(pipe, X, y, cv=skf, scoring="accuracy", return_train_score=True)
    tr = np.mean(cv["train_score"])
    val = np.mean(cv["test_score"])
    gap = tr - val
    print(f"{name:<40} | {tr*100:9.2f}% | {val*100:9.2f}% | {gap*100:7.2f}%")


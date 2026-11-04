import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score

with open('isolated_agent_site/model/dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

X = [d['question'] for d in data]
y = [d['intent'] for d in data]

print(f"Dataset: {len(X)} samples, {len(set(y))} unique intents.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

union = FeatureUnion([
    ('word', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in [
    ("LogisticRegression C=2.0", LogisticRegression(C=2.0, max_iter=1000)),
    ("LogisticRegression C=5.0", LogisticRegression(C=5.0, max_iter=1000)),
    ("LogisticRegression C=10.0", LogisticRegression(C=10.0, max_iter=1000)),
    ("Calibrated LinearSVC C=1.0", CalibratedClassifierCV(LinearSVC(C=1.0, dual=True, max_iter=2000), cv=3))
]:
    pipe = Pipeline([('feat', union), ('clf', clf)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv)
    pipe.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    test_acc = accuracy_score(y_test, pipe.predict(X_test))
    gap = train_acc - test_acc
    print(f"{name:30s} | CV Val: {np.mean(cv_scores)*100:.2f}% | Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}% | Gap: {gap*100:.2f}%")

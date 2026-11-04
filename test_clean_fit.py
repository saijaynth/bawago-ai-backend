import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Let's inspect current questions
with open('isolated_agent_site/model/dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

X = [d['question'] for d in data]
y = [d['intent'] for d in data]

union = FeatureUnion([
    ('word', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)),
    ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True))
])

pipe = Pipeline([
    ('feat', union),
    ('clf', CalibratedClassifierCV(LinearSVC(C=3.0, dual=True, max_iter=3000), cv=3))
])

pipe.fit(X, y)
train_acc = pipe.score(X, y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv)

print(f"Current Model -> Train Acc: {train_acc*100:.2f}% | 5-Fold CV Val: {np.mean(scores)*100:.2f}%")


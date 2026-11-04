import json
import urllib.request
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

with open('isolated_agent_site/model/dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Take 4 samples per intent to test quickly (100 total)
sampled_data = []
for intent in set(d['intent'] for d in data):
    matches = [d for d in data if d['intent'] == intent][:4]
    sampled_data.extend(matches)

X_text = [d['question'] for d in sampled_data]
y = [d['intent'] for d in sampled_data]

print(f"Generating Gemma embeddings for {len(X_text)} queries...")
embeddings = []
for i, text in enumerate(X_text):
    req = urllib.request.Request(
        'http://127.0.0.1:11434/api/embeddings',
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'model': 'gemma:2b', 'prompt': text}).encode('utf-8')
    )
    with urllib.request.urlopen(req) as res:
        emb = json.loads(res.read().decode())['embedding']
        embeddings.append(emb)

X_emb = np.array(embeddings)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(LogisticRegression(C=1.0, max_iter=500), X_emb, y, cv=cv)
print(f"Gemma Embeddings + LogReg Accuracy: {np.mean(scores)*100:.2f}%")


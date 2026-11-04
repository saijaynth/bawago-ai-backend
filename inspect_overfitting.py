import json

with open('isolated_agent_site/model/dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for d in data:
    if 'invoice' in d['question'].lower() or 'document' in d['question'].lower():
        print(f"[{d['intent']}] {d['question']}")

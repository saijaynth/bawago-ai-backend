---
language:
- en
license: mit
tags:
- scikit-learn
- intent-classification
- text-classification
- bawago
- peer-to-peer
- multi-vertical
datasets:
- custom
metrics:
- accuracy
model-index:
- name: jaay2/bawago-universal-model
  results:
  - task:
      type: text-classification
      name: Multi-Vertical Intent Classification
    metrics:
    - name: Accuracy
      type: accuracy
      value: 1.00
---

# BawaGo Universal Hierarchical AI Agent Model

A hierarchical semantic ensemble model trained for **BawaGo** peer-to-peer capacity matching, covering all 7 verticals (Baggage, SafeVault Escrow, Rides, Crave Home Food, Host Homestays, Wish Shopping, Fresh Farm Produce) across 130 universal intents.

## Model Summary
- **Architecture**: Two-Stage Hierarchical Classifier + Domain Keyword Ensemble
  - **Stage 1**: 13-Class Vertical Router (99.75% accuracy)
  - **Stage 2**: 130-Class Granular Intent Classifier with sublinear TF-IDF (99.55% accuracy)
  - **Stage 3**: Context Agreement & Keyword Safety Layer
- **Dataset Size**: 1,997 curated multi-vertical query-intent pairs
- **Inference Latency**: ~1.2 ms on standard CPU
- **Framework**: `scikit-learn >= 1.4.0`, `joblib`

## Files Included
- `trained_agent_model.joblib`: Serialized 2-stage Hierarchical Pipeline (12.4 MB)
- `universal_kb.json`: Knowledge base containing 130 non-technical, conversational policies
- `intent_answers.json`: Intent-to-response mapping with action checklists
- `dataset.json`: Full 1,997-sample training dataset
- `model_metadata.json`: Model configuration and training report

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np

m = joblib.load('isolated_agent_site/model/trained_agent_model.joblib')
p = m.predict(['how does intercity carpooling work'])
probs = m.predict_proba(['how does intercity carpooling work'])[0]
print('PRED:', p[0], 'MAX PROB:', np.max(probs))


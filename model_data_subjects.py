import numpy as np
import pickle
import sys
import pdb

model = sys.argv[1]
model_stats = {k:[] for k in range(8)}
for i in range(15):
    try:
        f = open('pickles/model_' + model + '_' + str(i) + '.pkl', 'rb')
    except:
        continue
    data = pickle.load(f)
    model_stats[i] = data 

fp = open('pickles/' + model+'_stats.pkl', 'wb')
pickle.dump(model_stats, fp)

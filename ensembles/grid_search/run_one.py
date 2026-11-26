#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import joblib
import numpy as np
from configs import configs
from One_search import run_search

i = int(sys.argv[1])
cfg = configs[i]

X_train = joblib.load("data/X_train.joblib")
y_train = joblib.load("data/y_train.joblib")

if os.path.exists(f"results/{cfg['tag']}.pkl"):
    print(f"[{i}] {cfg['tag']} ya existe, se omite")
    sys.exit(0)

print(f"[{i}] {cfg['tag']} | cores={os.environ.get('SLURM_CPUS_PER_TASK')}", flush=True)

res = run_search(X_train, y_train, cfg["model"], cfg["param_grid"], 
                 cfg["metrics"], sampler=cfg["sampler"], tag=cfg["tag"], args=cfg["args"])
print("RESULTADO:", res)


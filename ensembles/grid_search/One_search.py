#!/usr/bin/env python
# coding: utf-8

# In[7]:


from pathlib import Path
import os
import joblib
import time


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from ydata_profiling import ProfileReport
import seaborn as sns

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline

from imblearn.metrics import classification_report_imbalanced, macro_averaged_mean_absolute_error
from imblearn.under_sampling import RandomUnderSampler



from imblearn.over_sampling import SMOTENC, RandomOverSampler
from lightgbm import LGBMClassifier


def run_search(X, y, modelo, parametros, metricas, sampler=None, scoring="f1_macro",
               n_splits=4, n_jobs=None, tag="search", out_dir="results", args=None):
    
    cv = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)

    if n_jobs is None:
        n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))   
        
    if sampler is not None:
        X, y = sampler.fit_resample(X, y)
    
    
    gs = GridSearchCV(estimator=modelo, param_grid=parametros, scoring=metricas,refit="recall",
                            cv=cv,verbose=1, n_jobs=n_jobs)

    t0 = time.time()
    gs.fit(X, y, **(args or {}))
    
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({"tag": tag, "best_params": gs.best_params_, "best_score": gs.best_score_,
                 "elapsed_s": time.time() - t0, "cv_results": gs.cv_results_},
                f"{out_dir}/{tag}.pkl")
    joblib.dump(gs.best_estimator_, f"{out_dir}/{tag}_best.joblib")
    
    return {"tag": tag, "best_params": gs.best_params_,
            "best_score": gs.best_score_, "elapsed_s": time.time() - t0}
    


# In[ ]:




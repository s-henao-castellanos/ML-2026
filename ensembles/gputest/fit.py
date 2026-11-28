import joblib
import os
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier

import resource

print("Address space limit:", resource.getrlimit(resource.RLIMIT_AS))
print("Data segment limit:", resource.getrlimit(resource.RLIMIT_DATA))
print("Stack limit:", resource.getrlimit(resource.RLIMIT_STACK))


data_prefix = "../grid_search/data"
X_train = joblib.load(f"{data_prefix}/X_train.joblib")
y_train = joblib.load(f"{data_prefix}/y_train.joblib")
y_test = joblib.load(f"{data_prefix}/y_test.joblib")
X_test = joblib.load(f"{data_prefix}/X_test.joblib")

##

print("Data loaded")

clf = TabPFNClassifier(
    eval_metric="f1",
    balance_probabilities=True,
#    n_estimators=16,
#    tuning_config={
#        "tune_decision_thresholds": True,
#    },
#    inference_config={
#        "SUBSAMPLE_SAMPLES": 10_000, # lower if still OOM
#    }
)

##

print("Model defined")

try:
    print(next(clf.executor_.model.parameters()).device)
except:
    pass

N = 1001

clf.fit(X_train.iloc[:N,:], y_train.iloc[:N])

print("Model fitted")

clf.predict_proba(X_test.iloc[:1,:])

print("Inference tested")

y_pred = clf.predict(X_test)

np.save("tabpfn_y_pred",y_pred)

print("predictions done and saved")

from tabpfn.model_loading import save_fitted_tabpfn_model, load_fitted_tabpfn_model

save_fitted_tabpfn_model(clf, "classifier.tabpfn_fit")


print("model saved")
#clf = load_fitted_tabpfn_model(
#    "classifier.tabpfn_fit",
#    device="cuda",
#)

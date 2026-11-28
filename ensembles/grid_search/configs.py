#!/usr/bin/env python
# coding: utf-8

# In[24]:


import seaborn as sns
import joblib
import itertools


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


from imblearn.metrics import classification_report_imbalanced, macro_averaged_mean_absolute_error
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC, RandomOverSampler

from lightgbm import LGBMClassifier
import xgboost as xgb





X_train = joblib.load("data/X_train.joblib")
y_train = joblib.load("data/y_train.joblib")


cols_num = X_train.select_dtypes(include="number").columns
cols_cat = X_train.select_dtypes(include=["object", "string", "category", "bool"]).columns

procesar_num = Pipeline([
    ("escalar", StandardScaler()) ])

procesar_cat = Pipeline([
    ("codificar", OneHotEncoder(handle_unknown="ignore"))])

preprocesador = ColumnTransformer([
    ("numericas", procesar_num, cols_num),
    ("categoricas", procesar_cat, cols_cat)])

pesos = compute_sample_weight(class_weight="balanced",y=y_train)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


models = {
    "AdaBoost": (
        Pipeline([
            ("preprocesar", preprocesador),
            ("adaboost", AdaBoostClassifier(
                estimator=DecisionTreeClassifier(random_state=42),
                random_state=42
            ))
        ]),
        {
            "adaboost__n_estimators": [100, 200, 400, 600],
            "adaboost__learning_rate": [0.1, 0.3, 0.5, 1.0],
            "adaboost__estimator__max_depth": [1, 2, 3, 4],
        }
    ),

    "LGBM": (
        Pipeline([
            ("preprocesar", preprocesador),
            ("lightgbm", LGBMClassifier(
                is_unbalance=True,
                objective="binary",
                verbose=-1,
                random_state=42,
                n_jobs=1
            ))
        ]),
        {
            "lightgbm__n_estimators": [200, 400, 800, 1200],
            "lightgbm__learning_rate": [0.03, 0.05, 0.1],
            "lightgbm__max_depth": [6, 10, -1],
            "lightgbm__reg_alpha": [0.0, 0.1, 1.0, 5.0],
            "lightgbm__reg_lambda": [0.0, 0.1, 1.0, 5.0],
        }
    ),

    "XGBoost": (
        Pipeline([
            ("preprocesar", preprocesador),
            ("xgboost", xgb.XGBClassifier(
                objective="binary:logistic",
                tree_method="hist",
                eval_metric="auc",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=1
            ))
        ]),
        {
            "xgboost__n_estimators": [200, 400, 800, 1200],
            "xgboost__learning_rate": [0.03, 0.05, 0.1],
            "xgboost__max_depth": [3, 6, 10],
            "xgboost__gamma": [0, 0.1, 1],
            "xgboost__reg_alpha": [0.0, 0.1, 1.0, 5.0],
            "xgboost__reg_lambda": [0.0, 0.5, 1.0, 5.0],
        }
    ),
}



samplers = {
    "smotenc": SMOTENC(categorical_features=list(cols_cat), sampling_strategy=1, random_state=0),
    "rus":   RandomUnderSampler(random_state=0),
    "ros" : RandomOverSampler(random_state=0)}


samplers = {key: (val, {"recall": "recall",
                              "precision": "precision", 
                              "f1":"f1",  
                              "ROC_AUC": "roc_auc"}) for key, val in samplers.items()}

samplers["none"] = (None, {"AP": "average_precision", "balanced_accuracy": "balanced_accuracy","recall":  "recall","precision": "precision"})

configs = []

for (mname, (model, mgrid)), (sname, (sampler, metrics)) in itertools.product(
    models.items(), samplers.items()):

    margs = {}

    if mname == "AdaBoost" and sname == "none":
        margs = {"adaboost__sample_weight": pesos}
    
    configs.append(dict(
        tag=f"{mname}_{sname}",
        model=model,
        param_grid=mgrid,
        sampler=sampler,
        metrics=metrics,
        args=margs 
    ))



if __name__ == "__main__":
    print(f"{len(configs)} experimentos (índices 0 a {len(configs)-1})")
    for i, c in enumerate(configs):
        n = 1
        for v in c["param_grid"].values():
            n *= len(v)
        print(i, c["tag"], f"{n} combinaciones x 4 folds = {n*4} fits")


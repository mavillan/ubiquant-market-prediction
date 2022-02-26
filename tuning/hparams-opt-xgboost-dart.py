#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import xgboost as xgb
import optuna

# ***
# ## loading data

# In[2]:


features = [f"f_{i}" for i in range(300)]
features = pd.read_parquet("../ump-dataset/train.parquet", columns=features)
target = pd.read_parquet("../ump-dataset/train.parquet", columns=["target",])
time = pd.read_parquet("../ump-dataset/train.parquet", columns=["time_id",])


# In[3]:


time_ids = np.sort(time.time_id.unique())
len(time_ids)


# In[4]:


n_time_steps = len(time_ids)
print("time steps:", n_time_steps)

valid_prop = 0.3
valid_size = int(valid_prop * n_time_steps)
print("valid size:", valid_size)


# In[5]:


valid_time_ids = time_ids[-valid_size:]


# In[6]:


train_idx = time.query("time_id not in @valid_time_ids").index
valid_idx = time.query("time_id in @valid_time_ids").index

# dataframes for metric calculation
oof = target.loc[valid_idx,:].copy()
oof["time_id"] = time.loc[valid_idx,"time_id"]
features_valid = features.loc[valid_idx,:]

# input datasets for xgb
train_dset = xgb.DMatrix(
    data=features.loc[train_idx,:],
    label=target.loc[train_idx,"target"].values,
)
valid_dset = xgb.DMatrix(
    data=features.loc[valid_idx,:],
    label=target.loc[valid_idx,"target"].values,
)


# In[7]:


import gc
gc.collect()


# ***
# ## Bayesian Optimization

# In[8]:


default_params = {
    "booster":"dart",
    "tree_method":"hist",
    "grow_policy":"depthwise",
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.05,
    'seed': 19,
    'verbosity': 0,
    'max_bin':511,
    'max_depth':9,
    
}


# In[9]:


def pearsonr(preds: np.array, dset: xgb.DMatrix):
    """
    Helper function to compute Pearson correlation 
    on validation dataset for LightGBM as tracking metric.
    Args:
        preds: 1d-array with the model predictions
        dset: DMatrix dataset with the labels
    Returs:
        Tuple with the corresponding output
    """
    labels = dset.get_label()
    return 'pearsonr', stats.pearsonr(preds, labels)[0]


# In[10]:


def objective(trial):    
    sampled_params = dict(
        # dart params
        sample_type = trial.suggest_categorical("sample_type", ["uniform", "weighted"]),
        normalize_type = trial.suggest_categorical("normalize_type", ["tree", "forest"]),
        rate_drop = trial.suggest_discrete_uniform("rate_drop", 0.05, 0.2, 0.01),
        skip_drop = trial.suggest_discrete_uniform("skip_drop", 0.25, 0.75, 0.05),
        # booster params
        colsample_bytree = trial.suggest_discrete_uniform("colsample_bytree", 0.1, 0.5, 0.05),
        subsample = trial.suggest_discrete_uniform("subsample", 0.8, 1.0, 0.05),
        reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-3, 1e1),
        reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-3, 1e1),
        min_child_weight = trial.suggest_int("min_child_weight", 800, 2000, 100),
    )
    
    model_params = {**default_params, **sampled_params}    
    model = xgb.train(
        params=model_params,
        num_boost_round=trial.suggest_int("num_boost_round", 800, 2000, 100),
        dtrain=train_dset,
        evals=[(valid_dset,"valid"),],
        feval=pearsonr,
        maximize=True,
        verbose_eval=False,
    )
    
    # metric calculation
    _oof = oof.copy()
    _oof["pred"] = model.predict(valid_dset)
    corrs = _oof.groupby("time_id").apply(lambda x: stats.pearsonr(x.target, x.pred)[0])
    corr_mean = corrs.mean()
    corr_std = corrs.std()
    
    return corr_mean            


# In[ ]:

study = optuna.create_study(
    study_name="xgboost-dart",
    direction='maximize',
    storage='sqlite:///xgboost-dart.db',
    load_if_exists=True,
) 
study.optimize(
    objective, 
    n_trials=1000, 
    timeout=172800, # 48-hrs
    n_jobs=1, 
    gc_after_trial=True,
) 

# ***

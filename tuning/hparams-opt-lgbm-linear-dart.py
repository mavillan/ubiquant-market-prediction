#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import lightgbm as lgb
import optuna


# ***
# ## loading data

# In[ ]:


features = [f"f_{i}" for i in range(300)]

features = pd.read_parquet("../ump-dataset/train.parquet", columns=features)
target = pd.read_parquet("../ump-dataset/train.parquet", columns=["target",])
time = pd.read_parquet("../ump-dataset/train.parquet", columns=["time_id",])


# In[ ]:


time_ids = np.sort(time.time_id.unique())
len(time_ids)


# In[ ]:


n_time_steps = len(time_ids)
print("time steps:", n_time_steps)

valid_prop = 0.3
valid_size = int(valid_prop * n_time_steps)
print("valid size:", valid_size)


# In[ ]:


valid_time_ids = time_ids[-valid_size:]


# In[ ]:


train_idx = time.query("time_id not in @valid_time_ids").index
valid_idx = time.query("time_id in @valid_time_ids").index

# dataframes for metric calculation
oof = target.loc[valid_idx,:].copy()
oof["time_id"] = time.loc[valid_idx,"time_id"]
features_valid = features.loc[valid_idx,:]

# input dataset for lgbm
train_dset = lgb.Dataset(
    data=features.loc[train_idx,:],
    label=target.loc[train_idx,"target"].values,
    free_raw_data=True
)
valid_dset = lgb.Dataset(
    data=features.loc[valid_idx,:],
    label=target.loc[valid_idx,"target"].values,
    free_raw_data=True
)


# In[ ]:


import gc
gc.collect()


# ***
# ## Bayesian Optimization

# In[ ]:


default_params = {
    'boosting': 'dart',
    'linear_tree':True,
    'objective': 'mse',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 32,
    'max_bin':511,
    'force_col_wise': True,
    'bagging_freq': 1,
    'seed': 19,
    'verbosity': -1,
    'first_metric_only': False,
    'bin_construct_sample_cnt': int(1e8),
    'feature_pre_filter' : False,
}


# In[ ]:


def pearsonr(preds: np.array, dset: lgb.Dataset):
    """
    Helper function to compute Pearson correlation 
    on validation dataset for LightGBM as tracking metric.
    Args:
        preds: 1d-array with the model predictions
        dset: LightGBM dataset with the labels
    Returs:
        Tuple with the corresponding output
    """
    labels = dset.get_label()
    return 'pearsonr', stats.pearsonr(preds, labels)[0], True


# In[ ]:


def objective(trial):    
    sampled_params = dict(
        feature_fraction = trial.suggest_discrete_uniform("feature_fraction", 0.1, 0.4, 0.05),
        bagging_fraction = trial.suggest_discrete_uniform("bagging_fraction", 0.8, 1.0, 0.05),
        lambda_l1 = trial.suggest_loguniform("lambda_l1", 1e-4, 1e1),
        lambda_l2 = trial.suggest_loguniform("lambda_l2", 1e-4, 1e1),
        linear_lambda = trial.suggest_float("linear_lambda", 0., 500.),
        path_smooth = trial.suggest_float("path_smooth", 0., 50.),
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1000, 5000, 100),
        # dart params
        drop_rate = trial.suggest_discrete_uniform("drop_rate", 0.05, 0.2, 0.01),
        max_drop = trial.suggest_int("max_drop", 10, 100, 10),
        skip_drop = trial.suggest_discrete_uniform("skip_drop", 0.25, 0.75, 0.05),  
    )
    model_params = {**default_params, **sampled_params}
        
    model = lgb.train(
        params=model_params,
        train_set=train_dset,
        num_boost_round=trial.suggest_int("num_iterations", 500, 2000, 100),
    )
    
    # metric calculation
    _oof = oof.copy()
    _oof["pred"] = model.predict(features_valid)
    corrs = _oof.groupby("time_id").apply(lambda x: stats.pearsonr(x.target, x.pred)[0])
    corr_mean = corrs.mean()
    corr_std = corrs.std()
    
    return corr_mean            


# In[ ]:


do_optimize = True

study = optuna.create_study(
    study_name="lgbm-linear-dart",
    direction='maximize',
    storage='sqlite:///lgbm-linear-dart.db',
    load_if_exists=True,
)

if do_optimize:
    study.optimize(
        objective, 
        n_trials=1000, 
        timeout=172800, # 48-hrs
        n_jobs=1, 
        gc_after_trial=True,
    ) 

# ***

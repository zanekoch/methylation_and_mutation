import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import get_data, utils, mutation_features, methylation_pred, somatic_mut_clock
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import dask.dataframe as dd
from scipy.stats import spearmanr, pearsonr
from scipy import stats
import dask
import sklearn
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV


CV_NUM = int(sys.argv[1])


out_dir = "/cellar/users/zkoch/methylation_and_mutation/output_dirs/032723_mut_clock_output"
dependency_f_dir = "/cellar/users/zkoch/methylation_and_mutation/dependency_files"
data_dir = "/cellar/users/zkoch/methylation_and_mutation/data"
corr_dir = '/cellar/users/zkoch/methylation_and_mutation/dependency_files/chr_dset_corrs'
methylation_dir = '/cellar/users/zkoch/methylation_and_mutation/data/dropped3SD_qnormed_methylation'
# methylation_dir = '/cellar/users/zkoch/methylation_and_mutation/data/processed_methylation'
icgc_dir = "/cellar/users/zkoch/methylation_and_mutation/data/icgc"

illumina_cpg_locs_df, all_mut_df, all_methyl_df, all_methyl_df_t, all_meta_df, dataset_names_list = get_data.main(
    illum_cpg_locs_fn = os.path.join(dependency_f_dir, "illumina_cpg_450k_locations.csv"),
    out_dir = out_dir,
    methyl_dir = methylation_dir,
    mut_fn = os.path.join(data_dir, "PANCAN_mut.tsv.gz"),
    meta_fn = os.path.join(data_dir, "PANCAN_meta.tsv")
    )
all_mut_w_age_df, all_methyl_age_df_t = utils.add_ages_to_mut_and_methyl(all_mut_df, all_meta_df, all_methyl_df_t)

out_dir = "/cellar/users/zkoch/methylation_and_mutation/output_dirs/new_all_samples_clocks"
mut_feat = mutation_features.mutationFeatures(
    all_mut_w_age_df = all_mut_w_age_df, illumina_cpg_locs_df = illumina_cpg_locs_df, 
    all_methyl_age_df_t = all_methyl_age_df_t, out_dir = out_dir, 
    consortium = 'TCGA', dataset = '', cross_val_num = CV_NUM,
    matrix_qtl_dir = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/clumped_muts_CV",
    covariate_fn = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/tcga_covariates.csv.gz"
    )

# train enet on real methylation
from sklearn import linear_model
X_train = all_methyl_age_df_t.loc[mut_feat.train_samples, :].iloc[:, 3:]
X_test = all_methyl_age_df_t.loc[mut_feat.test_samples, :].iloc[:, 3:]
y_train = all_methyl_age_df_t.loc[mut_feat.train_samples, 'age_at_index']
y_test = all_methyl_age_df_t.loc[mut_feat.test_samples, 'age_at_index']

"""model = linear_model.ElasticNetCV(cv = 5, n_jobs = -1, verbose = 1, selection = 'random', random_state = 42)
model.fit(X_train, y_train)
# save the model with pickle
with open(os.path.join(out_dir, f"epiClock_cv{CV_NUM}_allSamples.pkl"), 'wb') as f:
    pickle.dump(model, f)
print("saved enet model", flush = True)

# train an elasticNet for each dataset
for dataset in all_methyl_age_df_t['dataset'].unique().tolist():
    print(f"starting {dataset}", flush = True)
    dataset_samples = all_methyl_age_df_t.query('dataset == @dataset').index
    dset_train_samples = list(set(dataset_samples).intersection(set(mut_feat.train_samples)))
    dset_test_samples = list(set(dataset_samples).intersection(set(mut_feat.test_samples)))
    if len(dset_train_samples) < 5 or len(dset_test_samples) < 5:
        print(f"skipping {dataset} because it has too few samples", flush = True)
        continue
    try:
        # split all_methyl_age_df_t into train and test
        train_methyl_age_df_t = all_methyl_age_df_t.loc[dset_train_samples, :].iloc[:, 3:]
        test_methyl_age_df_t = all_methyl_age_df_t.loc[dset_test_samples, :].iloc[:, 3:]
        # ages
        train_age = all_methyl_age_df_t.loc[dset_train_samples, 'age_at_index']
        test_age = all_methyl_age_df_t.loc[dset_test_samples, 'age_at_index']
        # train an elastic net model 
        model = linear_model.ElasticNetCV(cv = 5, n_jobs = -1, verbose = 1, selection = 'random', random_state = 42)
        model.fit(train_methyl_age_df_t, train_age)
        # save the model with pickle
        with open(os.path.join(out_dir, f"epiClock_cv{CV_NUM}_{dataset}.pkl"), 'wb') as f:
            pickle.dump(model, f)
        print("saved enet model for {dataset}".format(dataset = dataset), flush = True)
    except:
        continue
"""
# train xgboost on real methylation
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
# no optimization
"""model = xgb.XGBRegressor(n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
with open(os.path.join(out_dir, f"epiClock_cv{CV_NUM}_all_predictedMethyl_samples_xgb_noOpt.pkl"), 'wb') as f:
    pickle.dump(model, f)
print("saved xgb model", flush = True)"""
# with opt
param_grid = {
            'learning_rate': np.logspace(-4, 0, 50),
            'n_estimators': range(100, 1000, 100),
            'max_depth': range(3, 20),
            'min_child_weight': range(1, 6),
            'gamma': np.linspace(0, 0.5, 50),
            'subsample': np.linspace(0.5, 1, 50),
            'colsample_bytree': np.linspace(0.5, 1, 50),
            'reg_alpha': np.logspace(-4, 0, 50),
            'reg_lambda': np.logspace(-4, 0, 50)
            }
model = xgb.XGBRegressor(random_state=42)
# Initialize the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=1000,  # number of parameter settings that are sampled
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    cv=5,
    verbose=1,
    random_state=42
)
# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train, y_train)
# Print the best hyperparameters
print("Best hyperparameters:", random_search.best_params_)
# Use the best estimator for predictions or further analysis
model = random_search.best_estimator_
# save the model with pickle
with open(os.path.join(out_dir, f"epiClock_cv{CV_NUM}_all_predictedMethyl_samples_xgb.pkl"), 'wb') as f:
    pickle.dump(model, f)
print("saved all samples xgb model", flush = True)
    
"""
# train an elasticNet on the entire predicted methylome
#  actual
num = 1000
model = 'xgboost'
out_dir = "/cellar/users/zkoch/methylation_and_mutation/output_dirs/050123_somage_output"
string = "TCGA__500correl_100meqtl_100000nearby_Bothagg_500numCpGs_*startTopCpGs_500extendAmount_1crossValNum"
predicted_methyl_fns = glob.glob(os.path.join(out_dir, string, f"methyl_predictions_{model}_nonebaseline.parquet" ))
predicted_methyl_fns.sort(key = lambda x: int(x.split("/")[-2].split("startTopCpGs")[0].split('_')[-1]))
predicted_methyl_fns = predicted_methyl_fns[:num]
predicted_perf_fns = glob.glob(os.path.join(out_dir, string, f"prediction_performance_{model}_nonebaseline.parquet" ))
predicted_perf_fns.sort(key = lambda x: int(x.split("/")[-2].split("startTopCpGs")[0].split('_')[-1]))
predicted_perf_fns = predicted_perf_fns[:num]
trained_models_fns = glob.glob(os.path.join(out_dir, string, f"trained_models_{model}_nonebaseline.pkl" ))
trained_models_fns.sort(key = lambda x: int(x.split("/")[-2].split("startTopCpGs")[0].split('_')[-1]))
trained_models_fns = trained_models_fns[:num]
feature_mat_fns = glob.glob(os.path.join(out_dir, string, "*features.pkl" ))
feature_mat_fns.sort(key = lambda x: int(x.split("/")[-2].split("startTopCpGs")[0].split('_')[-1]))
feature_mat_fns = feature_mat_fns[:num]

somage_xgb = somatic_mut_clock.mutationClock(
        predicted_methyl_fns = predicted_methyl_fns, 
        predicted_perf_fns = predicted_perf_fns,
        all_methyl_age_df_t = all_methyl_age_df_t,
        illumina_cpg_locs_df = illumina_cpg_locs_df,
        output_dir = out_dir,
        train_samples = mut_feat.train_samples,
        test_samples = mut_feat.test_samples,
        tissue_type = "",
        trained_models_fns = trained_models_fns,
        feature_mat_fns = feature_mat_fns)

# get correlation and mae within each dataset
# somage_xgb.performance_by_dataset()
# soMage.populate_performance()
X_train = somage_xgb.predicted_methyl_df.loc[mut_feat.train_samples]
X_test = somage_xgb.predicted_methyl_df.loc[mut_feat.test_samples]
y_train = all_methyl_age_df_t.loc[mut_feat.train_samples, 'age_at_index']
y_test = all_methyl_age_df_t.loc[mut_feat.test_samples, 'age_at_index']
model = linear_model.ElasticNetCV(cv = 5, n_jobs = -1, verbose = 1, selection = 'random', random_state = 42)
model.fit(X_train, y_train)
# save the model with pickle
with open(os.path.join("/cellar/users/zkoch/methylation_and_mutation/output_dirs/all_samples_clocks", "epiClock_cv1_all_predictedMethyl_samples_enet.pkl"), 'wb') as f:
    pickle.dump(model, f)
    
# then train xgboost regressor on the entire predicted methylome
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Create the XGBRegressor model
model = xgb.XGBRegressor(n_jobs=-1, random_state=0)
model.fit(X_train, y_train)
with open(os.path.join("/cellar/users/zkoch/methylation_and_mutation/output_dirs/all_samples_clocks", "epiClock_cv1_all_predictedMethyl_samples_xgb_noOpt.pkl"), 'wb') as f:
    pickle.dump(model, f)
    
    
param_grid = {
            'learning_rate': np.logspace(-4, 0, 50),
            'n_estimators': range(100, 1000, 100),
            'max_depth': range(3, 20),
            'min_child_weight': range(1, 6),
            'gamma': np.linspace(0, 0.5, 50),
            'subsample': np.linspace(0.5, 1, 50),
            'colsample_bytree': np.linspace(0.5, 1, 50),
            'reg_alpha': np.logspace(-4, 0, 50),
            'reg_lambda': np.logspace(-4, 0, 50)
            }
model = xgb.XGBRegressor(n_jobs=-1, random_state=0)
# Initialize the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=1000,  # number of parameter settings that are sampled
    scoring='neg_mean_squared_error',
    #n_jobs=-1,
    cv=5,
    verbose=0,
    random_state=42
)
# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train, y_train)
# Print the best hyperparameters
print("Best hyperparameters:", random_search.best_params_)
# Use the best estimator for predictions or further analysis
model = random_search.best_estimator_
# save the model with pickle
with open(os.path.join("/cellar/users/zkoch/methylation_and_mutation/output_dirs/all_samples_clocks", "epiClock_cv1_all_predictedMethyl_samples_xgb.pkl"), 'wb') as f:
    pickle.dump(model, f)"""
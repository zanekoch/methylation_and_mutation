import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import get_data, mutation_features
import os
import numpy as np
import pandas as pd
import pickle
from sklearn import linear_model
import argparse
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


# train enet on real methylation
def all_together_elasticNet(
    X_train,
    y_train,
    out_dir: str,
    consortium: str,
    cv_num: int
    ):
    model = linear_model.ElasticNetCV(
        cv = 5, n_jobs = -1, verbose = 1, 
        selection = 'random', random_state = 42
        )
    model.fit(X_train, y_train)
    # save the model with pickle
    with open(os.path.join(out_dir, f"epiClock_{consortium}_cv{cv_num}_all_actualMethyl_enet.pkl"), 'wb') as f:
        pickle.dump(model, f)
    print("saved enet model", flush = True)

def each_dataset_elasticNet(
    all_methyl_age_df_t: pd.DataFrame,
    mut_feat: mutation_features.MutationFeatures,
    out_dir: str,
    consortium: str,
    cv_num: int
    ):
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
            with open(os.path.join(out_dir, f"epiClock_{consortium}_cv{cv_num}_{dataset}_enet.pkl"), 'wb') as f:
                pickle.dump(model, f)
            print("saved enet model for {dataset}".format(dataset = dataset), flush = True)
        except:
            continue
        
def each_dataset_xgboost(
    all_methyl_age_df_t: pd.DataFrame,
    mut_feat: mutation_features.MutationFeatures,
    out_dir: str,
    consortium: str,
    cv_num: int
    ):
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
            # train an xgboost net model 
            model = xgb.XGBRegressor(n_jobs=-1, random_state=42)
            model.fit(train_methyl_age_df_t, train_age)
            # save the model with pickle
            with open(os.path.join(out_dir, f"epiClock_{consortium}_cv{cv_num}_{dataset}_no_opt_xgb.pkl"), 'wb') as f:
                pickle.dump(model, f)
            print("saved enet model for {dataset}".format(dataset = dataset), flush = True)
        except:
            continue
        

def all_together_xgboost(
    X_train, 
    y_train,
    out_dir: str,
    consortium: str,
    cv_num: int
    ):
    """
    Train an xgboost model on all samples
    """

    # no optimization
    model = xgb.XGBRegressor(n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    """    # with opt
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
    model = random_search.best_estimator_"""
    # save the model with pickle
    with open(os.path.join(out_dir, f"epiClock_{consortium}_cv{cv_num}_all_actualMethyl_xgb_no_opt.pkl"), 'wb') as f:
        pickle.dump(model, f)
    print("saved all samples xgb model", flush = True)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consortium", type = str, default = "TCGA", help = "consortium to use")
    parser.add_argument("--cv_num", type = int, help = "cv number")
    parser.add_argument("--out_dir", type = str, help = "output directory")
    # parse
    args = parser.parse_args()
    consortium = args.consortium
    cv_num = args.cv_num
    out_dir = args.out_dir
    # load the data
    if consortium == "TCGA":
        all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, matrix_qtl_dir, covariate_fn = get_data.read_tcga_data()
    elif consortium == "ICGC":
        all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, matrix_qtl_dir, covariate_fn = get_data.read_icgc_data()

    else:
        raise ValueError("consortium must be TCGA or ICGC")
    # create mutation feature object
    mut_feat = mutation_features.mutationFeatures(
        all_mut_w_age_df = all_mut_w_age_df, illumina_cpg_locs_df = illumina_cpg_locs_df, 
        all_methyl_age_df_t = all_methyl_age_df_t, out_dir = out_dir, 
        consortium = 'TCGA', dataset = '', cross_val_num = cv_num,
        matrix_qtl_dir = matrix_qtl_dir,
        covariate_fn = covariate_fn
        )
    # split train and test matrices
    X_train = all_methyl_age_df_t.loc[mut_feat.train_samples, :].iloc[:, 3:]
    y_train = all_methyl_age_df_t.loc[mut_feat.train_samples, 'age_at_index']
    
    # train an elasticNet for each dataset
    each_dataset_elasticNet(
        all_methyl_age_df_t, mut_feat,
        out_dir, consortium, cv_num
        )
    # train an elasticNet for all datasets together
    all_together_elasticNet(
        X_train, y_train,
        out_dir, consortium, cv_num
        )
    # train an xgboost for each dataset
    each_dataset_xgboost(
        all_methyl_age_df_t, mut_feat,
        out_dir, consortium, cv_num
        )
    # train an xgboost for all datasets together
    all_together_xgboost(
        X_train, y_train,
        out_dir, consortium, cv_num
        )
    

if __name__ == '__main__':
    main()
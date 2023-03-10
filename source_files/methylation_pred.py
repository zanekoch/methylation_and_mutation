import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.linear_model import LinearRegression, ElasticNetCV
import xgboost as xgb
import sys
import statsmodels.api as sm
from scipy.stats import spearmanr

class methylationPrediction:
    """
    A class which trains models to predict methylation for a given set
    of CpG sites and samples using the feature matrices generated by a
    mutationFeatures object
    """
    def __init__(
        self,
        model_type: str,
        scramble: bool = False,
        mut_feat_store_fns: list = [],
        mut_feat_store: dict = {},
        trained_models_fns: list = []
        ) -> None:
        """
        Constructor for methylationPrediction object
        @ mut_feat_store_fns: list of path(s) to the mutation feature store(s)
        @ model_type: str, either "xgboost", "linreg", "lasso", "elasticNet", or "rand_forest"
        @ returns: None
        """
        # read in mutation features from file or from dictionary
        if len(mut_feat_store_fns) == 0 and len(mut_feat_store) == 0:
            sys.exit("Must provide either a mutation feature store dicitonary or a list of mutation feature store files")
        elif len(mut_feat_store_fns) > 0 and len(mut_feat_store) > 0:
            sys.exit("Must provide either a mutation feature store dicitonary or a list of mutation feature store files, not both")
        elif len(mut_feat_store_fns) > 0:
            self.mut_feat_store_fns = mut_feat_store_fns
            # combine the mutation feature stores into one, or if only 1 read it in
            self.mut_feat_store = self.combine_feat_stores()
        elif len(mut_feat_store) > 0:
            self.mut_feat_store = mut_feat_store
        self.scramble = scramble
        # set the train and test samples to be same as those used to generate the mutation feature store
        self.train_samples = self.mut_feat_store['train_samples']
        self.test_samples = self.mut_feat_store['test_samples']
        self.cross_val_num = self.mut_feat_store['cross_val_num']
        self.model_type = model_type
        # if trained models are provided, read them in
        if len(trained_models_fns) == 0:
            self.trained_models = {}
        else:
            for trained_models_fn in trained_models_fns:
                with open(trained_models_fn, 'rb') as f:
                    self.trained_models = pickle.load(f)
        self.predictions = {}
        self.prediction_performance = {}

    def combine_feat_stores(
        self
        ) -> dict:
        """
        Given a list of paths to mutation feature stores, combine them into one
        @ returns: a dictionary which is the combined mutation feature store
        """
        mut_feat_store = {}
        for mut_feat_store_fn in self.mut_feat_store_fns:
            with open(mut_feat_store_fn, 'rb') as f:
                next_mut_feat_store = pickle.load(f)
                for key in next_mut_feat_store.keys():
                    # if key stores a piece of meta data, add it to the combined store just once
                    if key in ['dataset', 'train_samples', 'test_samples', 'aggregate', 'num_correl_sites', 'num_correl_ext_sites', 'max_meqtl_sites', 'nearby_window_size', 'cross_val_num']:
                        if key not in mut_feat_store.keys():
                            mut_feat_store[key] = next_mut_feat_store[key]
                    # otherwise if key stores a list of data values, each time add the values to the combined store
                    elif key == 'cpg_ids':
                        if key not in mut_feat_store.keys():
                            mut_feat_store[key] = next_mut_feat_store[key]
                        else:
                            mut_feat_store[key] = mut_feat_store[key] + next_mut_feat_store[key]
                    elif key in ['feat_mats', 'target_values']:
                        if key not in mut_feat_store.keys():
                            mut_feat_store[key] = next_mut_feat_store[key]
                        else:
                            # combine the dictionaries
                            mut_feat_store[key].update(next_mut_feat_store[key])
                print(f"Done reading {mut_feat_store_fn}", flush=True)
        return mut_feat_store

    def apply_one_model(
        self,
        cpg_id: str,
        X: pd.DataFrame,
        y: pd.Series
        ):
        """
        Predict the methylation for a given CpG site on self.test_samples
        @ cpg_id: the id of the CpG site
        @ X: the mutation feature matrix for the CpG site
        @ y: the target methylation values for the CpG site
        """
        # predict methylation for test samples
        model = self.trained_models[cpg_id]
        y_pred = model.predict(X)
        self.predictions[cpg_id] = y_pred
        # get the number row that corresponds to the test samples in y
        y_test_index = y.index.get_indexer(self.test_samples)
        # get predictions for test samples
        y_pred_test = y_pred[y_test_index]
        y_test = y.loc[self.test_samples]
        # measure performance on test samples
        
        mae = np.mean(np.abs(y_test - y_pred_test))
        # check if y_pred_test is constant
        if np.var(y_pred_test) == 0:
            pearsonr = 0
            spearman = 0
        else:
            pearsonr = np.corrcoef(y_test, y_pred_test)[0,1]
            # get spearman corr coef also
            spearman = spearmanr(y_test, y_pred_test)[0]
        # if spearman is nan, set it to 0
        if np.isnan(spearman):
            spearman = 0
        if np.isnan(pearsonr):
            pearsonr = 0
        # robust linear regression and F-test
        """model = sm.robust.robust_linear_model.RLM(y_test, y_pred_test).fit()
        f_test = model.f_test(np.array([0, 1]))
        robust_f_test_pval = f_test.pvalue
        robust_f_test_fstat = f_test.F"""
        self.prediction_performance[cpg_id] = {
            'testing_methyl_pearsonr': pearsonr, 
            'testing_methyl_mae': mae,
            'testing_methyl_spearmanr': spearman}
        

    def apply_all_models(
        self,
        only_agg: bool = False
        ) -> None:
        """
        Predict methylation for all CpG in mutation feature store for test_samples
        using the models trained by train_all_models stored in self.trained_models
        @ mut_feat_store: the path to the mutation features store
        """       
        # for each cpg in the store, apply its trained model
        for i, cpg_id in enumerate(self.mut_feat_store['cpg_ids']):
            if self.scramble:
                X = self.mut_feat_store['feat_mats'][cpg_id]
                # randomly scramble the rows of the feature matrix, but keep the same order of samples
                save_index = X.index.copy(deep=True)
                X = X.sample(frac=1, random_state = 0)
                X.index = save_index
            else:
                if only_agg:
                    X = self.mut_feat_store['feat_mats'][cpg_id].loc[:, ~self.mut_feat_store['feat_mats'][cpg_id].columns.str.contains(':')]
                else:
                    X = self.mut_feat_store['feat_mats'][cpg_id]
            # do prediction
            self.apply_one_model(
                cpg_id = cpg_id,
                X = X,
                y = self.mut_feat_store['target_values'][cpg_id],
                )
            if i % 10 == 0:
                print(f'Predicted methylation for {i} CpGs of {len(self.mut_feat_store["cpg_ids"])}', flush=True)
        self.pred_df = pd.DataFrame(self.predictions, index = self.train_samples + self.test_samples)
        self.perf_df = pd.DataFrame(self.prediction_performance).T
        return

    def train_one_model(
        self,
        cpg_id: str,
        X: pd.DataFrame,
        y: pd.Series
        ) -> None:
        """
        Train a model of given type to predict methylation for a certain CpG
        @ cpg_id: the id of the CpG site
        @ X: the mutation feature matrix for the CpG site
        @ y: the target methylation values for the CpG site
        @ model_type: the type of model to train
        @ returns: None
        """
        if self.model_type == 'elasticNet':
            model = ElasticNetCV(
                cv=5, random_state=0, max_iter=5000,
                selection = 'random', n_jobs=-1
                )
        elif self.model_type == 'linreg':
            model = sklearn.linear_model.LinearRegression()
        elif self.model_type == 'rand_forest':
            model = sklearn.ensemble.RandomForestRegressor()
        elif self.model_type == 'lasso':
            model = sklearn.linear_model.LassoCV()
        elif self.model_type == 'xgboost':
            model = xgb.XGBRegressor()
        
        # fit model to training samples
        model.fit(X.loc[self.train_samples], y.loc[self.train_samples]) 
        # add to trained models dictionary
        self.trained_models[cpg_id] = model
        return
    
    def train_all_models(
        self, 
        only_agg: bool = False
        ) -> None:
        """
        Given a mutation features store, train a model for each
        cpg and features in the store
        @ mut_feat_store: the path to the mutation features store
        @ returns: None
        """    
        # for each cpg in the store train a model
        for i, cpg_id in enumerate(self.mut_feat_store['cpg_ids']):
            if self.scramble:
                X = self.mut_feat_store['feat_mats'][cpg_id]
                # randomly scramble the rows of the feature matrix, but keep the same order of samples
                save_index = X.index.copy(deep=True)
                X = X.sample(frac=1, random_state = 0)
                X.index = save_index
            else:
                if only_agg:
                    X = self.mut_feat_store['feat_mats'][cpg_id].loc[:, ~self.mut_feat_store['feat_mats'][cpg_id].columns.str.contains(':')]
                else:
                    X = self.mut_feat_store['feat_mats'][cpg_id]
            # for each feature set in the store train a model
            self.train_one_model(
                cpg_id = cpg_id,
                X = X,
                y = self.mut_feat_store['target_values'][cpg_id]
                )
            if i % 10 == 0:
                print(f"done {i} CpGs of {len(self.mut_feat_store['cpg_ids'])}", flush=True)
        return
    
    def save_models_and_preds(
        self,
        out_dir: str = ""
        ) -> None:
        """
        Write out the trained models, predictions, and performances to files
        """
        # default output directory is where the feature store came from
        if out_dir == "":
            out_dir = self.mut_feat_store_fns[0][:self.mut_feat_store_fns[0].rfind('/')]
        # write out files to there, including the model type in name
        with open(f"{out_dir}/trained_models_{self.model_type}_{self.scramble}scramble.pkl", 'wb') as f:
            pickle.dump(self.trained_models, f)
        # write to parquet files
        self.pred_df.to_parquet(f"{out_dir}/methyl_predictions_{self.model_type}_{self.scramble}scramble.parquet")
        self.perf_df.to_parquet(f"{out_dir}/prediction_performance_{self.model_type}_{self.scramble}scramble.parquet")
        print(f"wrote out trained models, predictions, and performances to {out_dir}", flush=True)
        
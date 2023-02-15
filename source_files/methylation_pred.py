import pandas as pd
import numpy as np
import pickle
import sklearn
import xgboost as xgb

class methylationPrediction:
    """
    A class which trains models to predict methylation for a given set
    of CpG sites and samples using the feature matrices generated by a
    mutationFeatures object
    """
    def __init__(
        self,
        mut_feat_store_fns: list,
        model_type: str
        ) -> None:
        """
        Constructor for methylationPrediction object
        @ mut_feat_store_fns: list of path(s) to the mutation feature store(s)
        @ model_type: str, either "xgboost", "linreg", "lasso", "elasticNet", or "rand_forest"
        @ returns: None
        """
        self.mut_feat_store_fns = mut_feat_store_fns
        # combine the mutation feature stores into one
        self.mut_feat_store = self.combine_feat_stores()
        # set the train and test samples to be same as those used to generate the mutation feature store
        self.train_samples = self.mut_feat_store['train_samples']
        self.test_samples = self.mut_feat_store['test_samples']
        self.model_type = model_type
        self.trained_models = {}
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
        # read each into a list
        mut_feat_stores = []
        for mut_feat_store_fn in self.mut_feat_store_fns:
            with open(mut_feat_store_fn, 'rb') as f:
                mut_feat_stores.append(pickle.load(f))
        # for each key of each store, add the values to the combined store
        for next_mut_feat_store in mut_feat_stores:
            for key in next_mut_feat_store.keys():
                # if key stores a piece of meta data, add it to the combined store just once
                if key in ['dataset', 'train_samples', 'test_samples', 'aggregate', 'num_correl_sites', 'num_correl_ext_sites', 'max_meqtl_sites', 'nearby_window_size']:
                    if key not in mut_feat_store.keys():
                        mut_feat_store[key] = next_mut_feat_store[key]
                # otherwise if key stores a list of data values, each time add the values to the combined store
                elif key == 'cpg_ids':
                    if key not in mut_feat_store.keys():
                        mut_feat_store[key] = next_mut_feat_store[key]
                    else:
                        mut_feat_store[key] = mut_feat_store[key] + next_mut_feat_store[key]
                # otherwise key stores a dictionary of data values
                elif key in ['feat_mats', 'target_values']:
                    if key not in mut_feat_store.keys():
                        mut_feat_store[key] = next_mut_feat_store[key]
                    else:
                        # combine the dictionaries
                        mut_feat_store[key].update(next_mut_feat_store[key])
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
        # subset X and y to test samples
        X_test = X.loc[self.test_samples]
        y_test = y.loc[self.test_samples]
        # predict methylation for test samples
        model = self.trained_models[cpg_id]
        y_pred = model.predict(X_test)
        self.predictions[cpg_id] = y_pred
        # measure performance
        r2 = np.corrcoef(y_test, y_pred)[0,1]**2
        mae = np.mean(np.abs(y_test - y_pred))
        self.prediction_performance[cpg_id] = {'r2': r2, 'mae': mae}
        return

    def apply_all_models(
        self
        ) -> None:
        """
        Predict methylation for all CpG in mutation feature store for test_samples
        using the models trained by train_all_models stored in self.trained_models
        @ mut_feat_store: the path to the mutation features store
        """       
        # for each cpg in the store, apply its trained model
        for i, cpg_id in enumerate(self.mut_feat_store['cpg_ids']):
            self.apply_one_model(
                cpg_id = cpg_id,
                X = self.mut_feat_store['feat_mats'][cpg_id],
                y = self.mut_feat_store['target_values'][cpg_id],
                )
            if i % 100 == 0:
                print(f'Predicted methylation for {i} CpGs of {len(self.mut_feat_store["cpg_ids"])}')
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
            model = sklearn.linear_model.ElasticNetCV()
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
        ) -> None:
        """
        Given a mutation features store, train a model for each
        cpg and features in the store
        @ mut_feat_store: the path to the mutation features store
        @ returns: None
        """    
        # for each cpg in the store train a model
        for i, cpg_id in enumerate(self.mut_feat_store['cpg_ids']):
            # for each feature set in the store train a model
            self.train_one_model(
                cpg_id = cpg_id,
                X = self.mut_feat_store['feat_mats'][cpg_id],
                y = self.mut_feat_store['target_values'][cpg_id]
                )
            if i % 10 == 0:
                print(f"done {i} CpGs of {len(self.mut_feat_store['cpg_ids'])}")
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
        with open(f"{out_dir}/trained_models_{self.model_type}.pkl", 'wb') as f:
            pickle.dump(self.trained_models, f)
        # create dataframes from predictions and performances
        self.pred_df = pd.DataFrame(self.predictions, index = self.test_samples)
        self.perf_df = pd.DataFrame(self.prediction_performance)
        # write to parquet files
        self.pred_df.to_parquet(f"{out_dir}/methyl_predictions_{self.model_type}.parquet")
        self.perf_df.to_parquet(f"{out_dir}/prediction_performance_{self.model_type}.parquet")
        
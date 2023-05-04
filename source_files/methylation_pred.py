import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.linear_model import LinearRegression, ElasticNetCV, ElasticNet
import sklearn.preprocessing
import xgboost as xgb
import sys
#import statsmodels.api as sm
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix


class methylationPrediction:
    """
    A class which trains models to predict methylation for a given set
    of CpG sites and samples using the feature matrices generated by a
    mutationFeatures object
    """
    def __init__(
        self,
        model_type: str,
        baseline: str = 'none',
        mut_feat_store_fns: list = [],
        mut_feat_store: dict = {},
        trained_models_fns: list = [],
        target_values: str = 'target_values'
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
        self.baseline = baseline
        # set the train and test samples to be same as those used to generate the mutation feature store
        self.train_samples = self.mut_feat_store['train_samples']
        self.test_samples = self.mut_feat_store['test_samples']
        self.cross_val_num = self.mut_feat_store['cross_val_num']
        self.model_type = model_type
        self.target_values = target_values
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
                    elif key in ['feat_mats', 'target_values', 'mad_target_values', 'feat_names']:
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
        X,
        y: pd.Series
        ):
        """
        Predict the methylation for a given CpG site on self.test_samples
        @ cpg_id: the id of the CpG site
        @ X: the mutation feature matrix for the CpG site, csr if model_type is xgboost, otherwise numpy array
        @ y: the target methylation values for the CpG site
        """
        
        if self.model_type == 'elasticNet':
            # scale the features of training and testing samples, seperately
            train_idx_num = [
                    self.mut_feat_store[self.target_values][cpg_id].index.get_loc(train_sample)
                    for train_sample in self.train_samples
                    ]
            test_idx_num = [
                    self.mut_feat_store[self.target_values][cpg_id].index.get_loc(test_sample)
                    for test_sample in self.test_samples
                ]
            # Create a MinMaxScaler using the training data
            scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X[train_idx_num, :])
            # Apply the same scaling factors from the training data to the testing data
            X_test = scaler.transform(X[test_idx_num, :])
            # recombine the training and testing samples into one matrix, they are in order of train then test
            # per mutation_features._create_feature_mat()
            X = np.concatenate((X_train, X_test), axis=0)
        
        # predict methylation for all samples
        model = self.trained_models[cpg_id]
        y_pred = model.predict(X)
        self.predictions[cpg_id] = y_pred
        # get the number rows that corresponds to the test samples in y
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
        # if not, calc pearson and spearman corr coef
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
            'testing_methyl_spearmanr': spearman
            }

    def apply_all_models(
        self,
        just_one: bool = False
        ) -> None:
        """
        Predict methylation for all CpG in mutation feature store for test_samples
        using the models trained by train_all_models stored in self.trained_models
        @ mut_feat_store: the path to the mutation features store
        """       
        # for each cpg in the store, apply its trained model
        for i, cpg_id in enumerate(self.mut_feat_store['cpg_ids']):
            if self.baseline == 'cov_only':
                # get the feature matrix for the cpg
                X = self.mut_feat_store['feat_mats'][cpg_id]
                X = pd.DataFrame(X.todense())
                # select only the covariate columns
                feat_names = pd.Series(self.mut_feat_store['feat_names'][cpg_id])
                is_covariate = feat_names.str.contains('dataset') | feat_names.str.contains('gender')
                covariate_cols = feat_names.index[is_covariate].values
                # select only the covariate columns from X
                X = X.iloc[:, covariate_cols]
            else:
                # get the feature matrix for the cpg, sparse
                X = self.mut_feat_store['feat_mats'][cpg_id] 
    
            # do prediction
            self.apply_one_model(
                cpg_id = cpg_id,
                X = X, 
                y = self.mut_feat_store[self.target_values][cpg_id],
                )
            if i % 10 == 0:
                print(f'Predicted methylation for {i} CpGs of {len(self.mut_feat_store["cpg_ids"])}', flush=True)
            if just_one:
                if i == 10:
                    break
        self.pred_df = pd.DataFrame(self.predictions, index = self.train_samples + self.test_samples)
        self.perf_df = pd.DataFrame(self.prediction_performance).T
        return

    def train_one_model(
        self,
        cpg_id: str,
        X: csr_matrix,
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
        y = y.loc[self.train_samples]
        if self.model_type == 'elasticNet':
            # min-max scale the features of sparse matrix X
            scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
            X = scaler.fit_transform(X.toarray())
            model = ElasticNetCV(
                cv=5, random_state=0, max_iter=1000,
                selection = 'random', n_jobs=-1
                )
            model.fit(X, y) 
            
            #model = ElasticNet(selection = 'random')
        elif self.model_type == 'linreg':
            model = sklearn.linear_model.LinearRegression()
        elif self.model_type == 'rand_forest':
            model = sklearn.ensemble.RandomForestRegressor()
        elif self.model_type == 'lasso':
            model = sklearn.linear_model.LassoCV()
        elif self.model_type == 'xgboost':
            # keep as sparse bc xgboost faster this way
            # Create the XGBRegressor model
            model = xgb.XGBRegressor(n_jobs=-1)
            model.fit(X, y)
            
            # Create a parameter grid for the XGBoost model
            
            """from sklearn.model_selection import RandomizedSearchCV
            model = xgb.XGBRegressor()
            param_grid = {
                'learning_rate': np.logspace(-4, 0, 50),
                'n_estimators': range(10, 150, 10),
                'max_depth': range(2, 10),
                'min_child_weight': range(1, 6),
                'gamma': np.linspace(0, 0.5, 50),
                'subsample': np.linspace(0.5, 1, 50),
                'colsample_bytree': np.linspace(0.5, 1, 50),
                'reg_alpha': np.logspace(-4, 0, 50),
                'reg_lambda': np.logspace(-4, 0, 50)
            }
            # Initialize the RandomizedSearchCV object
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=100,  # number of parameter settings that are sampled
                scoring='neg_mean_squared_error',
                #n_jobs=-1,
                cv=5,
                verbose=0,
                random_state=42
            )
            # Fit the RandomizedSearchCV object to the training data
            random_search.fit(X, y)
            # Print the best hyperparameters
            print("Best hyperparameters:", random_search.best_params_)
            # Use the best estimator for predictions or further analysis
            model = random_search.best_estimator_"""
        # fit model to training samples
        # X has already been subsetted to only contain training samples in order
        # add to trained models dictionary
        self.trained_models[cpg_id] = model
        return
    
    def train_all_models(
        self, 
        just_one: bool = False
        ) -> None:
        """
        Given a mutation features store, train a model for each
        cpg and features in the store
        @ mut_feat_store: the path to the mutation features store
        @ returns: None
        """    
        # for each cpg in the store train a model
        for i, cpg_id in enumerate(self.mut_feat_store['cpg_ids']):
            # get index of training samples from target
            train_idx_num = [
                self.mut_feat_store[self.target_values][cpg_id].index.get_loc(train_sample)
                for train_sample in self.train_samples
                ]
            if self.baseline == 'scramble':
                # get the feature matrix for the cpg
                X = self.mut_feat_store['feat_mats'][cpg_id]
                X = X.todense()
                # subset to only training samples
                X_train = pd.DataFrame(X[train_idx_num, :])
                # scramble the feature matrix
                X_train_scrambled = self.do_scramble(X_train, cpg_id)
                X = csr_matrix(X_train_scrambled)
            elif self.baseline == 'cov_only':
                # get the feature matrix for the cpg
                X = self.mut_feat_store['feat_mats'][cpg_id]
                X = X.todense()
                X = pd.DataFrame(X[train_idx_num, :])
                # select only the covariate columns
                feat_names = pd.Series(self.mut_feat_store['feat_names'][cpg_id])
                is_covariate = feat_names.str.contains('dataset') | feat_names.str.contains('gender')
                covariate_cols = feat_names.index[is_covariate].values
                # select only the covariate columns from X
                X = X.iloc[:, covariate_cols]
                # convert back to sparse
                X = csr_matrix(X)
            elif self.baseline == 'none': # actual model
                # get the feature matrix for the cpg, sparse
                X = self.mut_feat_store['feat_mats'][cpg_id] 
                # subset to only training samples
                X = X[train_idx_num, :]
            else:
                raise ValueError(f"baseline {self.baseline} not supported")
            
            # for each feature set in the store train a model
            self.train_one_model(
                cpg_id = cpg_id,
                X = X, #subset to training samples
                y = self.mut_feat_store[self.target_values][cpg_id] # gets subset in fxn
                )
            if i % 10 == 0:
                print(f"done {i} CpGs of {len(self.mut_feat_store['cpg_ids'])}", flush=True)
            if just_one:
                if i == 10:
                    break
    
    def do_scramble(
        self, 
        X_train: pd.DataFrame, 
        cpg_id: str
        ):
        """
        Scramble the non-covariate rows and columns of the feature matrix
        @ X_train: the feature matrix for the CpG site
        @ cpg_id: the id of the CpG site
        """
        # get column index of covariates, which are columns containing dataset or gender
        feat_names = pd.Series(self.mut_feat_store['feat_names'][cpg_id])
        is_covariate = feat_names.str.contains('dataset') | feat_names.str.contains('gender')
        covariate_cols = feat_names.index[is_covariate].values
        # save covariate columns and sample order
        save_covariate_cols = X_train.iloc[:, covariate_cols].copy(deep = True)
        save_X_train_idx = X_train.index.copy(deep = True)
        
        # drop covariate columns
        X_train_scrambled = X_train.drop(covariate_cols, axis = 1).copy(deep = True)
        # randomly select values from 
        def scramble_dataframe(df):
            vals = df.values
            # shuffle vals and every sub array
            np.random.shuffle(vals)
            for i in range(vals.shape[0]):
                np.random.shuffle(vals[i])
            scrambled_df = pd.DataFrame(vals, index = df.index, columns = df.columns)
            return scrambled_df
        X_train_scrambled = scramble_dataframe(X_train_scrambled)
        
        """X_train_scrambled = X_train_scrambled.sample(frac=1, axis=1, random_state=42, replace = False).sample(frac=1, axis=0, random_state=42, replace = False)"""
        
        # reset columns
        X_train_scrambled.columns = np.arange(X_train_scrambled.shape[1])
        # convert index back to original
        X_train_scrambled.index = save_X_train_idx
        # and add covariates back
        X_train_scrambled = pd.concat([X_train_scrambled, save_covariate_cols], axis = 1)
        return X_train_scrambled
         
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
        with open(f"{out_dir}/trained_models_{self.model_type}_{self.baseline}baseline.pkl", 'wb') as f:
            pickle.dump(self.trained_models, f)
        # write to parquet files
        self.pred_df.to_parquet(f"{out_dir}/methyl_predictions_{self.model_type}_{self.baseline}baseline.parquet")
        self.perf_df.to_parquet(f"{out_dir}/prediction_performance_{self.model_type}_{self.baseline}baseline.parquet")
        print(f"wrote out trained models, predictions, and performances to {out_dir}", flush=True)
        
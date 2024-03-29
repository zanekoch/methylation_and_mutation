import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.linear_model import LinearRegression, ElasticNetCV, ElasticNet
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import os
import sklearn.preprocessing
import xgboost as xgb
import sys
#import statsmodels.api as sm
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix, vstack

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
        target_values: str = 'target_values',
        agg_only: bool = False,
        scale_counts_within_dataset: bool = False,
        predict_with_random_feat: int = -1,
        illumina_cpg_locs_df: pd.DataFrame = pd.DataFrame(), 
        all_methyl_age_df_t: pd.DataFrame = pd.DataFrame(),
        use_gpu: bool = False
        ) -> None:
        """
        Constructor for methylationPrediction object
        @ model_type: str, either "xgboost", "linreg", "lasso", "elasticNet", or "rand_forest"
        @ baseline: str, either "none", "scramble", or "cov_only"
        @ mut_feat_store_fns: list of path(s) to the mutation feature store(s)
        @ mut_feat_store: dict, the mutation feature store
        @ trained_models_fns: list of path(s) to the trained models
        @ target_values: str, the key in the mutation feature store which stores the target values
        @ agg_only: bool, whether to only use aggregate features
        @ scale_counts_within_dataset: bool, whether to min-max scale feat mutation counts by sample tissue type
        @ predict_with_random_feat: int, if > 0, randomly choose this many CpGs from the store to predict with
        @ illumina_cpg_locs_df: pd.DataFrame, the dataframe of illumina cpg locations
        @ all_methyl_age_df_t: pd.DataFrame, the dataframe of all methyl age samples
        @ use_gpu: bool, whether to use gpu for xgboost
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
        self.validation_samples = self.mut_feat_store['validation_samples']
        self.cross_val_num = self.mut_feat_store['cross_val_num']
        self.model_type = model_type
        self.target_values = target_values
        self.agg_only = agg_only
        self.scale_counts_within_dataset = scale_counts_within_dataset
        if self.scale_counts_within_dataset:
            sys.exit("scale_counts_within_dataset deprecated")
        self.predict_with_random_feat = predict_with_random_feat
        # check that is an integer or negative
        assert isinstance(self.predict_with_random_feat, int), "predict_with_random_feat must be an integer"

        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.performance_by_dataset_df = None
        self.use_gpu = use_gpu
        # if predict_with_random_feat is specified, assert that illumina_cpg_locs_df is not empty
        assert self.predict_with_random_feat < 0 or not self.illumina_cpg_locs_df.empty
        # assert these agg_only is not True while baseline is cov_only
        if self.agg_only and self.baseline == 'cov_only':
            sys.exit("agg_only cannot be True while baseline is cov_only")
        # if trained models are provided, read them in
        if len(trained_models_fns) == 0:
            self.trained_models = {}
        else:
            for trained_models_fn in trained_models_fns:
                with open(trained_models_fn, 'rb') as f:
                    self.trained_models = pickle.load(f)
        self.predictions = {}

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
                    if key in ['dataset', 'train_samples', 'test_samples', 'validation_samples', 'aggregate', 'num_correl_sites', 'num_correl_ext_sites', 'max_meqtl_sites', 'nearby_window_size', 'cross_val_num']:
                        if key not in mut_feat_store.keys():
                            mut_feat_store[key] = next_mut_feat_store[key]
                    # otherwise if key stores a list of data values, each time add the values to the combined store
                    elif key == 'cpg_ids':
                        if key not in mut_feat_store.keys():
                            mut_feat_store[key] = next_mut_feat_store[key]
                        else:
                            mut_feat_store[key] = mut_feat_store[key] + next_mut_feat_store[key]
                    elif key in ['feat_mats', 'target_values', 'feat_names']:
                        if key not in mut_feat_store.keys():
                            mut_feat_store[key] = next_mut_feat_store[key]
                        else:
                            # combine the dictionaries
                            mut_feat_store[key].update(next_mut_feat_store[key])
                print(f"Done reading {mut_feat_store_fn}", flush=True)
        return mut_feat_store

    def apply_one_model(
        self,
        train_mat_cpg_id: str,
        target_cpg_id: str,
        X,
        ):
        """
        Predict the methylation for a given CpG site on self.test_samples
        @ train_mat_cpg_id: the id of the CpG site whose mat was used to train the model
        @ target_cpg_id: the id of the CpG site to predict
        @ X: the mutation feature matrix for the CpG site, csr if model_type is xgboost, otherwise numpy array. Contains all samples, in order of train then test then validation
        """
        if self.model_type == 'elasticNet':
            # scale the features of training samples, and apply this to testing and validation samples
            train_idx_num = [
                    self.mut_feat_store[self.target_values][train_mat_cpg_id].index.get_loc(train_sample)
                    for train_sample in self.train_samples
                    ]
            test_idx_num = [
                    self.mut_feat_store[self.target_values][train_mat_cpg_id].index.get_loc(test_sample)
                    for test_sample in self.test_samples
                ]
            validation_idx_num = [
                    self.mut_feat_store[self.target_values][train_mat_cpg_id].index.get_loc(validation_sample)
                    for validation_sample in self.validation_samples
                ]
            # Create a MinMaxScaler using the training data
            scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X[train_idx_num, :])
            # Apply the same scaling factors from the training data to the testing data
            X_test = scaler.transform(X[test_idx_num, :])
            X_validation = scaler.transform(X[validation_idx_num, :])
            # recombine the training and testing samples into one matrix, they are in order of train then test
            # per mutation_features._create_feature_mat()
            X = np.concatenate((X_train, X_test, X_validation), axis=0)
        
        # predict methylation for all samples
        # if predicting with random feat, then use combination as key
        if self.predict_with_random_feat > 0:
            dict_key = 'target_' + target_cpg_id + '_train_' + train_mat_cpg_id
        else:
            dict_key = target_cpg_id
        model = self.trained_models[dict_key]
        y_pred = model.predict(X)
        self.predictions[dict_key] = y_pred
        

    def apply_all_models(
        self,
        just_one: bool = False, 
        ) -> None:
        """
        Predict methylation for all CpG in mutation feature store for test_samples
        using the models trained by train_all_models stored in self.trained_models
        @ mut_feat_store: the path to the mutation features store
        """
        # if predict_with_random_feat is then get the name of cpgs being predicted
        if self.predict_with_random_feat > 0:
            cpg_with_trained_model = self.trained_models.keys()
            # split each on the underscore and take the second part
            target_cpgs = [cpg.split('_')[1] for cpg in cpg_with_trained_model]
            # split each on the underscore and take the third part
            train_mat_cpgs = [cpg.split('_')[3] for cpg in cpg_with_trained_model]
        # otherwise can just use all cpgs in trained_models
        else:
            target_cpgs = self.trained_models.keys()
            train_mat_cpgs = list(target_cpgs) # convert to list 
        
        total_cpgs = len(target_cpgs)
        if not just_one:
            # make sure all cpgs in trained_models identical to those in mut_feat_store, if not then error
            assert set(target_cpgs) == set(self.mut_feat_store['cpg_ids']), \
                "cpgs in trained_models not identical to those in mut_feat_store"
            
        # for each cpg in the store, predict its methylation using the appropriate feat mat and model
        for i, target_cpg_id in enumerate(target_cpgs):
            # get indices of training, testing, and validation samples from target
            train_idx_num = [
                    self.mut_feat_store[self.target_values][target_cpg_id].index.get_loc(train_sample)
                    for train_sample in self.train_samples
                    ]
            test_idx_num = [
                    self.mut_feat_store[self.target_values][target_cpg_id].index.get_loc(test_sample)
                    for test_sample in self.test_samples
                ]
            validation_idx_num = [
                    self.mut_feat_store[self.target_values][target_cpg_id].index.get_loc(validation_sample)
                    for validation_sample in self.validation_samples
                ]
            # get the feature matrix for the cpg used to train the model
            if self.baseline == 'cov_only':
                X = self.mut_feat_store['feat_mats'][train_mat_cpgs[i]]
                X = pd.DataFrame(X.todense())
                # select only the covariate columns
                feat_names = pd.Series(self.mut_feat_store['feat_names'][train_mat_cpgs[i]])
                is_covariate = feat_names.str.contains('dataset') | feat_names.str.contains('gender')
                covariate_cols = feat_names.index[is_covariate].values
                # select only the covariate columns from X
                X = X.iloc[:, covariate_cols]
                X = csr_matrix(X)
            elif self.baseline == 'scramble':
                X = self.mut_feat_store['feat_mats'][train_mat_cpgs[i]]
            else:
                X = self.mut_feat_store['feat_mats'][train_mat_cpgs[i]]
            # subset to only aggregate features
            if self.agg_only:
                X = self.subset_matrix_to_agg_only_feats(X, train_mat_cpgs[i])
            # and/or scale
            # DEPRECATED: scale counts within each dataset
            if self.scale_counts_within_dataset:
                # scale counts within each dset separately for training and testing samples
                scaled_X_train = self.do_scale_counts_within_dataset(
                    X[train_idx_num, :], train_mat_cpgs[i]
                    )
                scaled_X_test = self.do_scale_counts_within_dataset(
                    X[test_idx_num, :], train_mat_cpgs[i]
                    )
                scaled_X_validation = self.do_scale_counts_within_dataset(
                    X[validation_idx_num, :], train_mat_cpgs[i]
                    )
                # combine the three csr matrices
                X = vstack([scaled_X_train, scaled_X_test, scaled_X_validation])
                
            # do prediction
            self.apply_one_model(
                train_mat_cpg_id = train_mat_cpgs[i],
                target_cpg_id = target_cpg_id,
                X = X
                )
            if i % 10 == 0:
                print(f'Predicted methylation for {i} CpGs of {total_cpgs}', flush=True)
            if just_one:
                break
        # combine all predictions into one dataframe
        self.pred_df = pd.DataFrame(self.predictions, index = self.train_samples + self.test_samples + self.validation_samples)
        return

    def train_one_model(
        self,
        cpg_id: str,
        X: csr_matrix,
        y: pd.Series,
        cpg_for_train: str = ''
        ) -> None:
        """
        Train a model of given type to predict methylation for a certain CpG
        @ cpg_id: the id of the CpG site
        @ X: the mutation feature matrix for the CpG site
        @ y: the target methylation values for the CpG site
        @ cpg_for_train: the id of the CpG site to use for training, if using random features
        @ returns: None
        """
        # subset to only training samples
        y = y.loc[self.train_samples]
        # train a specific model type
        if self.model_type == 'elasticNet':
            # min-max scale the features of sparse matrix X
            scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
            X = scaler.fit_transform(X.toarray())
            model = ElasticNetCV(
                cv=5, random_state=0, max_iter=1000,
                selection = 'random', n_jobs=-1
                )
            model.fit(X, y) 
        elif self.model_type == 'linreg':
            model = sklearn.linear_model.LinearRegression()
        elif self.model_type == 'rand_forest':
            model = sklearn.ensemble.RandomForestRegressor()
        elif self.model_type == 'lasso':
            model = sklearn.linear_model.LassoCV()
        elif self.model_type == 'xgboost':
            # keep as sparse bc xgboost faster this way
            # Create the XGBRegressor model
            if self.use_gpu:
                model = xgb.XGBRegressor(
                    #n_jobs=-1,
                    learning_rate = .1,
                    objective = 'reg:squarederror',
                    tree_method = 'gpu_hist'
                    )
            else:
                model = xgb.XGBRegressor(
                    n_jobs=-1,
                    learning_rate = .1,
                    objective = 'reg:squarederror',
                    #tree_method = 'gpu_hist'
                    )
            model.fit(X, y)
            
            # Create a parameter grid for the XGBoost model
            """
            From paper
            loss=“deviance”
            learning_rate=0.1
            n_estimators=500
            max_depth=3
            max_features=“log2”
            """
            """model = xgb.XGBRegressor(
                learning_rate = .1,
                objective = 'reg:squarederror',
                tree_method = 'gpu_hist'
                )
            
            param_grid = {
                'n_estimators': range(50, 750, 100), # 10
                'max_depth': range(2, 10, 2), # 8
                'min_child_weight': range(1, 6, 2), # 3
                #'gamma': np.linspace(0, 0.5, 50),
                #'subsample': np.linspace(0.5, 1, 5),
                #'colsample_bytree': np.linspace(0.5, 1, 5),
                'reg_alpha': np.linspace(0, 1, 3), # 5
                'reg_lambda': np.linspace(0, 1, 3), #5
            }
            # grid search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                cv=5,
                verbose=0
            )
            grid_search.fit(X, y)
            model = grid_search.best_estimator_"""
            
            
        # fit model to training samples
        # X has already been subsetted to only contain training samples in order
        # add to trained models dictionary
        if cpg_for_train != '':
            dict_key = 'target_' + cpg_id + '_train_' + cpg_for_train
        else:
            dict_key = cpg_id
        self.trained_models[dict_key] = model
        return
    
    def train_all_models(
        self, 
        just_one: bool = False,
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
            if self.predict_with_random_feat < 0:
                # get the feature matrix for this cpg
                # based on specified baseline, scaling, and agg 
                X = self.make_training_mat(cpg_id, train_idx_num)
                # train a model
                self.train_one_model(
                    cpg_id = cpg_id,
                    X = X, # already subset to training samples
                    y = self.mut_feat_store[self.target_values][cpg_id] # gets subset in fxn
                    )
            else: # in this case randomly choose feature matrixes to predict with
                # exclude same chrom CpGs from random selection
                target_cpg_chrom = self.illumina_cpg_locs_df.loc[
                    self.illumina_cpg_locs_df['#id'] == cpg_id, 'chr'
                    ].values[0]
                diff_chrom_cpgs = set(self.illumina_cpg_locs_df.query(
                    "chr != @target_cpg_chrom"
                    )['#id'].values)
                to_choose_from = list(set(self.mut_feat_store['cpg_ids']).intersection(diff_chrom_cpgs))
                # randomly choose predict_with_random_feat CpGs from the store, without replacement
                random_cpgs = np.random.choice(
                    to_choose_from,
                    size = self.predict_with_random_feat,
                    replace = False
                    )
                # choose the first predict_with_random_feat of these
                if len(random_cpgs) >= self.predict_with_random_feat:
                    random_cpgs = random_cpgs[:self.predict_with_random_feat]
                else:
                    print(
                        f"WARNING: only {len(random_cpgs)} CpGs on different chromosomes than target CpG, using all of them",
                        flush=True
                        )
                # append the target cpg to the list of cpgs to predict with, so that it is trained on itself
                random_cpgs = np.append(random_cpgs, cpg_id)
                # iterate through these random cpg's feat mats and train a model to predict the target cpg
                for j, cpg_for_train in enumerate(random_cpgs):
                    X = self.make_training_mat(cpg_for_train, train_idx_num)
                    # train a model
                    self.train_one_model(
                        cpg_id = cpg_id, # train a model to predict target cpg
                        X = X, # already subset to training samples
                        y = self.mut_feat_store[self.target_values][cpg_id], # gets subset in fxn
                        cpg_for_train = cpg_for_train
                        )
                    if j % 50 == 0:
                        print(f"INNER: done training {j} CpGs of {len(random_cpgs)}", flush=True)
            if i % 10 == 0:
                print(f"OUTER: done training {i} CpGs of {len(self.mut_feat_store['cpg_ids'])}", flush=True)
            if just_one:
                break

    def make_training_mat(self, cpg_id, train_idx_num):
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
            X_train_df = pd.DataFrame(X.todense(), columns = self.mut_feat_store['feat_names'][cpg_id])
            # select columns that do not contain dataset or gender
        else:
            raise ValueError(f"baseline {self.baseline} not supported")
        # subset to only aggregate features and scale if specified
        #print("before agg only")
        #print(pd.DataFrame(X.todense()).sum(axis = 0).sort_values(ascending=False))
        if self.agg_only:
            X = self.subset_matrix_to_agg_only_feats(X, cpg_id)
        #print("after agg onlys")
        #print(pd.DataFrame(X.todense()).sum(axis = 0).sort_values(ascending=False))
        
        # DEPRECATED
        if self.scale_counts_within_dataset:
            # only given train samples because X is already subsetted to train samples
            X = self.do_scale_counts_within_dataset(X, cpg_id)
        return X
    
    def do_scale_counts_within_dataset(
        self, 
        feat_mat: csr_matrix,
        cpg_id: str,
        ) -> csr_matrix:
        """
        Given a matrix, scale the mutation counts within each dataset group
        @ feat_mat: the feature matrix to scale, may be training, testing, or validation
        @ cpg_id: the id of the CpG site
        """
        feat_names = pd.Series(self.mut_feat_store['feat_names'][cpg_id])
        # if agg_only, only select the aggregate features
        if self.agg_only:
            feat_names = feat_names[feat_names.str.contains('agg|dataset|gender')]
        feat_df = pd.DataFrame(feat_mat.toarray(), columns=feat_names)
        
        # select only the features which are dataset
        dset_columns = feat_df.columns[feat_df.columns.str.contains('dataset')]
        dset_feat_df = feat_df[dset_columns].copy(deep = True)
        # select only the features which are gender
        gender_columns = feat_df.columns[feat_df.columns.str.contains('gender')]
        gender_feat_df = feat_df[gender_columns]
        # turn back into one column
        dset_col = dset_feat_df.idxmax(axis = 1)
        # drop covariate columns
        non_cov_df = feat_df.drop(dset_columns, axis = 1).drop(gender_columns, axis = 1)
        non_cov_df['dataset'] = dset_col
        cols_to_scale = non_cov_df.columns[:-1]  
        
        # min-max scale the counts within each dataset
        non_cov_df = non_cov_df.groupby('dataset')[cols_to_scale].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
            )
        # fill na with 0
        non_cov_df.fillna(0, inplace = True)
        # add back dataset columns
        non_cov_df = pd.concat([non_cov_df, gender_feat_df, dset_feat_df], axis = 1)
        # convert to csr matrix
        scaled_feat_mat = csr_matrix(non_cov_df)
        return scaled_feat_mat
        
    
    def subset_matrix_to_agg_only_feats(
        self, 
        feat_mat: csr_matrix,
        cpg_id: str
        ) -> csr_matrix:
        """
        Given a matrix, subset its feature matrix to only include aggregate features
        """
        feat_names = pd.Series(self.mut_feat_store['feat_names'][cpg_id])
        feat_mat = pd.DataFrame(feat_mat.toarray(), columns=feat_names)
        # select only the features which are aggregate features or covariates (dataset, gender)
        selected_columns = feat_mat.columns[feat_mat.columns.str.contains('dataset|agg|gender')]
        feat_mat = feat_mat[selected_columns]
        # convert to csr matrix
        feat_mat = csr_matrix(feat_mat)
        return feat_mat

    
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
        print(feat_names[is_covariate])
        covariate_cols = feat_names.index[is_covariate].values
        # save covariate columns and sample order
        save_covariate_cols = X_train.iloc[:, covariate_cols].copy(deep = True)
        save_X_train_idx = X_train.index.copy(deep = True)
        # drop covariate columns
        print(X_train.sum(axis = 0).sort_values(ascending=False))
        
        X_train_scrambled = X_train.drop(covariate_cols, axis = 1).copy(deep = True)
        print(X_train_scrambled.sum(axis = 0).sort_values(ascending=False))
        # randomly select values from 
        def scramble_dataframe(df):
            # scramble values within each sample, seperately
            # this preserves the mutation burden within a scrambled sample 
            # (setting random_seed makes them all scramble the same)
            print("in scrambled fxn")
            print(df.sum(axis = 0).sort_values(ascending=False))
            scrambled = df.apply(
                lambda x: x.sample(frac=1, replace=False).values,
                axis = 1
                )
            #print(scrambled.sum(axis = 0).sort_values(ascending=False))
            # convert back to dataframe, keeping index and col order
            scrambled_df = pd.DataFrame(
                scrambled.values.tolist(),
                index = df.index,
                columns = df.columns
                )
            print(scrambled_df.sum(axis = 0).sort_values(ascending=False))
            print("finished scrambled fxn")
            
            """
            Old way
            vals = df.values
            # shuffle vals and every sub array
            np.random.shuffle(vals)
            for i in range(vals.shape[0]):
                np.random.shuffle(vals[i])
            scrambled_df = pd.DataFrame(vals, index = df.index, columns = df.columns)"""
            return scrambled_df
        
        # scramble values within each sample
        X_train_scrambled = scramble_dataframe(X_train_scrambled)
        # reset columns
        #X_train_scrambled.columns = np.arange(X_train_scrambled.shape[1])
        # convert index back to original
        X_train_scrambled.index = save_X_train_idx
        # and add covariates back
        X_train_scrambled = pd.concat([X_train_scrambled, save_covariate_cols], axis = 1)
        return X_train_scrambled
         
    def calc_performance_by_dataset(self):
        """
        Get the performance of the models by dataset
        """
        top_20_datasets = self.all_methyl_age_df_t['dataset'].value_counts().index[:20].to_list()
        # remove certain vals from top_20_datasets
        try:
            for dset in top_20_datasets:
                if dset == 'BRCA' or dset == 'LGG'  or dset == 'OV' or dset == 'LAML':
                    top_20_datasets.remove(dset)
        except:
            # ICGC data doesn't have these datasets
            pass
        # if we predicted with random features
        if self.predict_with_random_feat > 0:
            # convert the predicted methyl df columns to the target cpg names
            # splitting each column name on the underscore and taking the first element
            target_cpgs = [col.split('_')[1] for col in self.pred_df.columns]
            target_train_names = self.pred_df.columns
            pred_for_corr_df = self.pred_df.copy(deep = True)
            pred_for_corr_df.columns = target_cpgs
        else:
            target_cpgs = self.pred_df.columns
            target_train_names = self.pred_df.columns
            pred_for_corr_df = self.pred_df
        
        print(top_20_datasets)
        dataset_perf_dfs = []
        for dataset in top_20_datasets:
            # get the correlation between actual testing sample methylation
            # and predicted testing sample methylation from this dataset
            this_dataset_samples = self.all_methyl_age_df_t.loc[
                self.all_methyl_age_df_t['dataset'] == dataset, 
                :].index
            this_dataset_samples = list(
                set(this_dataset_samples).intersection(set(pred_for_corr_df.index))
                )
            this_dataset_train_samples = list(
                set(this_dataset_samples).intersection(set(self.train_samples))
                )
            this_dataset_validation_samples = list(
                set(this_dataset_samples).intersection(set(self.validation_samples))
                )
            this_dataset_test_samples = list(
                set(this_dataset_samples).intersection(set(self.test_samples))
                )
            
            real_methyl_df_test = self.all_methyl_age_df_t.loc[
                this_dataset_test_samples, 
                target_cpgs
                ]
            pred_methyl_df_test = pred_for_corr_df.loc[
                this_dataset_test_samples, 
                :]
            # same for validation samples
            real_methyl_df_validation = self.all_methyl_age_df_t.loc[
                this_dataset_validation_samples,
                target_cpgs
                ]
            pred_methyl_df_validation = pred_for_corr_df.loc[
                this_dataset_validation_samples,
                :]
            # same for training samples
            real_methyl_df_train = self.all_methyl_age_df_t.loc[
                this_dataset_train_samples, 
                target_cpgs
                ]
            pred_methyl_df_train = pred_for_corr_df.loc[
                this_dataset_train_samples, 
                :]
            # get the correlation 
            test_methyl_corr = real_methyl_df_test.corrwith(pred_methyl_df_test, method = 'pearson')
            this_dataset_test_age_df = self.all_methyl_age_df_t.loc[
                this_dataset_test_samples, 
                'age_at_index'
                ]
            test_age_corr = pred_methyl_df_test.corrwith(
                this_dataset_test_age_df, method = 'pearson'
                )
            # same for validation samples
            validation_methyl_corr = real_methyl_df_validation.corrwith(
                pred_methyl_df_validation, method = 'pearson'
                )
            this_dataset_validation_age_df = self.all_methyl_age_df_t.loc[
                this_dataset_validation_samples,
                'age_at_index'
                ]
            validation_age_corr = pred_methyl_df_validation.corrwith(
                this_dataset_validation_age_df, method = 'pearson'
                )
            # same for training samples
            train_methyl_corr = real_methyl_df_train.corrwith(
                pred_methyl_df_train, method = 'pearson'
                ) 
            this_dataset_train_age_df = self.all_methyl_age_df_t.loc[
                this_dataset_train_samples, 
                'age_at_index'
                ]            
            train_age_corr = real_methyl_df_train.corrwith(
                this_dataset_train_age_df, method = 'pearson'
                )
            # create dataframe
            dataset_perf_df = pd.DataFrame({
                'AvP_methyl_pearson_test': test_methyl_corr,
                'AvP_methyl_pearson_train': train_methyl_corr,
                'AvP_methyl_pearson_validation': validation_methyl_corr,
                'PvAge_pearson_test': test_age_corr,
                'PvAge_pearson_train': train_age_corr,
                'PvAge_pearson_validation': validation_age_corr
                })
            dataset_perf_df['dataset'] = dataset
            dataset_perf_df['cpg'] = target_train_names
            dataset_perf_dfs.append(dataset_perf_df)
            print("Done getting performance of dataset: " + dataset, flush = True)
        all_dataset_perf_df = pd.concat(dataset_perf_dfs)
        # make cpg a column
        all_dataset_perf_df.reset_index(inplace = True, drop = True)
        if self.predict_with_random_feat > 0:
            all_dataset_perf_df['self_pred'] = all_dataset_perf_df['cpg'].apply(
                lambda x: True if x.split('_')[1] == x.split('_')[3] else False
                )
        self.performance_by_dataset_df = all_dataset_perf_df 
         
    def save_models_and_preds(
        self,
        out_dir: str = ""
        ) -> None:
        """
        Write out the trained models and predictions to files
        """
        # default output directory is where the feature store came from
        if out_dir == "":
            out_dir = self.mut_feat_store_fns[0][:self.mut_feat_store_fns[0].rfind('/')]
            
        agg_only_str = ''
        if self.agg_only:
            agg_only_str = '_agg_only'
        else:
            agg_only_str = '_all_feats'
        if self.predict_with_random_feat > 0:
            predict_with_random_feat_str = f"_predict_with_random_feat_{self.predict_with_random_feat}"
        else:
            predict_with_random_feat_str = ''
        # write out files to there, including the model type in name
        with open(f"{out_dir}/trained_models_{self.model_type}_{self.baseline}baseline{agg_only_str}{predict_with_random_feat_str}.pkl", 'wb') as f:
            pickle.dump(self.trained_models, f)
        
        # write to parquet files
        self.pred_df.to_parquet(
            f"{out_dir}/methyl_predictions_{self.model_type}_{self.baseline}baseline{agg_only_str}{predict_with_random_feat_str}.parquet"
            )
        print(f"wrote out trained models and predictions to {out_dir}", flush=True)
        if self.performance_by_dataset_df is not None:
            self.performance_by_dataset_df.to_parquet(
                f"{out_dir}/performance_by_dataset_{self.model_type}_{self.baseline}baseline{agg_only_str}{predict_with_random_feat_str}.parquet"
                )
            print(f"wrote out performance by dataset to {out_dir}", flush=True)
        
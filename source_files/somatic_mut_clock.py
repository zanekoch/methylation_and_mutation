import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import time
import seaborn as sns

from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNetCV, RidgeCV, LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

class mutationClock:
    def __init__(
        self,
        all_mut_w_age_df: pd.DataFrame,
        illumina_cpg_locs_df: pd.DataFrame, 
        all_methyl_age_df_t: pd.DataFrame,
        output_dir: str,
        matrix_qtl_dir: str = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/muts",
        tissue_type: str = ""
        ) -> None:
        self.all_mut_w_age_df = all_mut_w_age_df
        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.output_dir = output_dir
        self.matrix_qtl_dir = matrix_qtl_dir
        # if a mut_loc column does not exit, add it
        if 'mut_loc' not in self.all_mut_w_age_df.columns:
            self.all_mut_w_age_df['mut_loc'] = self.all_mut_w_age_df['chr'] + ':' \
                                         + self.all_mut_w_age_df['start'].astype(str)
        # only non X and Y chromosomes and that occured in samples with measured methylation
        self.all_mut_w_age_df = self.all_mut_w_age_df.loc[
            (self.all_mut_w_age_df['chr'] != 'X') 
            & (self.all_mut_w_age_df['chr'] != 'Y')
            & (self.all_mut_w_age_df['chr'] != 'MT')
            & (self.all_mut_w_age_df['case_submitter_id'].isin(self.all_methyl_age_df_t.index)),
            :]
        # join self.all_mut_w_age_df with the illumina_cpg_locs_df
        all_mut_w_age_illum_df = self.all_mut_w_age_df.copy(deep=True)
        all_mut_w_age_illum_df['start'] = pd.to_numeric(self.all_mut_w_age_df['start'])
        self.all_mut_w_age_illum_df = all_mut_w_age_illum_df.merge(
                                        self.illumina_cpg_locs_df, on=['chr', 'start'], how='left'
                                        )
        # subset illumina_cpg_locs_df to only the CpGs that are measured, and remove XY
        self.illumina_cpg_locs_df = self.illumina_cpg_locs_df.loc[
            self.illumina_cpg_locs_df['#id'].isin(self.all_methyl_age_df_t.columns)
            & (self.illumina_cpg_locs_df['chr'] != 'X') 
            & (self.illumina_cpg_locs_df['chr'] != 'Y')
            ]
        # drop CpGs that are not in the illumina_cpg_locs_df (i.e. on XY)
        self.all_methyl_age_df_t = self.all_methyl_age_df_t.loc[:, 
            set(self.all_methyl_age_df_t.columns).intersection(set(self.illumina_cpg_locs_df['#id'].to_list() + ['dataset', 'gender', 'age_at_index']))
            ]
        # one hot encode gender and tissue type
        dset_col = self.all_methyl_age_df_t['dataset'].to_list()
        self.all_methyl_age_df_t = pd.get_dummies(self.all_methyl_age_df_t, columns=["gender", "dataset"])
        # add back in the dataset column
        self.all_methyl_age_df_t['dataset'] = dset_col
        # cache :P
        self.matrixQTL_store = {}
        # if tissue type is specified, subset the data to only this tissue type
        if tissue_type != "":
            self.all_methyl_age_df_t = self.all_methyl_age_df_t.loc[self.all_methyl_age_df_t['dataset'] == tissue_type, :]
            self.all_mut_w_age_df = self.all_mut_w_age_df.loc[self.all_mut_w_age_df['dataset'] == tissue_type, :]
        
        
            
    def evaluate_predictor(
        self, 
        cpg_id: str,
        train_samples: list,
        num_correl_sites: int,
        max_meqtl_sites: int,
        nearby_window_size: int,
        aggregate: str = "False",
        binarize: bool = False,
        feat_store: str = "",
        scramble: bool = False
        ) -> pd.DataFrame:
        """
        Evaluate the predictor for one CpG
        @ cpg_id: the id of the CpG
        @ train_samples: the samples to be used for training
        @ num_correl_sites: the number of corr sites to be used as predictors
        @ max_meqtl_sites: the maximum number of matrixQTL sites to be used as predictors
        @ nearby_window_size: the window size to be used for nearby sites
        @ aggregate: whether to aggregate the mutation status by predictor group
        @ return: a dataframe with model results
        """
        if feat_store == "": # not reading from saved pkl file
            # get the sites to be used as predictors
            predictor_groups = self.get_predictor_site_groups(
                cpg_id, train_samples, num_correl_sites,
                max_meqtl_sites, nearby_window_size
                )
            pred_group_fn = os.path.join(self.output_dir, f"{cpg_id}_pred_groups.pkl")
            pickle.dump(predictor_groups, open(pred_group_fn, "wb"))
        else: # read from saved pkl file
            predictor_groups_fn = os.path.join(feat_store, f"{cpg_id}.pkl")
            # open the file
            predictor_groups = pickle.load(open(predictor_groups_fn, 'rb'))
        # if empty for some reason, return empty dataframe
        if predictor_groups == {}:
            return pd.DataFrame()
        
        model = ElasticNetCV(cv=5, random_state=0, max_iter=5000, n_jobs=-1, selection='random')
        # actual model
        # create training matrix
        X, y = self._create_training_mat(
            cpg_id, predictor_groups, train_samples, 
            aggregate = aggregate, binarize = binarize
            )        
        cv = KFold(n_splits=3, shuffle=True, random_state=0)
        # do the cross validation
        maes, r2s, feature_names, coefs, intercepts = [], [], [], [], []
        for train_index, test_index in cv.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            maes.append(mean_absolute_error(y_test, preds))
            r2s.append(r2_score(y_test, preds))
            feature_names.append(X_train.columns[model.coef_ != 0].to_list())
            coefs.append(model.coef_[model.coef_ != 0])
            intercepts.append(model.intercept_)
            
        # baseline model
        base_maes, base_r2s, base_feature_names, base_coefs, base_intercepts = [], [], [], [], []
        coviariate_col_names = [
            col for  col in X.columns if col.startswith('dataset_') or col.startswith('gender_')
            ]
        X_cov_only = X[coviariate_col_names]
        for train_index, test_index in cv.split(X):
            X_train, X_test = X_cov_only.iloc[train_index, :], X_cov_only.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            base_maes.append(mean_absolute_error(y_test, preds))
            base_r2s.append(r2_score(y_test, preds))
            base_feature_names.append(X_train.columns[model.coef_ != 0].to_list())
            base_coefs.append(model.coef_[model.coef_ != 0])
            base_intercepts.append(model.intercept_)
            
        # if scramble, randomly shuffle the mutation rows of X, but not covariate
        if scramble:
            # shuffle rows of the mutation columns
            non_coviariate_col_names = [
                col for  col in X.columns \
                if not col.startswith('dataset_') and not col.startswith('gender_')
                ]
            scrambled_X = X[non_coviariate_col_names].copy()
            scrambled_X = scrambled_X.sample(frac = 1, random_state = 0).reset_index(drop=True)
            # add the unshuffled covariate columns back
            covariate_df = X.loc[:, coviariate_col_names]
            # concat ignoring the index
            scrambled_X.index = covariate_df.index
            scrambled_X = pd.concat([scrambled_X, covariate_df], axis = 1)
            scrambled_maes, scrambled_r2s, scrambled_feature_names, scrambled_coefs, scrambled_intercepts = [], [], [], [], []
            # train again again but with scrambled X
            for train_index, test_index in cv.split(X):
                X_train, X_test = scrambled_X.iloc[train_index, :], scrambled_X.iloc[test_index, :]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                scrambled_maes.append(mean_absolute_error(y_test, preds))
                scrambled_r2s.append(r2_score(y_test, preds))
                scrambled_feature_names.append(X_train.columns[model.coef_ != 0].to_list())
                scrambled_coefs.append(model.coef_[model.coef_ != 0])
                scrambled_intercepts.append(model.intercept_)
        
            result_df = pd.DataFrame({
                'cpg_id': cpg_id, 'mae': np.mean(maes), 'r2': np.mean(r2s), 'feature_names': [feature_names], 'coefs': [coefs], 'intercepts': [intercepts], 
                'base_mae': np.mean(base_maes), 'base_r2': np.mean(base_r2s), 'base_feature_names': [base_feature_names], 'base_coefs': [base_coefs], 'base_intercepts': [base_intercepts],
                'scrambled_mae': np.mean(scrambled_maes), 'scrambled_r2': np.mean(scrambled_r2s), 'scrambled_feature_names': [scrambled_feature_names], 'scrambled_coefs': [scrambled_coefs], 'scrambled_intercepts': [scrambled_intercepts]
                })
            return result_df
        result_df = pd.DataFrame({
                'cpg_id': cpg_id, 'mae': np.mean(maes), 'r2': np.mean(r2s), 'feature_names': [feature_names], 'coefs': [coefs], 'intercepts': [intercepts], 
                'base_mae': np.mean(base_maes), 'base_r2': np.mean(base_r2s), 'base_feature_names': [base_feature_names], 'base_coefs': [base_coefs], 'base_intercepts': [base_intercepts]
                })
        return result_df
        
    def feature_informations(self, 
        cpg_id: str,
        train_samples: list,
        num_correl_sites: int,
        max_meqtl_sites: int,
        nearby_window_size: int,
        aggregate: str = 'True',
        binarize: bool = False,
        feat_store: str = ""
        ) -> list:
        """
        Calculate the mutual information (across samples) between each feature and the target MFs
        @ cpg_id: the id of the CpG
        @ train_samples: the samples to be used for training
        @ num_correl_sites: the number of corr sites to be used as predictors
        @ max_meqtl_sites: the maximum number of matrixQTL sites to be used as predictors
        @ nearby_window_size: the window size to be used for nearby sites
        @ aggregate: whether to aggregate the mutation status by predictor group
        @ return: a list of mutual info values for each feature
        """
        if feat_store == "": # not reading from saved pkl file
            # get the sites to be used as predictors
            predictor_groups = self.get_predictor_site_groups(
                cpg_id, train_samples, num_correl_sites,
                max_meqtl_sites, nearby_window_size
                )
            pred_group_fn = os.path.join(self.output_dir, f"{cpg_id}_pred_groups.pkl")
            pickle.dump(predictor_groups, open(pred_group_fn, "wb"))
        else: # read from saved pkl file if can
            try:
                predictor_groups_fn = os.path.join(feat_store, f"{cpg_id}_pred_groups.pkl")
                # open the file
                predictor_groups = pickle.load(open(predictor_groups_fn, 'rb'))
            except:
                predictor_groups = self.get_predictor_site_groups(
                    cpg_id, train_samples, num_correl_sites,
                    max_meqtl_sites, nearby_window_size
                    )
                pred_group_fn = os.path.join(self.output_dir, f"{cpg_id}_pred_groups.pkl")
                pickle.dump(predictor_groups, open(pred_group_fn, "wb"))
        if predictor_groups == {}:
            return []
        
        # create the training matrix
        X, y = self._create_training_mat(
            cpg_id, predictor_groups, train_samples, 
            aggregate = aggregate, binarize=binarize
            )
        # calculate the mutual information btwn each feature and the target methylation
        mi = self.mutual_info(X, covariate = y, bins=10)
        return mi
    
    def train_predictor(
        self, 
        cpg_id: str,
        train_samples: list,
        num_correl_sites: int,
        max_meqtl_sites: int,
        nearby_window_size: int,
        aggregate: str,
        binarize: bool = False,
        scramble: bool = False,
        feat_store: str = "",
        do_prediction: bool = False
        ):
        """
        Build the predictor for one CpG
        @ cpg_id: the id of the CpG to predict
        @ train_samples: list of samples to use for training
        @ num_correl_sites: number of sites to use as predictors
        @ max_meqtl_sites: maximum number of meqtl db sites to use as predictors
        @ nearby_window_size: window size to use for choosing nearby sites
        @ aggregate: whether to aggregate the mutation status by predictor group or not
        @ binarize: whether to binarize the mutation status or not
        @ scramble: whether to also train baseline models (same predictor sites but rows switched)
        """
        # get the sites to be used as predictors
        if feat_store == "": # not reading from saved pkl file
            # get the sites to be used as predictors
            predictor_groups = self.get_predictor_site_groups(
                cpg_id, train_samples, num_correl_sites,
                max_meqtl_sites, nearby_window_size
                )
            pred_group_fn = os.path.join(self.output_dir, f"{cpg_id}_pred_groups.pkl")
            pickle.dump(predictor_groups, open(pred_group_fn, "wb"))
        else: # read from saved pkl file if can
            try:
                predictor_groups_fn = os.path.join(feat_store, f"{cpg_id}_pred_groups.pkl")
                # open the file
                predictor_groups = pickle.load(open(predictor_groups_fn, 'rb'))
                predictor_groups = {k: predictor_groups[k] for k in ['100_matrixqtl_neg_beta', '100_matrixqtl_pos_beta', '5000_pos_corr', '5000_neg_corr',
                                                                     '100_pos_corr_ext', '100_neg_corr_ext', '25000_nearby']}
            except:
                predictor_groups = self.get_predictor_site_groups(
                    cpg_id, train_samples, num_correl_sites,
                    max_meqtl_sites, nearby_window_size
                    )
                pred_group_fn = os.path.join(self.output_dir, f"{cpg_id}_pred_groups.pkl")
                pickle.dump(predictor_groups, open(pred_group_fn, "wb"))
        if predictor_groups == {}:
            return []
        
        # create the training matrix
        X, y = self._create_training_mat(
            cpg_id, predictor_groups, train_samples, 
            aggregate = aggregate, binarize = binarize
            )
        # train one elasticNet model to predict y from X
        # model = ElasticNetCV(cv=5, random_state=0, max_iter=10000, selection = 'random', n_jobs=-1)
        model = LinearRegression()
        # model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
        model.fit(X, y)
        model_fn = os.path.join(self.output_dir, f"{cpg_id}_model.pkl")
        pickle.dump(model, open(model_fn, "wb"))
        if do_prediction:
            X, y = self._create_training_mat(
                cpg_id, predictor_groups, self.all_methyl_age_df_t.index, 
                aggregate = aggregate, binarize = binarize
                )
            y_pred = model.predict(X)
        # if scramble, randomly shuffle the mutation rows of X, but not covariate
        if scramble:
            # shuffle rows of the mutation columns
            non_coviariate_col_names = [
                col for  col in X.columns \
                if not col.startswith('dataset_') and not col.startswith('gender_')
                ]
            scrambled_X = X[non_coviariate_col_names].copy()
            scrambled_X = scrambled_X.sample(frac = 1, random_state = 0).reset_index(drop=True)
            # add the unshuffled covariate columns back
            coviariate_col_names = [
                col for  col in X.columns \
                if col.startswith('dataset_') or col.startswith('gender_')
                ]
            covariate_df = X.loc[:, coviariate_col_names]
            # concact ignoring the index
            scrambled_X.index = covariate_df.index
            scrambled_X = pd.concat([scrambled_X, covariate_df], axis = 1)
            # train again again but with scrambled X
            model.fit(scrambled_X, y)
            model_fn = os.path.join(self.output_dir, f"{cpg_id}_scrambled_model.pkl")
            pickle.dump(model, open(model_fn, "wb"))
            if do_prediction:
                X, y = self._create_training_mat(
                    cpg_id, predictor_groups, self.all_methyl_age_df_t.index, 
                    aggregate = aggregate, binarize = binarize
                    )
                y_pred_scrambled = model.predict(X)
        # combine predictions if do_prediction
        if do_prediction:
            if scramble:
                return pd.DataFrame({cpg_id: y_pred, cpg_id + "_scrambled": y_pred_scrambled}, index=self.all_methyl_age_df_t.index)
            else:
                return pd.DataFrame({cpg_id: y_pred}, index=self.all_methyl_age_df_t.index)
        
        
    def predict_cpg(
        self, 
        cpg_id: str,
        samples: list,
        model_fn: str,
        pred_group_fn: str,
        aggregate: str,
        binarize: bool
        ) -> pd.DataFrame:
        """
        Predict the methylation level of one CpG for a list of samples
        @ cpg_id: the id of the CpG to predict
        @ samples: list of samples to predict
        @ model_fn: the file name of the pickled model
        @ pred_group_fn: the file name of the pickled predictor groups
        @ aggregate: whether to aggregate the methylation levels of the predictor sites
        @ binarize: whether to binarize the methylation levels of the predictor sites
        """
        # get the predictor model and predictor sites
        model = pickle.load(open(model_fn, "rb"))
        pred_groups = pickle.load(open(pred_group_fn, "rb"))
        pred_groups = {k: pred_groups[k] for k in ['100_matrixqtl_neg_beta', '100_matrixqtl_pos_beta', '5000_pos_corr', '5000_neg_corr',
                                                    '100_pos_corr_ext', '100_neg_corr_ext', '25000_nearby']}
        # create the X matrix
        X, _ = self._create_training_mat(cpg_id, pred_groups, samples, aggregate, binarize)
        # predict the methylation levels
        preds = model.predict(X)
        # create a df with the samples as rows and the predicted methylation level as a column called cpg_id
        return pd.DataFrame({cpg_id: preds}, index=samples)

    def predict_all_cpgs(
        self,
        cpg_ids: list,
        samples: list,
        model_dir: str,
        aggregate: str = "False",
        binarize: bool = False,
        scrambled: bool = False
        ) -> pd.DataFrame:
        """
        Predict the methylation level of some CpGs for a list of samples using pretrained models
        @ cpg_ids: list of CpG ids to predict
        @ samples: list of samples to predict for
        @ model_dir: directory containing the pickled models and predictor group files
        @ aggregate: whether to aggregate the features or not (must be the same as when the models were trained)
        @ binarize: whether to binarize the features or not (must be the same as when the models were trained)
        @ scrambled: whether to use the baseline scrambled model or not
        @ return: a dataframe with the samples as rows and the predicted methylation levels as columns
        """
        # intersection with all_methyl_age_df_t index, so if a sample is a diff tissue type it will be removed
        samples = list(set(samples) & set(self.all_methyl_age_df_t.index.to_list()))
            
        predictions = []
        for i, cpg_id in enumerate(cpg_ids):
            # wether to use the baseline scrambled model or not
            if scrambled:
                model_fn = os.path.join(model_dir, f"{cpg_id}_scrambled_modelRF.pkl")
            else:
                model_fn = os.path.join(model_dir, f"{cpg_id}_modelRF.pkl")
            pred_group_fn = os.path.join(model_dir, f"{cpg_id}_pred_groups.pkl")
            predictions.append(
                self.predict_cpg(cpg_id, samples, model_fn, pred_group_fn, aggregate, binarize)
                )
            if i % 100 == 0:
                print(f"Finished {100*i/len(cpg_ids)}% of CpGs", flush=True)
        # concatenate the predictions into a single df
        pred_methyl_df = pd.concat(predictions, axis=1)
        return pred_methyl_df
    
    def choose_clock_cpgs(
        self, 
        pred_methyl: pd.DataFrame,
        actual_methyl: pd.DataFrame,
        mi_df: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Given predicted and actual values of training methylation for a set of CpGs, and the mutual information of each CpG with age across training samples, choose the best CpGs to use as predictors of age
        @ pred_methyl: a df with samples as rows and predicted CpG methylation levels as columns
        @ actual_methyl: a df with samples as rows and actual CpG methylation levels as columns
        @ mi_df: a df with CpGs as rows and mutual information with age the column
        @ return: a df of CpGs sorted by first their predicted methylation mutual information with age, then their correlation with actual methylation values, then MAE with actual methylation values, then their actual methylation mutual information with age
        """
        # get pairwise correlations between predicted and actual methylation
        corrs = pred_methyl.corrwith(actual_methyl, axis=0, method='pearson')
        # get pairwise MAEs between predicted and actual methylation
        maes = np.abs(pred_methyl - actual_methyl).mean(axis=0)
        # get MIs for these CpGs
        mi = mi_df.loc[pred_methyl.columns, "mutual_info"]
        # get MI of predicted methylation with age in training data
        pred_methyl_mi = self.mutual_info(
            X = pred_methyl, 
            covariate = self.all_methyl_age_df_t.loc[actual_methyl.index, 'age_at_index']
            )
        # get corr of predicted methylation with age in training data
        pred_methyl_corr = pred_methyl.corrwith(self.all_methyl_age_df_t.loc[actual_methyl.index, 'age_at_index'])
        # prioritize CpGs by correlation, then MI, then MAE
        cpg_priority_df = pd.DataFrame({"pred_mutual_info": pred_methyl_mi, "methyl_corr": corrs, "mae": maes, "mutual_info": mi, "age_corr": pred_methyl_corr})
        cpg_priority_df = cpg_priority_df.sort_values(
            by=["pred_mutual_info", "methyl_corr", "mutual_info", "mae", "age_corr"], ascending=[False, False, False, True, False]
            )
        return cpg_priority_df
    
    def train_epi_clock(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        out_fn: str,
        cpg_ids: list = []
        ):
        """
        Trains an epigenetic clock to predict chronological age from cpg methylation
        @ X: a df with samples as rows and cpgs as columns. Predicted methylation
        @ y: a series of chronological ages for the samples
        @ out_fn: the file to save the trained model to
        @ cpg_subset: a list of cpgs to use as predictors, e.g. accurately predictable or high MI. If empty, all cpgs are used
        @ return: the trained model
        """
        if len(cpg_ids) > 0:
            X = X[cpg_ids]
        # Create an ElasticNetCV object
        model = ElasticNetCV(
            cv=5, random_state=0, max_iter=10000,
            selection = 'random', n_jobs=-1, verbose=1
            )
        # Fit the model using cross-validation
        cv = KFold(n_splits=3, shuffle=True, random_state=0)
        maes, r2s, preds, tests = [], [], [], []
        # do the cross validation
        for train_index, test_index in cv.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            preds.append(pred)
            tests.append(y_test)
            maes.append(mean_absolute_error(y_test, pred))
            r2s.append(r2_score(y_test, pred))
        # create dataframe with  r2s, maes, preds, tests as columns
        results_df = pd.DataFrame({'r2': r2s, 'mae': maes, 'preds': preds, 'tests': tests})
        return results_df
    
    def visualize_clock_perf(
        self, 
        results_df: pd.DataFrame
        ) -> None:
        fig, axes = plt.subplots(1, len(results_df), figsize=(15, 4), sharex=True, dpi=100)
        for i in range(len(results_df)):
            to_plot = results_df.loc[i, 'tests'].to_frame().join(self.all_methyl_age_df_t.loc[:, 'dataset'])
            to_plot['preds'] = results_df.loc[i, 'preds']
            to_plot.columns = ['Actual age (years)', 'Dataset', 'Predicted age']
            sns.scatterplot(data=to_plot, x='Actual age (years)', y='Predicted age', hue='Dataset', ax=axes[i], legend=False)
            # axes[i].scatter(results_df.loc[i, 'tests'], results_df.loc[i, 'preds'], s=4, c=)
            # plot the identity line
            axes[i].plot([15, 95], [15, 95], color='red', linestyle='--')
            axes[i].set_xlabel('Actual age (years)')
            axes[i].set_ylabel('Predicted age (years)')
            # write r2 and mae in upper left corner
            axes[i].text(.01, .99, f"R2 = {results_df.loc[i, 'r2']:.3f}", ha='left', va='top',  transform=axes[i].transAxes )
            axes[i].text(.01, .9, f"MAE = {results_df.loc[i, 'mae']:.3f}", ha='left', va='top',  transform=axes[i].transAxes)
    
    def choose_cpgs_to_train(
        self,
        training_samples: list, 
        mi_df: pd.DataFrame,
        bin_size: int = 10000
        ) -> pd.DataFrame:
        """
        Based on count of mutations nearby and mutual information, choose cpgs to train models for
        @ training_samples: list of samples to train on
        @ mi_df: dataframe of mutual information
        @ bin_size: size of bins to count mutations in
        @ returns: cpg_pred_priority, dataframe of cpgs with priority for prediction
        """
        def mutation_bin_count(all_mut_w_age_df, training_samples):
            # count the number of mutations in each 10kb bin across all training samples
            mut_bin_counts_dfs = []
            for chrom in all_mut_w_age_df['chr'].unique():
                chr_df = all_mut_w_age_df.loc[(all_mut_w_age_df['chr'] == chrom) & (all_mut_w_age_df['case_submitter_id'].isin(training_samples))]
                counts, edges = np.histogram(chr_df['start'], bins = np.arange(0, chr_df['start'].max(), bin_size))
                one_mut_bin_counts_df = pd.DataFrame({'counts': counts, 'edges': edges[:-1]})
                one_mut_bin_counts_df['chr'] = chrom
                mut_bin_counts_dfs.append(one_mut_bin_counts_df)
            mut_bin_counts_df = pd.concat(mut_bin_counts_dfs, axis = 0)
            mut_bin_counts_df.reset_index(inplace=True, drop=True)
            return mut_bin_counts_df
        
        def round_down(num):
            """
            round N down to nearest bin_size
            """
            return num - (num % bin_size)
        
        # count mutations in each bin
        mutation_bin_counts_df = mutation_bin_count(self.all_mut_w_age_df, training_samples)
        # get count for each cpg
        illumina_cpg_locs_w_methyl_df = self.illumina_cpg_locs_df.loc[self.illumina_cpg_locs_df['#id'].isin(self.all_methyl_age_df_t.columns)]
        illumina_cpg_locs_w_methyl_df.loc[:,'rounded_start'] = illumina_cpg_locs_w_methyl_df.loc[:, 'start'].apply(round_down)
        cpg_pred_priority = illumina_cpg_locs_w_methyl_df.merge(mutation_bin_counts_df, left_on=['chr', 'rounded_start'], right_on=['chr', 'edges'], how='left')
        # get mi for each cpg
        cpg_pred_priority = cpg_pred_priority.merge(mi_df, left_on='#id', right_index=True, how='left')
        # sort by count and mi
        cpg_pred_priority.sort_values(by=['counts', 'mutual_info'], ascending=[False, False], inplace=True)
        # reset index
        cpg_pred_priority.reset_index(inplace=True, drop=True)
        return cpg_pred_priority
    
    def driver(
        self, 
        do: str,
        num_correl_sites: int,
        max_meqtl_sites: int,
        nearby_window_size: int,
        cpg_ids: list = [],
        train_samples: list = [],
        aggregate: str = "False",
        binarize: bool = False,
        feat_store: str = "",
        scramble: bool = False,
        do_prediction: bool = False
        ):
        """
        Train the predictor for all CpGs
        """
        # get the list of all CpGs
        if len(cpg_ids) == 0:
            cpg_ids = self.illumina_cpg_locs_df['#id'].to_list()
        # get the list samples to train with
        if len(train_samples) == 0:
            train_samples = self.all_methyl_age_df_t.index.to_list()
        else: # intersection with all_methyl_age_df_t index, so if a training sample is a diff tissue type it will be removed
            train_samples = list(set(train_samples) & set(self.all_methyl_age_df_t.index.to_list()))
        # do one of 3 options
        if do == 'train':
            predicted_methyl = []
            # for each cpg, train the predictor and save trained model
            for i, cpg_id in enumerate(cpg_ids):
                preds = self.train_predictor(
                    cpg_id, train_samples, num_correl_sites, max_meqtl_sites,
                    nearby_window_size, aggregate, binarize, scramble, feat_store,
                    do_prediction
                    )
                if do_prediction:
                    predicted_methyl.append(preds)
                if i % 10 == 0:
                    print(f"Finished {100*i/len(cpg_ids)}% of CpGs", flush=True)
            if do_prediction:
                predicted_methyl_df = pd.concat(predicted_methyl, axis=1)
                predicted_methyl_df.to_parquet(os.path.join(self.output_dir, f'predicted_methyl_{num_correl_sites}correl_{max_meqtl_sites}matrixQtl_{nearby_window_size}nearby_{aggregate}Aggregate_{binarize}binarize_{scramble}Scrambled_best_mi_linreg.parquet'))
        elif do == 'evaluate': # evaluate
            result_dfs = []
            for i, cpg_id in enumerate(cpg_ids):
                result_df = self.evaluate_predictor(
                    cpg_id, train_samples, num_correl_sites, max_meqtl_sites,
                    nearby_window_size, aggregate, binarize, feat_store, scramble
                    )
                # check if result_df is empty
                if len(result_df) != 0:
                    result_dfs.append(result_df)
                if i % 10 == 0:
                    print(f"Finished {100*i/len(cpg_ids)}% of CpGs", flush=True)
            result_df = pd.concat(result_dfs)
            return result_df
        elif do == 'eval_features':
            feat_info = {}
            for i, cpg_id in enumerate(cpg_ids):
                feat_info[cpg_id] = self.feature_informations(
                    cpg_id, train_samples, num_correl_sites, max_meqtl_sites,
                    nearby_window_size, aggregate, binarize, feat_store
                    )
                if i % 10 == 0:
                    print(f"Finished {(i*100)/len(cpg_ids)}% of CpGs", flush=True)
            # remove any elements that are length 0
            feat_info = {k: v for k, v in feat_info.items() if len(v) > 0}
            feat_info_df = pd.DataFrame(data = feat_info)
            return feat_info_df
        else:
            print("'do' must be one of 'train', 'evaluate', 'eval_features'")
            sys.exit(1)
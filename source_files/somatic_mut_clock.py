import pandas as pd
import numpy as np
import os
import sys
#from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import time
import seaborn as sns

from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNetCV, RidgeCV, LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb


class mutationClock:
    """
    Train epigenetic clocks on predicted methylation and actual methylation, comparing
    """
    def __init__(
        self,
        predicted_methyl_fns: list,
        predicted_perf_fns: list,
        all_methyl_age_df_t: pd.DataFrame,
        illumina_cpg_locs_df: pd.DataFrame, 
        output_dir: str,
        train_samples: list,
        test_samples: list,
        tissue_type: str = "",
        scrambled_predicted_methyl_fns: list =[],
        trained_models_fns: list = [],
        feature_mat_fns: list = []
        ) -> None:
        """
        @ predicted_methyl_fns: a list of paths to the predicted methylation files
        @ predicted_perf_fns: a list of paths to the performance files
        @ scrambled_predicted_methyl_fns: a list of paths to the scrambled predicted methylation files
        @ all_methyl_age_df_t: a dataframe of all the methylation data, with age as the index
        @ illumina_cpg_locs_df: a dataframe of the locations of the CpGs in the methylation data
        @ output_dir: the path to the output directory where the results will be saved
        @ train_samples: a list of the training samples, from mut_feat
        @ test_samples: a list of the testing samples, from mut_feat
        @ tissue_type: the tissue type to use for the analysis
        """
        self.predicted_methyl_df = self._combine_fns(predicted_methyl_fns, axis = 1)
        self.performance_df = self._combine_fns(predicted_perf_fns, axis=0)
        if len(scrambled_predicted_methyl_fns) > 0:
            self.scrambled_predicted_methyl_df = self._combine_fns(scrambled_predicted_methyl_fns, axis = 1)
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.output_dir = output_dir
        self.tissue_type = tissue_type
        # if tissue type is specified, subset the data to only this tissue type
        if self.tissue_type != "":
            if self.tissue_type == 'RCC':
                RCC_datasets = ['KIRC', 'KIRP' , 'KICH']
                self.all_methyl_age_df_t = self.all_methyl_age_df_t.loc[
                    self.all_methyl_age_df_t['dataset'].isin(RCC_datasets), :]
                self.all_methyl_age_df_t['dataset'] = 'RCC'
            else:
                self.all_methyl_age_df_t = self.all_methyl_age_df_t.loc[self.all_methyl_age_df_t['dataset'] == tissue_type, :]
        self.test_samples = test_samples
        self.train_samples = train_samples
        self.trained_models = {}
        if len(trained_models_fns) > 0:
            for fn in trained_models_fns:
                # read in dictionary from pickle file
                with open(fn, 'rb') as f:
                    these_models = pickle.load(f)
                    # add to dictionary
                    self.trained_models.update(these_models)
        self.feature_mats = {}
        if len(feature_mat_fns) > 0:
            first = True
            for fn in feature_mat_fns:
                # read in dictionary from pickle file
                with open(fn, 'rb') as f:
                    this_feat_dict = pickle.load(f)
                    if first:
                        self.feature_mats.update(this_feat_dict)
                        first = False
                    else:
                        minor_dicts_to_update = ['feat_mats', 'target_values']
                        for minor_dict in minor_dicts_to_update:
                            self.feature_mats[minor_dict].update(this_feat_dict[minor_dict])
                            
        
    def _populate_performance(self):
        """
        Add interesting column to self.performance_df
        """
        # add training CpG real age correlation
        self.performance_df['training_real_mf_age_pearsonr'] = self.all_methyl_age_df_t.loc[self.train_samples, self.performance_df.index ].corrwith(self.all_methyl_age_df_t.loc[self.train_samples, 'age_at_index'])
        # add training CpG predicted age correlation
        self.performance_df['training_pred_mf_age_pearsonr'] = self.predicted_methyl_df.loc[self.train_samples, :].corrwith(self.all_methyl_age_df_t.loc[self.train_samples, 'age_at_index'])
        self.performance_df['training_real_mf_age_spearmanr'] = self.all_methyl_age_df_t.loc[self.train_samples, self.performance_df.index ].corrwith(self.all_methyl_age_df_t.loc[self.train_samples, 'age_at_index'], method='spearman')
        # add training CpG predicted age correlation
        self.performance_df['training_pred_mf_age_spearmanr'] = self.predicted_methyl_df.loc[self.train_samples, :].corrwith(self.all_methyl_age_df_t.loc[self.train_samples, 'age_at_index'], method='spearman')
        # add correlation between training predicted methylation and training actual methylation
        self.performance_df['training_pred_real_mf_r'] = self.predicted_methyl_df.loc[self.train_samples, :].corrwith(self.all_methyl_age_df_t.loc[self.train_samples, self.performance_df.index])
        # same but spearman
        self.performance_df['training_pred_real_mf_spearmanr'] = self.predicted_methyl_df.loc[self.train_samples, :].corrwith(self.all_methyl_age_df_t.loc[self.train_samples, self.performance_df.index], method='spearman')
        # get correlation between testing scrambled predicted methylation and testing actual methylation
        #self.performance_df['testing_scrambled_real_mf_r'] = self.scrambled_predicted_methyl_df.loc[self.test_samples, :].corrwith(self.all_methyl_age_df_t.loc[self.test_samples, self.performance_df.index])
        # add correlation of predicted testing samples with age
        self.performance_df['testing_pred_mf_age_pearsonr'] = self.predicted_methyl_df.loc[self.test_samples, :].corrwith(self.all_methyl_age_df_t.loc[self.test_samples, 'age_at_index'])
        # and actual testing samples with age
        self.performance_df['testing_real_mf_age_pearsonr'] = self.all_methyl_age_df_t.loc[self.test_samples, self.performance_df.index].corrwith(self.all_methyl_age_df_t.loc[self.test_samples, 'age_at_index'])
        self.performance_df['testing_pred_mf_age_spearmanr'] = self.predicted_methyl_df.loc[self.test_samples, :].corrwith(self.all_methyl_age_df_t.loc[self.test_samples, 'age_at_index'], method='spearman')
        # and actual testing samples with age
        self.performance_df['testing_real_mf_age_spearmanr'] = self.all_methyl_age_df_t.loc[self.test_samples, self.performance_df.index].corrwith(self.all_methyl_age_df_t.loc[self.test_samples, 'age_at_index'], method='spearman')
    
    
    def _combine_fns(
        self,
        fns: list,
        axis: int
        ) -> pd.DataFrame:
        """
        Combine the data from multiple files into one dataframe
        @ fns: a list of paths to the predicted methylation files or performance files
        @ axis: the axis to concatenate the dataframes on
        """
        all_dfs = []
        for fn in fns:
            df = pd.read_parquet(fn)
            all_dfs.append(df)    
        combined_df = pd.concat(all_dfs, axis=axis)
        # drop duplicate columns if they exist, may happen when running many jobs
        if axis == 1:
            combined_df = combined_df.loc[:,~combined_df.columns.duplicated()].copy()
        else: # if axis = 0 
            combined_df = combined_df.loc[~combined_df.index.duplicated(), :].copy()

        return combined_df
 
    def feature_selection(self):
        """
        Perform feature selection on the predicted methylation data of the training samples
        """
        # get correlation of each CpG with age
        training_age_corr = self.predicted_methyl_df.loc[self.train_samples].corrwith(self.all_methyl_age_df_t.loc[self.train_samples, 'age_at_index'], axis = 0)
        # get correlation between each cpg's predicted methylation and actual methylation in all_methyl_age_df_t for training samples
        #training_methyl_corr = self.predicted_methyl_df.loc[self.train_samples].corrwith(self.all_methyl_age_df_t.loc[self.train_samples, self.predicted_methyl_df.columns], axis = 0)
        return training_age_corr
 
    def train_epi_clock(
        self,
        X, 
        y
        ) -> None:
        """
        Trains an epigenetic clock to predict chronological age from cpg methylation
        @ X: a df with samples as rows and cpgs as columns. Predicted methylation
        @ y: a series of chronological ages for the samples
        @ return: the trained model
        """
        # X = self.all_methyl_age_df_t.loc[samples, cpgs]
        #y = self.all_methyl_age_df_t.loc[samples, 'age_at_index']
        #X = self.predicted_methyl_df.loc[samples, cpgs]
        #model = xgb.XGBRegressor()
        # model = RandomForestRegressor(n_estimators=1000, max_depth=100, n_jobs=-1, verbose=1)
        
        # Create an ElasticNetCV object
        model = ElasticNetCV(
            cv=5, random_state=0, max_iter=10000,
            selection = 'random', n_jobs=-1, verbose=0
            )
        model.fit(X, y)
        return model
        # write the model to a .pkl file in output_dir
        #out_fn = os.path.join(self.output_dir, f"{self.tissue_type}_{len(cpgs)}numCpgsMostAgeCorr_trained_epiClock.pkl")
        """with open(out_fn, 'wb') as f:
            pickle.dump(model, f)"""
        
   
    def apply_epi_clock(
        self,
        X: pd.DataFrame,
        model: object
        ) -> pd.Series:
        """
        Apply a trained epigenetic clock to predict chronological age from cpg methylation
        @ X: a df with samples as rows and cpgs as columns to use for prediction
        @ returns: a series of predicted ages
        """
        return model.predict(X)
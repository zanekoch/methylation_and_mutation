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
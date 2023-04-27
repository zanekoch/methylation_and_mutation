import pandas as pd
import numpy as np
import os
import sys
#from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNetCV, RidgeCV, LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import glob



class optimizeSomage:
    """
    Take in multiple mutationClock objects and compare them to find the best model
    and hyperparameters
    """
    def __init__(
        self,
        results_dir: str, 
        model_dirs: list,
        all_methyl_age_df_t: pd.DataFrame,
        illumina_cpg_locs_df: pd.DataFrame,
        train_samples: list,
        test_samples: list,
        tissue_type: str,
        top_dataset_n: int = 20,
        ):
        """
        @ results_dir: the directory where the trained models are saved
        @ model_dirs: list of the sub-directories where the trained models are saved
        @ all_methyl_age_df_t: the dataframe of all the methylation ages
        @ illumina_cpg_locs_df: the dataframe of the locations of the CpGs
        @ train_samples: the list of training samples (from mut_feat)
        @ test_samples: the list of testing samples (from mut_feat)
        @ tissue_type: the tissue type to use for the analysis
        @ top_dataset_n: the number of datasets to use for the analysis, by sample num
        """
        self.somages = []
        # iterate through the model directories and create mutationClock objects
        for model_dir in model_dirs:
            predicted_methyl_fns =  glob.glob(
                os.path.join(
                    results_dir,
                    model_dir,
                    "methyl_predictions_xgboost_Falsescramble.parquet")
                )
            predicted_perf_fns = glob.glob(
                os.path.join(
                    results_dir,
                    model_dir, 
                    "prediction_performance_xgboost_Falsescramble.parquet")
                )
            trained_models_fns = glob.glob(
                os.path.join(
                    results_dir,
                    model_dir, 
                    "trained_models_xgboost_Falsescramble.pkl")
                )
            feature_mat_fns = glob.glob(
                os.path.join(
                    results_dir,
                    model_dir,
                    "*features.pkl")
                )
            somage = mutationClock(
                predicted_methyl_fns = predicted_methyl_fns, 
                predicted_perf_fns = predicted_perf_fns,
                all_methyl_age_df_t = all_methyl_age_df_t,
                illumina_cpg_locs_df = illumina_cpg_locs_df,
                output_dir = "",
                train_samples = train_samples,
                test_samples = test_samples,
                tissue_type = tissue_type,
                trained_models_fns = trained_models_fns,
                feature_mat_fns = feature_mat_fns
                )
            somage.populate_performance()
            somage.performance_by_dataset()
            self.somages.append(somage)
            print("Finished loading soMage object for {}".format(model_dir))
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.top_datasets = self.all_methyl_age_df_t['dataset'].value_counts().index[:top_dataset_n]

    def top_corr_by_dataset(
        self,
        top_cpg_num: int,
        metric: str
        ) -> list:
        """
        For each somage object, select the top top_cpg_num cpgs, based on metric, within each dataset and get the mean value of the metric across this CpGs for each dataset
        @ returns: a df of the mean metric values within each dataset for each somage object, columns indexed in same order as self.somages
        """
        mean_metrics = []
        for somage in self.somages:
            # get mean metric for the highest top_cpg_num cpgs within each dataset
            mean_metric_by_dset = somage.performance_by_dataset_df.groupby("dataset").apply(
                lambda x: x.sort_values(
                    metric, ascending = False
                    ).head(top_cpg_num)[metric].mean(axis = 0)
                )
            mean_metrics.append(mean_metric_by_dset)
        mean_metrics_df = pd.concat(mean_metrics, axis = 1)
        return mean_metrics_df


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
                        minor_dicts_to_update = ['feat_mats', 'target_values', 'feat_names']
                        for minor_dict in minor_dicts_to_update:
                            self.feature_mats[minor_dict].update(this_feat_dict[minor_dict])
                            
    def performance_by_dataset(self):
        """
        Get the performance of the models by dataset
        """
        top_20_datasets = self.all_methyl_age_df_t['dataset'].value_counts().index[:20].to_list()
        # remove certain vals from top_20_datasets
        for dset in top_20_datasets:
            if dset == 'BRCA' or dset == 'LGG' or dset == 'THCA' or dset == 'LAML' or dset == 'KIRC' or dset == 'KIRP' or dset == 'SARC' or dset =='THCA' or dset =='LIHC' or dset =='LUSC' or dset =='CESC' or dset == 'SARC':
                top_20_datasets.remove(dset) 
        print(top_20_datasets)
        dataset_perf_dfs = []
        for dataset in top_20_datasets:
            # get the correlation between actual testing sample methylation
            # and predicted testing sample methylation from this dataset
            this_dataset_samples = self.all_methyl_age_df_t.loc[
                self.all_methyl_age_df_t['dataset'] == dataset, 
                :].index
            this_dataset_test_samples = list(
                set(this_dataset_samples).intersection(set(self.test_samples))
                )
            real_methyl_df = self.all_methyl_age_df_t.loc[
                this_dataset_test_samples, 
                self.predicted_methyl_df.columns
                ]
            pred_methyl_df = self.predicted_methyl_df.loc[
                this_dataset_test_samples, 
                :]
            # also get training samples
            this_dataset_train_samples = list(
                set(this_dataset_samples).intersection(set(self.train_samples))
                )
            real_methyl_df_train = self.all_methyl_age_df_t.loc[
                this_dataset_train_samples, 
                self.predicted_methyl_df.columns
                ]
            pred_methyl_df_train = self.predicted_methyl_df.loc[
                this_dataset_train_samples, 
                :]
            
            # get the correlation and mutual informaiton
            dataset_pearson = real_methyl_df.corrwith(pred_methyl_df, method = 'pearson')
            dataset_spearman = real_methyl_df.corrwith(pred_methyl_df, method = 'spearman')
            dataset_mae = np.mean(np.abs(real_methyl_df - pred_methyl_df), axis = 0)
            # get mutual informaiton using sklearn
            try:
                dataset_mi = mutual_info_score(real_methyl_df, pred_methyl_df)
            except:
                dataset_mi = np.nan
            try:
                train_dataset_mi = mutual_info_score(real_methyl_df_train, pred_methyl_df_train)
            except:
                train_dataset_mi = np.nan
                
            train_dataset_spearman = real_methyl_df_train.corrwith(
                pred_methyl_df_train, method = 'spearman'
                )   
            # also get correlation between testing sample methylation and age
            this_dataset_test_age_df = self.all_methyl_age_df_t.loc[
                this_dataset_test_samples, 
                'age_at_index'
                ]
            this_dataset_train_age_df = self.all_methyl_age_df_t.loc[
                this_dataset_train_samples, 
                'age_at_index'
                ]
            dataset_age_pearson = pred_methyl_df.corrwith(this_dataset_test_age_df, method = 'pearson').abs()
            dataset_age_spearman = pred_methyl_df.corrwith(this_dataset_test_age_df, method = 'spearman').abs()
            train_dataset_age_spearman = pred_methyl_df_train.corrwith(
                this_dataset_train_age_df, method = 'spearman'
                ).abs()
            
            # create dataframe
            dataset_perf_df = pd.DataFrame({
                'AvP_methyl_pearson': dataset_pearson,
                'AvP_methyl_spearman': dataset_spearman,
                'AvP_methyl_mi': dataset_mi,
                'train_AvP_methyl_mi': train_dataset_mi,
                'train_AvP_methyl_spearman': train_dataset_spearman,
                'AvP_methyl_mae': dataset_mae,
                'abs_Pmethyl_v_Age_pearson': dataset_age_pearson,
                'abs_Pmethyl_v_Age_spearman': dataset_age_spearman,
                'train_abs_Pmethyl_v_Age_spearman': train_dataset_age_spearman,

                }, index = self.predicted_methyl_df.columns)
            dataset_perf_df['dataset'] = dataset
            dataset_perf_dfs.append(dataset_perf_df)
            print("done with dataset: " + dataset, flush = True)
        all_dataset_perf_df = pd.concat(dataset_perf_dfs)
        # make cpg a column
        all_dataset_perf_df.reset_index(inplace = True)
        all_dataset_perf_df.rename(columns = {'index':'cpg'}, inplace = True)
        self.performance_by_dataset_df = all_dataset_perf_df
    
    def populate_performance(self):
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
    
    def plot_real_vs_predicted_methylation(
        self, 
        cpg: str, 
        dataset: str = "", 
        sample_set: str = "test"
        ):
        sns.set_context("notebook", font_scale=1.1)
        if sample_set == "test":
            samples = self.test_samples
        elif sample_set == "train":
            samples = self.train_samples
        else: # both
            samples = self.test_samples + self.train_samples
        
        if dataset == "":
            predicted_values = self.predicted_methyl_df.loc[samples, cpg]
            actual_values = self.all_methyl_age_df_t.loc[samples, cpg]
        else:
            # subset samples to only those in the dataset
            dset_samples = self.all_methyl_age_df_t.loc[self.all_methyl_age_df_t['dataset'] == dataset, :].index
            samples = list(set(samples) & set(dset_samples))
            predicted_values = self.predicted_methyl_df.loc[samples, cpg]
            actual_values = self.all_methyl_age_df_t.loc[samples, cpg]
        # plot scatterplot of predicted vs actual
        fig, axes = plt.subplots(figsize=(6, 4))
        fig2, axes2 = plt.subplots(figsize=(6, 4))
        if dataset != "":
            sns.scatterplot(
                y=predicted_values, x=actual_values,
                ax=axes, hue = self.all_methyl_age_df_t.loc[samples, 'age_at_index']
                )
            pred_act_df = pd.DataFrame({
                'Methylation fraction': predicted_values.to_list() + actual_values.to_list(), 
                'Age': self.all_methyl_age_df_t.loc[samples, 'age_at_index'].to_list()* 2, 
                'Type': ['Predicted']*len(predicted_values) + ['Actual']*len(actual_values)
                })
            sns.scatterplot(
                y='Methylation fraction', x='Age',
                ax=axes2, hue = 'Type', data=pred_act_df,
            )
        else:
            sns.scatterplot(
                y=predicted_values, x=actual_values,
                ax=axes, hue = self.all_methyl_age_df_t.loc[samples, 'dataset']
                )
            pred_act_df = pd.DataFrame({
                'Methylation fraction': predicted_values.to_list() + actual_values.to_list(), 
                'Age': self.all_methyl_age_df_t.loc[samples, 'age_at_index'].to_list()* 2, 
                'Type': ['Predicted']*len(predicted_values) + ['Actual']*len(actual_values)
                })
            sns.scatterplot(
                y='Methylation fraction', x='Age',
                ax=axes2, hue = 'Type', data=pred_act_df,
            )
        axes.set_ylabel(f'Predicted Methylation {cpg}')
        axes.set_xlabel(f'Actual Methylation {cpg}')
        # change legend title
        axes.legend(title='Age')
        # y = x line based on min and max values
        min_val = min(min(predicted_values), min(actual_values))
        max_val = max(max(predicted_values), max(actual_values))
        axes.plot([min_val, max_val], [min_val, max_val], color='black')
    
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
        y,
        model_type: str = 'elasticnet'
        ) -> None:
        """
        Trains an epigenetic clock to predict chronological age from cpg methylation
        @ X: a df with samples as rows and cpgs as columns. Predicted methylation
        @ y: a series of chronological ages for the samples
        @ return: the trained model
        """
        from sklearn.model_selection import RandomizedSearchCV
        if model_type == 'elasticnet':
            # Create an ElasticNetCV object
            model = ElasticNetCV(
                cv=5, random_state=0, max_iter=10000,
                selection = 'random', n_jobs=-1, verbose=0
                )
            model.fit(X, y)
        elif model_type == 'xgboost':
            """# Create a parameter grid for the XGBoost model
            param_grid = {
                'learning_rate': np.logspace(-4, 0, 50),
                'n_estimators': range(100, 500, 100),
                'max_depth': range(3, 10),
                'min_child_weight': range(1, 6),
                'gamma': np.linspace(0, 0.5, 50),
                'subsample': np.linspace(0.5, 1, 50),
                'colsample_bytree': np.linspace(0.5, 1, 50),
                'reg_alpha': np.logspace(-4, 0, 50),
                'reg_lambda': np.logspace(-4, 0, 50)
            }
            # Create the XGBRegressor model
            model = xgb.XGBRegressor()
            # Initialize the RandomizedSearchCV object
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=100,  # number of parameter settings that are sampled
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                cv=5,
                verbose=0,
                random_state=42
            )
            # Fit the RandomizedSearchCV object to the training data
            random_search.fit(X, y)
            # Print the best hyperparameters
            print("Best hyperparameters:", random_search.best_params_)
            # Use the best estimator for predictions or further analysis
            model = random_search.best_estimator_ """
            model = xgb.XGBRegressor()
            model.fit(X, y)
        else:
            raise ValueError("model_type must be 'elasticnet' or 'xgboost'")
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
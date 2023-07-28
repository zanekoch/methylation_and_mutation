import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVR
import xgboost as xgb
import glob
from scipy.stats import spearmanr, pearsonr
import shap
from tqdm import tqdm
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42




class mutationClock:
    """
    Train epigenetic clocks on predicted methylation and actual methylation
    """
    def __init__(
        self,
        predicted_methyl_fns: list,
        all_methyl_age_df_t: pd.DataFrame,
        illumina_cpg_locs_df: pd.DataFrame, 
        output_dir: str,
        train_samples: list,
        test_samples: list,
        validation_samples: list,
        tissue_type: str = "",
        trained_models_fns: list = [],
        feature_mat_fns: list = [],
        performance_by_dataset_fns: list = []
        ) -> None:
        """
        @ predicted_methyl_fns: a list of paths to the predicted methylation files
        @ scrambled_predicted_methyl_fns: a list of paths to the scrambled predicted methylation files
        @ all_methyl_age_df_t: a dataframe of all the methylation data, with age as the index
        @ illumina_cpg_locs_df: a dataframe of the locations of the CpGs in the methylation data
        @ output_dir: the path to the output directory where the results will be saved
        @ train_samples: a list of the training samples, from mut_feat
        @ test_samples: a list of the testing samples, from mut_feat
        @ validation_samples: a list of the validation samples, from mut_feat
        @ tissue_type: the tissue type to use for the analysis
        @ trained_models_fns: a list of paths to the trained models
        @ feature_mat_fns: a list of paths to the feature matrices
        @ performance_by_dataset_fns: a list of paths to the performance by dataset files
        """
        self.predicted_methyl_df = self._combine_fns(predicted_methyl_fns, axis = 1)
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.output_dir = output_dir
        self.tissue_type = tissue_type
        # if tissue type is specified, subset the data to only this tissue type
        if self.tissue_type != "":
            self.all_methyl_age_df_t = self.all_methyl_age_df_t.loc[
                self.all_methyl_age_df_t['dataset'] == tissue_type,
                :]
        self.test_samples = test_samples
        self.train_samples = train_samples
        self.validation_samples = validation_samples
        # if there are trained models, read them in and combine them
        self.trained_models = {}
        if len(trained_models_fns) > 0:
            """for fn in tqdm(trained_models_fns, desc="Loading trained model fns"):
                # read in dictionary from pickle file
                with open(fn, 'rb') as f:
                    these_models = pickle.load(f)
                    # add to dictionary
                    self.trained_models.update(these_models)
                # read in using xgb method
                model = xgb.XGBRegressor()
                model.load_model(fn)
                # get the cpg name from the model name
                cpg_name = os.path.basename(fn).split('.')[0]
                self.trained_models[cpg_name] = model
                print("loaded model for " + cpg_name, flush = True)"""
        # if there are feature matrices, read them in and combine them
        self.feature_mats = {}
        if len(feature_mat_fns) > 0:
            first = True
            for fn in tqdm(feature_mat_fns, desc="Loading feature matrices"):
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
                        minor_lists_to_update = ['cpg_ids']
                        for minor_list in minor_lists_to_update:
                            self.feature_mats[minor_list] += this_feat_dict[minor_list]
        # if there are performance_by_dataset files, read them in and combine them
        performance_by_dataset_l = []
        if len(performance_by_dataset_fns) > 0:
            for fn in tqdm(performance_by_dataset_fns, desc="Loading performance data"):
                # read in dataframe from parquet file
                this_df = pd.read_parquet(fn)
                performance_by_dataset_l.append(this_df)
            self.performance_by_dataset_df = pd.concat(performance_by_dataset_l)
    
    @classmethod
    def construct_from_paths(
        cls,
        somage_path, 
        directory_glob, 
        file_suffix, 
        mut_feat, 
        illumina_cpg_locs_df, 
        all_methyl_age_df_t,
        out_dir
        ):
        """
        Given a somage path, directory glob, and file suffix, create a somage object from all the files in the directory
        """
        print("Creating soMage object", flush = True)
        predicted_methyl_fns = glob.glob(
            os.path.join(somage_path, directory_glob, f"methyl_predictions_{file_suffix}.parquet")
            )
        trained_model_fns = glob.glob(
            os.path.join(somage_path, directory_glob, f"trained_models_{file_suffix}.pkl")
            )
        feature_mat_fns = glob.glob(
            os.path.join(somage_path, directory_glob, "*features.pkl")
            )
        performance_by_dataset_fns = glob.glob(
            os.path.join(somage_path, directory_glob, f"performance_by_dataset_{file_suffix}.parquet")
            )
        somage = cls(
                predicted_methyl_fns = predicted_methyl_fns, 
                all_methyl_age_df_t = all_methyl_age_df_t,
                illumina_cpg_locs_df = illumina_cpg_locs_df,
                output_dir = out_dir,
                train_samples = mut_feat.train_samples,
                test_samples = mut_feat.test_samples,
                validation_samples = mut_feat.validation_samples,
                tissue_type = "",
                trained_models_fns = trained_model_fns,
                feature_mat_fns = feature_mat_fns,
                performance_by_dataset_fns = performance_by_dataset_fns
                )
        return somage           
                            
    def performance_by_dataset(self, predicted_with_random_feat = False):
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
        if predicted_with_random_feat:
            # convert the predicted methyl df columns to the target cpg names but splitting each column name on the underscore and taking the first element
            target_cpgs = [col.split('_')[1] for col in self.predicted_methyl_df.columns]
            target_train_names = self.predicted_methyl_df.columns
            pred_for_corr_df = self.predicted_methyl_df.copy(deep = True)
            pred_for_corr_df.columns = target_cpgs
        else:
            target_cpgs = self.predicted_methyl_df.columns
            target_train_names = self.predicted_methyl_df.columns
            pred_for_corr_df = self.predicted_methyl_df
        
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
            this_dataset_test_samples = list(
                set(this_dataset_samples).intersection(set(self.test_samples))
                )
            
            real_methyl_df = self.all_methyl_age_df_t.loc[
                this_dataset_test_samples, 
                target_cpgs
                ]
            pred_methyl_df = pred_for_corr_df.loc[
                this_dataset_test_samples, 
                :]
            
            real_methyl_df_train = self.all_methyl_age_df_t.loc[
                this_dataset_train_samples, 
                target_cpgs
                ]
            pred_methyl_df_train = pred_for_corr_df.loc[
                this_dataset_train_samples, 
                :]
            # get the correlation and mutual informaiton
            dataset_pearson = real_methyl_df.corrwith(pred_methyl_df, method = 'pearson')
            this_dataset_test_age_df = self.all_methyl_age_df_t.loc[
                this_dataset_test_samples, 
                'age_at_index'
                ]
            dataset_age_pearson = pred_methyl_df.corrwith(this_dataset_test_age_df, method = 'pearson').abs()
            # same for training samples
            train_dataset_pearson = real_methyl_df_train.corrwith(
                pred_methyl_df_train, method = 'pearson'
                ) 
            this_dataset_train_age_df = self.all_methyl_age_df_t.loc[
                this_dataset_train_samples, 
                'age_at_index'
                ]            
            train_actual_methyl_age_pearson = real_methyl_df_train.corrwith(
                this_dataset_train_age_df, method = 'pearson'
                ).abs()
            """dataset_spearman = real_methyl_df.corrwith(pred_methyl_df, method = 'spearman')
            dataset_mae = np.mean(np.abs(real_methyl_df - pred_methyl_df), axis = 0)
            real_methyl_df_rounded = np.round(real_methyl_df)
            pred_methyl_df_rounded = np.round(pred_methyl_df)
            dataset_mi = real_methyl_df_rounded.apply(
                lambda col: mutual_info_score(col, pred_methyl_df_rounded[col.name]), axis=0
                )
            
            dataset_age_spearman = pred_methyl_df.corrwith(this_dataset_test_age_df, method = 'spearman').abs()
            methyl_age_mi = pred_methyl_df_rounded.apply(
                lambda col: mutual_info_score(col, this_dataset_test_age_df), axis=0
                )
            
            train_dataset_spearman = real_methyl_df_train.corrwith(
                pred_methyl_df_train, method = 'spearman'
                )
            real_methyl_df_train_rounded = np.round(real_methyl_df_train)
            pred_methyl_df_train_rounded = np.round(pred_methyl_df_train)
            train_dataset_mi = real_methyl_df_train_rounded.apply(
                lambda col: mutual_info_score(col, pred_methyl_df_train_rounded[col.name]), axis=0
                )  
            
            train_dataset_age_spearman = pred_methyl_df_train.corrwith(
                this_dataset_train_age_df, method = 'spearman'
                ).abs()
            train_methyl_age_mi = pred_methyl_df_train_rounded.apply(
                lambda col: mutual_info_score(col, this_dataset_train_age_df), axis=0
                )
            
            train_actual_methyl_age_spearman = real_methyl_df_train.corrwith(
                this_dataset_train_age_df, method = 'spearman'
                ).abs()
            train_actual_methyl_age_mi = real_methyl_df_train_rounded.apply(
                lambda col: mutual_info_score(col, this_dataset_train_age_df), axis=0
                )"""
            # create dataframe
            dataset_perf_df = pd.DataFrame({
                'AvP_methyl_pearson': dataset_pearson,
                'Pmethyl_v_Age_pearson_abs': dataset_age_pearson,
                'train_AvP_methyl_pearson': train_dataset_pearson,
                'train_Amethyl_v_Age_pearson_abs': train_actual_methyl_age_pearson,
                })#, index = self.predicted_methyl_df.columns)
            """'AvP_methyl_spearman': dataset_spearman,
                'AvP_methyl_mi': dataset_mi,
                'AvP_methyl_mae': dataset_mae,
                'Pmethyl_v_Age_pearson_abs': dataset_age_pearson,
                'Pmethyl_v_Age_spearman_abs': dataset_age_spearman,
                'Pmethyl_v_Age_mi': methyl_age_mi,
                # train
                'train_AvP_methyl_pearson': train_dataset_pearson,
                'train_AvP_methyl_spearman': train_dataset_spearman,
                'train_AvP_methyl_mi': train_dataset_mi,
                'train_Pmethyl_v_Age_spearman_abs': train_dataset_age_spearman,
                'train_Pmethyl_v_Age_mi': train_methyl_age_mi,
                # actual methyl vs age
                'train_Amethyl_v_Age_pearson_abs': train_actual_methyl_age_pearson,
                'train_Amethyl_v_Age_spearman_abs': train_actual_methyl_age_spearman,
                'train_Amethyl_v_Age_mi': train_actual_methyl_age_mi"""
                
            dataset_perf_df['dataset'] = dataset
            dataset_perf_df['cpg'] = target_train_names
            dataset_perf_dfs.append(dataset_perf_df)
            print("done with dataset: " + dataset, flush = True)
        all_dataset_perf_df = pd.concat(dataset_perf_dfs)
        # make cpg a column
        all_dataset_perf_df.reset_index(inplace = True, drop = True)
        if predicted_with_random_feat:
            all_dataset_perf_df['self_pred'] = all_dataset_perf_df['cpg'].apply(
                lambda x: True if x.split('_')[1] == x.split('_')[3] else False
                )
        self.performance_by_dataset_df = all_dataset_perf_df
    
    
    def get_model_feat_names_train_test(
        self, 
        cpg_name,
        dataset = ""
        ):
        """
        For a given CpG name, return the trained model object and the feature names
        """
        model = self.trained_models[cpg_name]
        feat_names = self.feature_mats['feat_names'][cpg_name]
        
        if dataset != "":
            # subset samples to only those in the dataset
            dset_samples = self.all_methyl_age_df_t.loc[self.all_methyl_age_df_t['dataset'] == dataset, :].index
            test_samples = list(set(self.test_samples) & set(dset_samples))
            train_samples = list(set(self.train_samples) & set(dset_samples))
        else:
            test_samples = self.test_samples
            train_samples = self.train_samples
        
        train_idx_num = [
            self.feature_mats['target_values'][cpg_name].index.get_loc(train_sample)
            for train_sample in self.train_samples
            ]
        test_idx_num = [
                self.feature_mats['target_values'][cpg_name].index.get_loc(test_sample)
                for test_sample in self.test_samples
            ]
        feat_mat = self.feature_mats['feat_mats'][cpg_name].todense()
        train_mat = feat_mat[train_idx_num, :]
        test_mat = feat_mat[test_idx_num, :]
        return model, feat_names, train_mat, test_mat
    
    def load_model_fn_for_cpg(
        self,
        cpg_name,
        batch_size = 1000
        ):
        """
        Given a CpG, load the corresponding model
        @ cpg_name: the name of the CpG
        @ batch_size: the number of models written out together
        """
        # figure out which batch this CpG is from
        
    
    def get_one_cpg_feat_score_by_cat(
        self,
        model, 
        feat_names,
        train_mat,
        test_mat, 
        importance_calculator = 'xgb', # xgb or shap
        importance_type = 'gain'
        ):
        """
        Get the feature importances from model of feat_names and assign each feature to the category it came from.
        """
        import re
        
        # calculate feature importances
        if importance_calculator == 'xgb':
            importances = (
                model.get_booster()
                .get_score(importance_type = importance_type)
                )
            # get the feature names
            index_to_name = {
                f'f{i}': name for i, name in enumerate(feat_names)
                }
            # add the feature names to the importances dictionary
            importance_with_names = {
                index_to_name.get(key, key): value 
                for key, value in importances.items()
                }
            # make a dataframe of the feature importances and names
            feat_imp_df = pd.DataFrame(
                importance_with_names, index = ['importance']
                ).T.reset_index()
            feat_imp_df.rename(
                columns = {'index':'feat_name'}, inplace = True
            )
            # for values in feat_names that are not in feat_imp_df, add them with importance 0
            not_in_df = list(set(feat_names) - set(feat_imp_df['feat_name']))
            to_add_df =  pd.DataFrame({
                'feat_name': not_in_df,
                'importance': [0]*len(not_in_df)
                })
            feat_imp_df = pd.concat([feat_imp_df, to_add_df], axis = 0)
        elif importance_calculator == 'shap':
            # calculate shap values
            explainer = shap.TreeExplainer(model, train_mat)
            shap_values = explainer.shap_values(test_mat)
            # get the sum of absolute shap values for each feature
            shap_sum = np.abs(shap_values).sum(axis=0)
            # make a dataframe of the feature importances and names
            feat_imp_df = pd.DataFrame({
                'feat_name': feat_names,
                'importance': shap_sum
                })
        else:
            raise ValueError("importance_calculator must be 'xgb' or 'shap'")
        
        
        def name_to_cat(feat_name):
            """
            Given a feature name, return a category name
            """
            if feat_name[:7] == 'dataset':
                return 'dataset'
            elif 'MM' in feat_name:
                return 'Specific methylating motif'
            elif 'UM' in feat_name:
                return 'Specific unmethylating motif'
            elif 'mm' in feat_name:
                return 'Aggregate methylating motifs'
            elif 'um' in feat_name:
                return 'Aggregate unmethylating motifs'
            elif feat_name[:6] == 'gender':
                return 'gender'
            elif re.search(r'ext|corr', feat_name):
                # if it starts with a number
                if re.search(r'^\d+', feat_name):
                    return 'specific_' + '_'.join(feat_name.split('_')[1:])
                else:
                    return feat_name
            # if ends with tesselated, nested
            elif re.search(r'tesselated|nested', feat_name):
                # if it starts with a number
                if re.search(r'^\d+', feat_name):
                    return  'specific_' + feat_name.split('_')[-1]
                else:
                    return feat_name
            # if has no numbers or :, return the whole string
            elif not re.search(r'\d+:\d+', feat_name) and not re.search(r'\d+:\d+-\w+', feat_name):
                return feat_name
            # if onyl contains a number on either size of :, return loci_nearby
            elif re.search(r'\d+:\d+', feat_name):
                return 'Nearby locus'
            else:
                return "confused_" + feat_name
    
        # based on the feat_name, set category column
        feat_imp_df['category'] = feat_imp_df['feat_name'].apply(name_to_cat)
        # get the sum, mean, median, and count of the importances for each category
        feat_imp_by_cat_df = pd.DataFrame({
            'sum': feat_imp_df.groupby('category')['importance'].sum(),
            'mean': feat_imp_df.groupby('category')['importance'].mean(),
            'median': feat_imp_df.groupby('category')['importance'].median(),
            'max': feat_imp_df.groupby('category')['importance'].max(),
            'min': feat_imp_df.groupby('category')['importance'].min(),
            'count': feat_imp_df.groupby('category')['importance'].count(),
            # number of non-zero values
            'num_nonzero': feat_imp_df.groupby('category')['importance'].apply(lambda x: len(x[x > 0]))
                
            }).reset_index()
        
        feat_imp_by_cat_df['prop_nonzero'] = (
            feat_imp_by_cat_df['num_nonzero'] 
            / feat_imp_by_cat_df['count']
            )
        return feat_imp_df, feat_imp_by_cat_df
            
    def plot_real_vs_predicted_methylation(
        self, 
        cpg: str, 
        dataset: str = "", 
        sample_set: str = "test",
        ):
        sns.set_context("paper")
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
        fig3, axes3 = plt.subplots(figsize=(6, 4))
        fig4, axes4 = plt.subplots(figsize=(6, 4))
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
        axes.legend(title='Age')
        min_val = min(min(predicted_values), min(actual_values))
        max_val = max(max(predicted_values), max(actual_values))
        axes.plot([min_val, max_val], [min_val, max_val], color='black')

        for_box_df = pd.DataFrame({'Predicted': predicted_values, 'Actual': actual_values})
        # make interval index for bins
        bins = pd.IntervalIndex.from_tuples([(0, .2), (.2, .4), (.4, .6), (.6, .8)])          
        for_box_df['actual_bin'] = pd.cut(for_box_df['Actual'].values, bins=bins)
        sns.violinplot(
            x='actual_bin', y='Predicted', data=for_box_df, ax=axes3,
            palette=['maroon'], alpha = .5
        )
        axes3.set_ylabel(f'UCEC predicted methylation {cpg}')
        axes3.set_xlabel(f'UCEC actual methylation {cpg}')
        # plot age vs methylation value for both predicted and actual as a violin plot
        # bin age into 4 equal width bins
        pred_act_df['bin_age'] = pd.cut(pred_act_df['Age'].values, bins=4)
        sns.violinplot(
            x='bin_age', y='Methylation fraction', hue = 'Type', data=pred_act_df, ax=axes4,
            palette=['maroon', 'steelblue'], alpha = .5
        )
        #plt.savefig('/cellar/users/zkoch/methylation_and_mutation/output_dirs/final_figures/figure5/figure5B_methyl_pred_example.svg', dpi=300, format = 'svg')
        return for_box_df
        # change legend title
        # y = x line based on min and max values
        
    
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
        if model_type == 'elasticNet':
            # Create an ElasticNetCV object
            model = ElasticNetCV(
                cv=5, random_state=0, max_iter=10000,
                selection = 'random', n_jobs=-1, verbose=0
                )
            model.fit(X, y)
        elif model_type == 'xgboost':
            """# Create a parameter grid for the XGBoost model
            param_grid = {
                #'learning_rate': np.logspace(-4, 0, 50),
                #'n_estimators': range(100, 500, 100),
                #'max_depth': range(3, 10),
                #'min_child_weight': range(1, 6),
                #'gamma': np.linspace(0, 0.5, 50),
                #'subsample': np.linspace(0.5, 1, 50),
                #'colsample_bytree': np.linspace(0.5, 1, 50),
                #'reg_alpha': np.logspace(-4, 0, 50),
                #'reg_lambda': np.logspace(-4, 0, 50)
            }
            # Create the XGBRegressor model
            model = xgb.XGBRegressor(n_jobs=-1, random_state=0)
            # Initialize the RandomizedSearchCV object
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=100,  # number of parameter settings that are sampled
                scoring='neg_mean_absolute_error',
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
            model = xgb.XGBRegressor(n_jobs=-1)
            model.fit(X, y)
        else:
            raise ValueError("model_type must be 'elasticnet' or 'xgboost'")
        return model
   
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
    
    def scan_for_best_clock_actual_methylation(
        self,
        datasets: list,
        cpg_choosing_metrics: list,
        number_of_cpgs: list,
        model_types: list,
        train_tissues: list
        ):
        iters = len(datasets) * len(cpg_choosing_metrics) * len(number_of_cpgs) * len(model_types) * len(train_tissues)
        print(f"doing {iters} iterations")
        i = 0
        results = []
        for dataset in datasets:
            for cpg_choosing_metric in cpg_choosing_metrics:
                for number_of_cpg in number_of_cpgs: 
                    for model_type in model_types:
                        for train_tissue in train_tissues:
                            # get the top CpGs from this dataset based on the metric
                            top_cpgs = self.performance_by_dataset_df.query(
                                "dataset == @dataset").sort_values(
                                    by = cpg_choosing_metric, ascending = False).head(
                                        number_of_cpg)['cpg'].to_list()
                            # get this datasets sample's
                            this_dset_samples = self.all_methyl_age_df_t.loc[
                                self.all_methyl_age_df_t['dataset'] == dataset, :
                                    ].index.tolist()
                            this_dset_test_samples = list(
                                set(this_dset_samples).intersection(set(self.test_samples))
                                )
                            # if we are using the tissue being predicted to train
                            if train_tissue == "self":
                                # select this tissue's training samples
                                chosen_train_samples = list(
                                    set(this_dset_samples).intersection(set(self.train_samples))
                                    )
                            elif train_tissue == "all_others":
                                # otherwise select all other tissue's training samples
                                all_other_dset_samples = self.all_methyl_age_df_t.loc[
                                    self.all_methyl_age_df_t['dataset'] != dataset, :
                                        ].index.tolist()
                                chosen_train_samples = list(
                                    set(all_other_dset_samples).intersection(set(self.train_samples))
                                    )
                            else: # train_tissue == some specific dataset
                                if train_tissue == dataset:
                                    continue
                                # otherwise select all other tissue's training samples
                                other_specific_dset_samples = self.all_methyl_age_df_t.loc[
                                    self.all_methyl_age_df_t['dataset'] != train_tissue, :
                                        ].index.tolist()
                                chosen_train_samples = list(
                                    set(all_other_dset_samples).intersection(set(self.train_samples))
                                    )
                            # get the methyl data to use for training, predicted or actual
                            methyl_to_use = self.all_methyl_age_df_t
                            train_samples = chosen_train_samples
                            
                            X_train = methyl_to_use.loc[train_samples, top_cpgs]
                            y_train = self.all_methyl_age_df_t.loc[train_samples, 'age_at_index']
                            X_test = self.predicted_methyl_df.loc[this_dset_test_samples, top_cpgs]
                            y_test = self.all_methyl_age_df_t.loc[this_dset_test_samples, 'age_at_index'] 
                            
                            #train clock
                            trained_clock = self.train_epi_clock(
                                X_train, y_train, model_type = model_type
                                )
                            # apply clock
                            y_pred = trained_clock.predict(X_test)
                            p = pearsonr(y_test, y_pred)[0]
                            s = spearmanr(y_test, y_pred)[0]
                            
                            # add everything to the results dict
                            this_results = {
                                "dataset": dataset,
                                "cpg_choosing_metric": cpg_choosing_metric,
                                "number_of_cpg": number_of_cpg,
                                "model_type": model_type,
                                "train_tissue": train_tissue,
                                "pearson": p,
                                "spearman": s,
                                "trained_clock": trained_clock,
                                "y_pred": y_pred,
                                "y_test": y_test
                            }
                            i += 1
                            results.append(this_results)
                    print(f"Done {i / iters}")
        return pd.DataFrame(results).sort_values('pearson', ascending = False)

    
    def scan_for_best_clock(
        self,
        datasets: list,
        cpg_choosing_metrics: list,
        number_of_cpgs: list,
        training_methylation_types: list,
        train_sets: list,
        model_types: list,
        train_tissues: list
        ):
        """
        Scan through all combinations of parameters to find the best clock for each dataset
        @ datasets: a list of datasets to use
        @ cpg_choosing_metrics: a list of metrics to use to choose the top cpgs
        @ number_of_cpgs: a list of number of cpgs to use
        @ training_methylation_types: [pred, actual]
        @ train_sets: [train, test]
        @ model_types: a list of model types to use
        @ train_tissues: [self, all_others]
        """
        # based on length of lists, print number of iterations that will be performed
        iters = len(datasets) * len(cpg_choosing_metrics) * len(number_of_cpgs) * len(training_methylation_types) * len(train_sets) * len(model_types) * len(train_tissues)
        print(f"doing {iters} iterations")
        i = 0
        results = []
        predicted_samples = self.predicted_methyl_df.index.tolist()
        # track nested for loop progress based on expected iters 
        for dataset in datasets:
            for cpg_choosing_metric in cpg_choosing_metrics:
                for number_of_cpg in number_of_cpgs:
                    for training_methylation_type in training_methylation_types:
                        for train_set in train_sets:
                            for model_type in model_types:
                                for train_tissue in train_tissues:
                                    # get the top CpGs from this dataset based on the metric
                                    top_cpgs = self.performance_by_dataset_df.query(
                                        "dataset == @dataset").sort_values(
                                            by = cpg_choosing_metric, ascending = False
                                            ).head(number_of_cpg)['cpg'].unique().tolist()
                                    # get this datasets sample's
                                    this_dset_samples = self.all_methyl_age_df_t.loc[
                                        self.all_methyl_age_df_t['dataset'] == dataset, :
                                            ].index.tolist()
                                    this_dset_test_samples = list(
                                        set(this_dset_samples).intersection(set(self.test_samples))
                                        )
                                    this_dset_test_samples = list(
                                        set(this_dset_test_samples).intersection(set(predicted_samples))
                                        )
                                    
                                    # if we are using the tissue being predicted to train
                                    if train_tissue == "self":
                                        # select this tissue's training samples
                                        chosen_train_samples = list(
                                            set(this_dset_samples).intersection(set(self.train_samples))
                                            )
                                    elif train_tissue == "all_others":
                                        # otherwise select all other tissue's training samples
                                        all_other_dset_samples = self.all_methyl_age_df_t.loc[
                                            self.all_methyl_age_df_t['dataset'] != dataset, :
                                                ].index.tolist()
                                        chosen_train_samples = list(
                                            set(all_other_dset_samples).intersection(set(self.train_samples))
                                            )
                                    else: # train_tissue == some specific dataset
                                        if train_tissue == dataset:
                                            continue
                                        # otherwise select all other tissue's training samples
                                        other_specific_dset_samples = self.all_methyl_age_df_t.loc[
                                            self.all_methyl_age_df_t['dataset'] != train_tissue, :
                                                ].index.tolist()
                                        chosen_train_samples = list(
                                            set(all_other_dset_samples).intersection(set(self.train_samples))
                                            )
                                    chosen_train_samples = list(
                                        set(chosen_train_samples).intersection(set(predicted_samples))
                                        )
                                    
                                    # get the methyl data to use for training, predicted or actual
                                    methyl_to_use = self.predicted_methyl_df \
                                                    if training_methylation_type != "actual" \
                                                    else self.all_methyl_age_df_t
                                                    
                                    # if we are using actual methyl, we can choose either train or test
                                    if training_methylation_type == "actual":
                                        # we can choose either train or test
                                        train_samples = this_dset_test_samples \
                                                    if train_set == "test" \
                                                    else chosen_train_samples
                                    # otherwise, we have to use the training samples            
                                    elif training_methylation_type == "pred":
                                        train_samples = chosen_train_samples
                                        # set train_set to reflect
                                        train_set = "train"

                                    X_train = methyl_to_use.loc[train_samples, top_cpgs]
                                    y_train = self.all_methyl_age_df_t.loc[train_samples, 'age_at_index']
                                    X_test = self.predicted_methyl_df.loc[this_dset_test_samples, top_cpgs]
                                    y_test = self.all_methyl_age_df_t.loc[this_dset_test_samples, 'age_at_index'] 
                                    
                                    #train clock
                                    trained_clock = self.train_epi_clock(
                                        X_train, y_train, model_type = model_type
                                        )
                                    # apply clock
                                    y_pred = trained_clock.predict(X_test)
                                    p = pearsonr(y_test, y_pred)[0]
                                    s = spearmanr(y_test, y_pred)[0]
                                    
                                    # add everything to the results dict
                                    this_results = {
                                        "dataset": dataset,
                                        "cpg_choosing_metric": cpg_choosing_metric,
                                        "number_of_cpg": number_of_cpg,
                                        "training_methylation_type": training_methylation_type,
                                        "train_set": train_set,
                                        "model_type": model_type,
                                        "train_tissue": train_tissue,
                                        "pearson": p,
                                        "spearman": s,
                                        "trained_clock": trained_clock,
                                        "y_pred": y_pred,
                                        "y_test": y_test
                                    }
                                    i += 1
                                    results.append(this_results)
                        print(f"Done {i / iters}")
        return pd.DataFrame(results).sort_values('pearson', ascending = False)
    
    def plot_feature_matrix_heatmap(self, cpg, dataset = "", only_agg_feats = False):
        """
        Plot the feature matrix for a particular CpG, optionally in a particular dataset 
        """
        sns.set_context('paper')
        fig, axes = plt.subplots(figsize=(8, 6), dpi=100)
        # get the feature names
        model, feat_names = self.get_model_and_feat_names(
            cpg_name = cpg
            )
        # get the feature matrix for this cpg
        feat_mat = pd.DataFrame(
            self.feature_mats['feat_mats'][cpg].toarray(),
            columns=feat_names, index = self.feature_mats['target_values'][cpg].index.tolist()
            )
        if dataset != "":
            col_name = 'dataset_' + dataset
            feat_mat = feat_mat.loc[feat_mat[col_name] == 1]
        if only_agg_feats:
            pattern = '^[0-9:]*$'
            # select columns that do not match the pattern
            selected_columns = feat_mat.columns[~feat_mat.columns.str.contains(pattern, regex=True)]
            feat_mat = feat_mat[selected_columns]
        # put in order of this datasets training then testing samples
        this_dset_samples = self.all_methyl_age_df_t.query("dataset == @dataset").index
        this_dset_train_samples = list(
            set(this_dset_samples.tolist()).intersection(set(self.train_samples))
            )
        this_dset_test_samples = list(
            set(this_dset_samples.tolist()).intersection(set(self.test_samples))
            )
        # reindex the feature matrix so that it is in the order of the training then testing samples
        feat_mat = feat_mat.reindex(
            this_dset_train_samples + this_dset_test_samples
            )
        # scale each column to 0-1
        feat_mat_scaled = feat_mat.apply(
            lambda x: (x - np.min(x)) / max(1e-20, (np.max(x) - np.min(x)))
            )
        # drop columns with no nonzero values 
        feat_mat_scaled = feat_mat_scaled.loc[
            :, (feat_mat_scaled != 0).any(axis=0)
            ]
        # plot a heatmap
        sns.heatmap(feat_mat_scaled, cmap = 'rocket', ax = axes, rasterized = True)
        plt.savefig('/cellar/users/zkoch/methylation_and_mutation/output_dirs/final_figures/figure5/figure5A_feat_mat.svg', dpi=300, format = 'svg')
        return feat_mat_scaled
    
    
class optimizeSomage:
    """
    Take in multiple mutationClock objects and compare them to find the best model and hyperparameters
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
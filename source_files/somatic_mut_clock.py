import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNetCV, RidgeCV
import matplotlib.pyplot as plt
import time

class mutationClock:
    def __init__(
        self,
        all_mut_w_age_df: pd.DataFrame,
        illumina_cpg_locs_df: pd.DataFrame, 
        all_methyl_age_df_t: pd.DataFrame,
        output_dir: str,
        matrix_qtl_dir: str = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/muts",
        godmc_meqtl_fn: str = "/cellar/users/zkoch/methylation_and_mutation/data/meQTL/goDMC_meQTL/goDMC_meQTLs.parquet",
        pancan_meqtl_fn: str = "/cellar/users/zkoch/methylation_and_mutation/data/meQTL/pancan_tcga_meQTL/pancan_meQTL.parquet"
        ) -> None:

        self.all_mut_w_age_df = all_mut_w_age_df
        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.output_dir = output_dir
        self.matrix_qtl_dir = matrix_qtl_dir
        # only non X and Y chromosomes and that occured in samples with measured methylation
        self.all_mut_w_age_df['mut_loc'] = self.all_mut_w_age_df['chr'] + ':' \
                                         + self.all_mut_w_age_df['start'].astype(str)
        self.all_mut_w_age_df = self.all_mut_w_age_df.loc[
            (self.all_mut_w_age_df['chr'] != 'X') 
            & (self.all_mut_w_age_df['chr'] != 'Y')
            & (self.all_mut_w_age_df['case_submitter_id'].isin(self.all_methyl_age_df_t.index)),
            :]
        # join self.all_mut_w_age_df with the illumina_cpg_locs_df
        all_mut_w_age_illum_df = self.all_mut_w_age_df.copy(deep=True)
        all_mut_w_age_illum_df['start'] = pd.to_numeric(self.all_mut_w_age_df['start'])
        self.all_mut_w_age_illum_df = all_mut_w_age_illum_df.merge(
                                        self.illumina_cpg_locs_df, on=['chr', 'start'], how='left'
                                        )
        # subset illumina_cpg_locs_df to only the CpGs that are measured, and remove XT
        self.illumina_cpg_locs_df = self.illumina_cpg_locs_df.loc[
            self.illumina_cpg_locs_df['#id'].isin(self.all_methyl_age_df_t.columns)
            & (self.illumina_cpg_locs_df['chr'] != 'X') 
            & (self.illumina_cpg_locs_df['chr'] != 'Y')
            ]
        # read in the matrixQTL results from databases
        self.godmc_meqtl_df = pd.read_parquet(godmc_meqtl_fn)        
        self.pancan_meqtl_df = pd.read_parquet(
            pancan_meqtl_fn,
            columns = ['snp', 'cpg', 'beta', 'p-value']
            )
        # one hot encode gender and tissue type
        dset_col = self.all_methyl_age_df_t['dataset'].to_list()
        self.all_methyl_age_df_t = pd.get_dummies(self.all_methyl_age_df_t, columns=["gender", "dataset"])
        # add back in the dataset column
        self.all_methyl_age_df_t['dataset'] = dset_col
        # cache :P
        self.matrixQTL_store = {}
        
    def mutual_info(
        self, 
        X: pd.DataFrame,
        covariate: pd.Series, 
        bins: int = 5
        ) -> pd.Series:
        '''
        Estimates mutual information between X (samples x CpG sites, samples x features, etc.) /
        and some covariate. Uses methylation in self.all_methyl_age_df_t
        @ X: samples X something matrix
        @ covariate: pandas series of covariate to use
        @ bins: number of bins to use for discretization
        '''
        # transpose X so can be input in usual dimension
        X = X.T
        assert len(covariate.index) == len(X.columns), \
            'dimensions of covariate are %s, but X matrix are %s' \
             %  (covariate.shape, X.shape)
        def shan_entropy(c): # inner func for entropy
            c_normalized = c / float(np.sum(c))
            c_normalized = c_normalized[np.nonzero(c_normalized)]
            H = -sum(c_normalized* np.log2(c_normalized))  
            return H
        MI = []
        for col in X.values:
            nas = np.logical_and(~np.isnan(col), ~np.isnan(covariate))
            c_XY = np.histogram2d(col[nas], covariate[nas],bins)[0]
            c_X = np.histogram(col[nas], bins)[0]
            c_Y = np.histogram(covariate[nas], bins)[0]
            H_X = shan_entropy(c_X)
            H_Y = shan_entropy(c_Y)
            H_XY =shan_entropy(c_XY)
            MI.append(H_X + H_Y - H_XY)
        MI = pd.Series(MI, index=X.index)
        return MI        
        
    def _select_correl_sites(
        self,
        cpg_id: str,
        cpg_chr: str,
        num_correl_sites: int,
        train_samples: list
        ) -> dict:
        """
        Just in time correlation to find the most correlated sites to the mutation event CpG in matched train_samples
        @ cpg_id: the id cpg to corr with
        @ cpg_chr: the chromosome of the cpg
        @ num_correl_sites: the number of sites to return, half pos half neg
        @ train_samples: the samples to use for correlation
        @ return: dict of {pos_cor: list of pos correlated positions, neg_cor: list of neg correlated positions}
        """
        # get the CpG's MF
        cpg_mf = self.all_methyl_age_df_t.loc[train_samples, cpg_id]
        # get the MF of all same chrom CpGs
        same_chrom_cpgs = self.illumina_cpg_locs_df.loc[
            self.illumina_cpg_locs_df['chr'] == cpg_chr, # exclude the mut_cpg
            '#id'].values
        same_chrom_cpgs_mf = self.all_methyl_age_df_t.loc[train_samples, same_chrom_cpgs]
        # get correlation between mut_cpg and all same chrom CpGs
        corrs = same_chrom_cpgs_mf.corrwith(cpg_mf, axis=0)
        corrs.sort_values(ascending=True, inplace=True)
        idx = corrs.index.to_list()
        pos_corrs = idx[-int(num_correl_sites/2):]
        neg_corrs = idx[:int(num_correl_sites/2)]
        # convert to locations
        pos_corr_locs = (
            self.illumina_cpg_locs_df.loc[
                self.illumina_cpg_locs_df['#id'].isin(pos_corrs)
                ].assign(location=lambda df: df['chr'] + ':' + df['start'].astype(str))['location']
            .tolist()
            )
        neg_corr_locs = (
            self.illumina_cpg_locs_df.loc[
                self.illumina_cpg_locs_df['#id'].isin(neg_corrs)
                ].assign(location=lambda df: df['chr'] + ':' + df['start'].astype(str))['location']
            .tolist()
            )
        return {'pos_corr': pos_corr_locs, 'neg_corr': neg_corr_locs}
        
    def _get_matrixQTL_sites(
        self,
        cpg_id: str,
        chrom: str,
        max_meqtl_sites: int
        ) -> dict:
        """
        Given a CpG, get the max_meqtl_sites number meQTLs with smallest p-value in relation to this CpG
        @ cpg_id: the CpG id
        @ chrom: the chromosome of the CpG, to read in the correct matrixQTL results
        @ max_meqtl_sites: the maximum number of meQTLs to return
        @ return: a dict of the meQTLs split by beta, 'chr:start', with smallest p-value
        """
        # if chrom is not in the keys of the matrixQTL_store
        if chrom not in self.matrixQTL_store:
            # read in the matrixQTL results for this chromosome        
            meqtl_df = pd.read_parquet(
                os.path.join(self.matrix_qtl_dir, f"chr{chrom}_meqtl.parquet"),
                columns=['#id', 'SNP', 'p-value', 'beta'])
            self.matrixQTL_store[chrom] = meqtl_df
        else:
            meqtl_df = self.matrixQTL_store[chrom]
        # get the meQTLs for this CpG
        meqtls = meqtl_df.loc[meqtl_df['#id'] == cpg_id, :]
        # get the max_meqtl_sites meQTLS with smallest p-value
        meqtls = meqtls.nsmallest(max_meqtl_sites, 'p-value')
        return {'matrixqtl_neg_beta': meqtls.loc[meqtls['beta'] < 0, 'SNP'].to_list(),
                'matrixqtl_pos_beta': meqtls.loc[meqtls['beta'] > 0, 'SNP'].to_list()}
            
    def _get_db_sites(
        self,
        cpg_id: str
        ) -> dict:
        """
        Return the meQTL locations in either of the databases for the given CpG
        @ cpg_id: the CpG id
        @ returns: a dict of the meQTLs from each db split by beta, 'chr:start'
        """
        godmc_metqtls = self.godmc_meqtl_df.loc[self.godmc_meqtl_df['cpg'] == cpg_id, :]
        pos_godmc_meqtls = godmc_metqtls.loc[godmc_metqtls['beta_a1'] > 0, 'snp'].to_list()
        neg_godmc_meqtls = godmc_metqtls.loc[godmc_metqtls['beta_a1'] < 0, 'snp'].to_list()
        pancan_meqtls = self.pancan_meqtl_df.loc[self.pancan_meqtl_df['cpg'] == cpg_id, :]
        pos_pancan_meqtls = pancan_meqtls.loc[pancan_meqtls['beta'] > 0, 'snp'].to_list()
        neg_pancan_meqtls = pancan_meqtls.loc[pancan_meqtls['beta'] < 0, 'snp'].to_list()
        return {'pos_godmc': pos_godmc_meqtls, 'neg_godmc': neg_godmc_meqtls,
                'pos_pancan': pos_pancan_meqtls, 'neg_pancan': neg_pancan_meqtls}
                
    def get_predictor_site_groups(
        self, 
        cpg_id: str,
        train_samples: list,
        num_correl_sites: int,
        max_meqtl_sites: int,
        nearby_window_size: int
        ) -> list:
        """
        Get the sites to be used as predictors of cpg_id's methylation
        @ cpg_id: the id of the CpG
        @ train_samples: the train_samples to be used
        @ num_correl_sites: the number of correlated sites to be used
        @ max_meqtl_sites: the maximum number of meQTLs to be used
        @ nearby_window_size: the window size to be used to find nearby sites
        @ returns: list of genomic locations of the sites to be used as predictors in format chr:start
        """
        # get cpg_id's chromosome and start position
        try:
            chrom = self.illumina_cpg_locs_df.loc[
                self.illumina_cpg_locs_df['#id'] == cpg_id, 'chr'
            ].values[0]
            start = self.illumina_cpg_locs_df.loc[
                self.illumina_cpg_locs_df['#id'] == cpg_id, 'start'
            ].values[0]
        # for some reason this cpg is not in illumina_cpg_locs_df, return empty dict
        except:
            return {}
        # get dict of positions of most positively and negatively correlation CpG
        corr_dict = self._select_correl_sites(cpg_id, chrom, num_correl_sites, train_samples)
        # get sites (and cpg_id itself position) within nearby_window_size of cpg_id
        very_nearby_site_locs = [chrom + ':' + str(start + i) for i in range(-int(50), int(50) + 1)]
        nearby_site_locs = [chrom + ':' + str(start + i) for i in range(-int(nearby_window_size/2), int(nearby_window_size/2) + 1) if chrom + ':' + str(start + i) not in very_nearby_site_locs ]
        # get sites from databases
        db_meqtl_locs = self._get_db_sites(cpg_id)
        # get sites from matrixQTL 
        matrix_meqtl_locs = self._get_matrixQTL_sites(cpg_id, chrom, max_meqtl_sites)
        return {
            'pos_corr': corr_dict['pos_corr'], 'neg_corr': corr_dict['neg_corr'],
            'nearby': nearby_site_locs, 'very_nearby' : very_nearby_site_locs, 
            'matrixqtl_neg_beta': matrix_meqtl_locs['matrixqtl_neg_beta'], 
            'matrixqtl_pos_beta': matrix_meqtl_locs['matrixqtl_pos_beta'], 
            'pos_godmc': db_meqtl_locs['pos_godmc'], 'neg_godmc': db_meqtl_locs['neg_godmc'],
            'pos_pancan': db_meqtl_locs['pos_pancan'], 'neg_pancan': db_meqtl_locs['neg_pancan']
            }
    
    def _create_training_mat(
        self, 
        cpg_id: str, 
        predictor_groups: dict,
        samples: list,
        aggregate: bool
        ) -> tuple:
        """
        Create the training matrix for the given cpg_id and predictor_sites
        @ cpg_id: the id of the CpG
        @ predictor_groups: dict of lists of sites to be used as predictors of cpg_id's methylation
        @ samples: the samples to be included in the training matrix
        @ aggregate: whether to aggregate the mutation status by predictor group
        @ returns: X, y where X is the training matrix and y is the methylation values of cpg_id across samples
        """
        # for each sample, get the cpg_id methylation values
        y = self.all_methyl_age_df_t.loc[samples, cpg_id]
        if not aggregate:
            # get list of unique predictor sites
            predictor_sites = set()
            for key in predictor_groups:
                predictor_sites.update(predictor_groups[key])
            predictor_sites = list(predictor_sites)
            # get the mutation status of predictor sites
            mut_status = self.all_mut_w_age_df.loc[
                self.all_mut_w_age_df['mut_loc'].isin(predictor_sites),
                ['DNA_VAF', 'case_submitter_id', 'mut_loc']
                ]
            # create a new dataframe with columns = predictor sites, rows = y.index,
            # and values = variant allele frequencies
            X = pd.pivot_table(mut_status, index='case_submitter_id', columns='mut_loc', values='DNA_VAF', fill_value = 0)
            # add rows of all 0s for samples that don't have any mutations in predictor sites
            X = X.reindex(y.index, fill_value=0)
        else:
            aggregated_muts = []
            for group, one_group_sites in predictor_groups.items():
                mut_status = self.all_mut_w_age_df.loc[
                    self.all_mut_w_age_df['mut_loc'].isin(one_group_sites),
                    ['DNA_VAF', 'case_submitter_id', 'mut_loc']
                    ]
                mut_status = pd.pivot_table(
                    mut_status, index='case_submitter_id', columns='mut_loc',
                    values='DNA_VAF', fill_value = 0
                    )
                mut_status = mut_status.reindex(y.index, fill_value=0)
                agg_mut_status = mut_status.sum(axis=1)
                aggregated_muts.append(agg_mut_status)
            # create dataframe from binned_muts with columns pred_type
            X = pd.concat(aggregated_muts, axis=1)
            X.columns = predictor_groups.keys()
        # add covariate columns
        coviariate_col_names = [
            col for  col in self.all_methyl_age_df_t.columns \
            if col.startswith('dataset_') or col.startswith('gender_')
            ]
        covariate_df = self.all_methyl_age_df_t.loc[X.index, coviariate_col_names]
        X = pd.merge(X, covariate_df, left_index=True, right_index=True)
        return X, y
            
    def evaluate_predictor(
        self, 
        cpg_id: str,
        train_samples: list,
        num_correl_sites: int,
        max_meqtl_sites: int,
        nearby_window_size: int,
        aggregate: bool = True,
        feat_store: str = ""
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
        else: # read from saved pkl file
            predictor_groups_fn = os.path.join(feat_store, f"{cpg_id}.pkl")
            # open the file
            predictor_groups = pickle.load(open(predictor_groups_fn, 'rb'))
        if predictor_groups == {}:
            return []
        # create the training matrix
        X, y = self._create_training_mat(
            cpg_id, predictor_groups, train_samples, aggregate = aggregate
            )

        model = ElasticNetCV(cv=5, random_state=0, max_iter=5000, n_jobs=-1, selection='random')
        # model = RidgeCV(cv=5)
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

        model2 = ElasticNetCV(cv=5, random_state=0, max_iter=5000, n_jobs=-1, selection='random')
        # model2 = RidgeCV(cv=5)
        # repeat with just the covariates
        base_maes, base_r2s, base_feature_names, base_coefs, base_intercepts = [], [], [], [], []
        coviariate_col_names = [
            col for  col in X.columns \
            if col.startswith('dataset_') or col.startswith('gender_')
            ]
        X_cov_only = X[coviariate_col_names]
        for train_index, test_index in cv.split(X):
            X_train, X_test = X_cov_only.iloc[train_index, :], X_cov_only.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model2.fit(X_train, y_train)
            preds = model2.predict(X_test)
            base_maes.append(mean_absolute_error(y_test, preds))
            base_r2s.append(r2_score(y_test, preds))
            base_feature_names.append(X_train.columns[model2.coef_ != 0].to_list())
            base_coefs.append(model2.coef_[model2.coef_ != 0])
            base_intercepts.append(model2.intercept_)
        # create df with columns cpg_id, mae, r2, feature_names, base_mae, base_r2, base_feature_names, with entries as the lists, not expanding
        """result_df = pd.DataFrame({
            'cpg_id': cpg_id, 'mae': np.mean(maes), 'r2': np.mean(r2s), 'feature_names': [feature_names], 'coefs': [coefs],'intercepts': [intercepts], 'base_mae': np.mean(base_maes), 'base_r2': np.mean(base_r2s),
            'base_feature_names': [base_feature_names], 'base_coefs': [base_coefs], 'base_intercepts': [base_intercepts] 
            })"""
        result_df = pd.DataFrame({
            'mae': np.mean(maes), 'r2': np.mean(r2s), 
            'base_mae': np.mean(base_maes), 'base_r2': np.mean(base_r2s)
            }, index=[cpg_id])
        return result_df
    
    def feature_informations(self, 
        cpg_id: str,
        train_samples: list,
        num_correl_sites: int,
        max_meqtl_sites: int,
        nearby_window_size: int,
        aggregate: bool = True,
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
        else: # read from saved pkl file
            predictor_groups_fn = os.path.join(feat_store, f"{cpg_id}.pkl")
            # open the file
            predictor_groups = pickle.load(open(predictor_groups_fn, 'rb'))
        if predictor_groups == {}:
            return []
        # create the training matrix
        X, y = self._create_training_mat(
            cpg_id, predictor_groups, train_samples, 
            aggregate = aggregate
            )
        # calculate the mutual information
        mi = self.mutual_info(X, covariate = y)
        return mi
        
    def save_all_predictor_groups(
        self, 
        num_correl_sites: int,
        max_meqtl_sites: int,
        nearby_window_size: int,
        out_dir: str,
        cpg_ids: list = [],
        train_samples: list = [],
        ) -> None:
        """
        Save the predictor sites for each CpG to be later loaded
        @ num_correl_sites: the number of corr sites to be used as predictors
        @ max_meqtl_sites: the maximum number of matrixQTL sites to be used as predictors
        @ nearby_window_size: the window size to be used for nearby sites
        @ cpg_ids: the list of CpGs to be preprocessed
        @ train_samples: the list of samples to be used (only matter for correl sites)
        """
        # make the output directory if it doesn't exist
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for i, cpg_id in enumerate(cpg_ids):
            predictor_groups = self.get_predictor_site_groups(
                cpg_id, train_samples, num_correl_sites,
                max_meqtl_sites, nearby_window_size
                )
            with open(os.path.join(out_dir, f"{cpg_id}.pkl"), 'wb+') as f:
                pickle.dump(predictor_groups, f)
            if i % 10 == 0:
                print(f"Finished {i/len(cpg_ids)}% of CpGs")
    
    def train_predictor(
        self, 
        cpg_id: str,
        train_samples: list,
        num_correl_sites: int,
        max_meqtl_sites: int,
        nearby_window_size: int,
        aggregate: bool
        ):
        """
        Build the predictor for one CpG
        @ cpg_id: the id of the CpG to predict
        @ train_samples: list of samples to use for training
        @ num_correl_sites: number of sites to use as predictors
        @ max_meqtl_sites: maximum number of meqtl db sites to use as predictors
        @ nearby_window_size: window size to use for choosing nearby sites
        """
        # get the sites to be used as predictors
        predictor_sites = self.get_predictor_site_groups(
            cpg_id, train_samples, num_correl_sites,
            max_meqtl_sites, nearby_window_size
            )
        # create the training matrix
        X, y = self._create_training_mat(
            cpg_id, predictor_sites, train_samples, aggregate = aggregate
            )
        # train one elasticNet model to predict y from X
        model = ElasticNetCV(cv=5, random_state=0, max_iter=5000, selection = 'random', n_jobs=-1)
        model.fit(X, y)
        # write the model to a file using pickle
        model_fn = os.path.join(self.output_dir, f"{cpg_id}.pkl")
        pickle.dump(model, open(model_fn, "wb"))
        
    def train_all_predictors(
        self, 
        num_correl_sites: int,
        max_meqtl_sites: int,
        nearby_window_size: int,
        cpg_ids: list = [],
        train_samples: list = [],
        do: str = 'train',
        aggregate: bool = True,
        feat_store: str = ""
        ):
        """
        Train the predictor for all CpGs
        """
        # get the list of all CpGs
        if len(cpg_ids) == 0:
            cpg_ids = self.illumina_cpg_locs_df['#id'].to_list()
        if len(train_samples) == 0:
            train_samples = self.all_methyl_age_df_t.index.to_list()
        if do == 'train':
            # for each cpg, train the predictor
            for i, cpg_id in enumerate(cpg_ids):
                self.train_predictor(
                    cpg_id, train_samples, num_correl_sites, max_meqtl_sites,
                    nearby_window_size, aggregate) 
                if i % 10 == 0:
                    print(f"Finished {i/len(cpg_ids)}% of CpGs", flush=True)
        elif do == 'evaluate': # evaluate
            result_dfs = []
            for i, cpg_id in enumerate(cpg_ids):
                result_df = self.evaluate_predictor(
                    cpg_id, train_samples, num_correl_sites, max_meqtl_sites,
                    nearby_window_size, aggregate) 
                result_dfs.append(result_df)
                if i % 10 == 0:
                    print(f"Finished {100*i/len(cpg_ids)}% of CpGs", flush=True)
            result_df = pd.concat(result_dfs)
            return result_df.reset_index(drop=True)
        elif do == 'mutual_info':
            mis = {}
            for i, cpg_id in enumerate(cpg_ids):
                mis[cpg_id] = self.feature_informations(
                    cpg_id, train_samples, num_correl_sites,
                    max_meqtl_sites, nearby_window_size, aggregate,
                    feat_store
                    )
                if i % 100 == 0:
                    print(f"Finished {i/len(cpg_ids)}% of CpGs", flush=True)
            # remove any elements that are length 0
            mis = {k: v for k, v in mis.items() if len(v) > 0}
            mis_df = pd.DataFrame(data = mis)
            return mis_df
        else:
            print("'do' must be one of 'train', 'evaluate', 'mutual_info'")
            sys.exit(1)
        
    def predict_cpg(
        self, 
        cpg_id: str,
        test_samples: list,
        model_fn: str
        ) -> pd.DataFrame:
        """
        Predict the methylation level of one CpG for a list of samples
        @ cpg_id: the id of the CpG to predict
        @ test_samples: list of samples to predict
        @ model_fn: the file name of the pickled model
        """
        # get the predictor model
        model = pickle.load(open(model_fn, "rb"))
        # print list of attributes of the model
        predictor_sites = model.feature_names_in_
        # create the X matrix
        X, _ = self._create_training_mat(cpg_id, predictor_sites, test_samples)
        # predict the methylation levels
        preds = model.predict(X)
        # create a df with the samples as rows and the predicted methylation level as a column called cpg_id
        #return preds.tolist()
        return pd.DataFrame({cpg_id: preds}, index=test_samples)

    def predict_all_cpgs(
        self,
        cpg_ids: list,
        test_samples: list,
        model_dir: str
        ) -> pd.DataFrame:
        """
        Predict the methylation level of some CpGs for a list of samples using pretrained models
        @ cpg_ids: list of CpG ids to predict
        @ test_samples: list of samples to predict for
        @ model_dir: directory containing the pickled models
        """
        predictions = []
        for i, cpg_id in enumerate(cpg_ids):
            model_fn = os.path.join(model_dir, f"{cpg_id}.pkl")
            predictions.append(self.predict_cpg(cpg_id, test_samples, model_fn))
            if i % 100 == 0:
                print(i, flush=True)
        # concatenate the predictions into a single df
        all_pred_methyl_df = pd.concat(predictions, axis=1)
        return all_pred_methyl_df
    
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
        # import kfold cross validation
        from sklearn.model_selection import KFold
        # Create an ElasticNetCV object
        model = ElasticNetCV(
            cv=5, random_state=0, max_iter=5000,
            selection = 'random', n_jobs=48, verbose=1
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
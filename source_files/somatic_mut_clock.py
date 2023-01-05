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
from sklearn.linear_model import ElasticNetCV
# import random forest regressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


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
        # Preprocessing: subset to only mutations that are non X and Y chromosomes and that occured in samples with measured methylation
        self.all_mut_w_age_df['mut_loc'] = self.all_mut_w_age_df['chr'] + ':' + self.all_mut_w_age_df['start'].astype(str)
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
        # subset illumina_cpg_locs_df to only the CpGs that are measured
        # and remove chr X and Y
        self.illumina_cpg_locs_df = self.illumina_cpg_locs_df.loc[
            self.illumina_cpg_locs_df['#id'].isin(self.all_methyl_age_df_t.columns)
            & (self.illumina_cpg_locs_df['chr'] != 'X') 
            & (self.illumina_cpg_locs_df['chr'] != 'Y')
            ]
        # read in the matrixQTL results from databases
        self.godmc_meqtl_df = pd.read_parquet(godmc_meqtl_fn)        
        self.pancan_meqtl_df = pd.read_parquet(pancan_meqtl_fn)
        # one hot encode gender and tissue type
        self.all_methyl_age_df_t = pd.get_dummies(self.all_methyl_age_df_t, columns=["gender", "dataset"])
        
        self.matrixQTL_store = {}
        
    def _select_correl_sites(
        self,
        cpg_id: str,
        cpg_chr: str,
        num_correl_sites: int,
        samples: list
        ) -> list:
        """
        Just in time correlation to find the most correlated sites to the mutation event CpG in matched samples
        """
        # get the  CpG's MF
        cpg_mf = self.all_methyl_age_df_t.loc[samples, cpg_id]
        # get the MF of all same chrom CpGs
        same_chrom_cpgs = self.illumina_cpg_locs_df.loc[
            self.illumina_cpg_locs_df['chr'] == cpg_chr, # exclude the mut_cpg
            '#id'].values
        same_chrom_cpgs_mf = self.all_methyl_age_df_t.loc[samples, same_chrom_cpgs]
        # get correlation between mut_cpg and all same chrom CpGs
        corrs = same_chrom_cpgs_mf.corrwith(cpg_mf, axis=0)
        # choose the sites with largest absolute correlation
        return corrs.abs().sort_values(ascending=False).index[:num_correl_sites].to_list()
        
    def _get_matrixQTL_sites(
        self,
        cpg_id: str,
        chrom: str,
        max_meqtl_sites: int
        ) -> list:
        """
        Given a CpG, get the max_meqtl_sites number meQTLs with smallest p-value in relation to this CpG
        @ cpg_id: the CpG id
        @ chrom: the chromosome of the CpG, to read in the correct matrixQTL results
        @ max_meqtl_sites: the maximum number of meQTLs to return
        @ return: a list of the meQTLs, 'chr:start', with smallest p-value
        """
        # if chrom is not in the keys of the matrixQTL_store
        if chrom not in self.matrixQTL_store:
            # read in the matrixQTL results for this chromosome        
            meqtl_df = pd.read_parquet(
                os.path.join(self.matrix_qtl_dir, f"chr{chrom}_meqtl.parquet"),
                columns=['#id', 'SNP', 'p-value'])
            self.matrixQTL_store[chrom] = meqtl_df
        else:
            meqtl_df = self.matrixQTL_store[chrom]
        # get the meQTLs for this CpG
        meqtls = meqtl_df.loc[meqtl_df['#id'] == cpg_id, :]
        # get the max_meqtl_sites meQTLS with smallest p-value
        meqtls = meqtls.sort_values(by='p-value', ascending=True).head(max_meqtl_sites)
        return meqtls['SNP'].to_list()
            
    def _get_db_sites(
        self,
        cpg_id:str
        ) -> list:
        """
        Return the meQTL locations in either of the databases for the given CpG
        @ cpg_id: the CpG id
        @ returns: a list of the meQTLs, 'chr:start'
        """
        godmc_metqtls = self.godmc_meqtl_df.loc[self.godmc_meqtl_df['cpg'] == cpg_id, 'snp'].to_list()    
        pancan_meqtls = self.pancan_meqtl_df.loc[self.pancan_meqtl_df['cpg'] == cpg_id, 'snp'].to_list()
        return godmc_metqtls + pancan_meqtls

    def get_predictor_sites(
        self, 
        cpg_id: str,
        samples: list,
        num_correl_sites: int,
        max_meqtl_sites: int,
        nearby_window_size: int
        ) -> list:
        """
        Get the sites to be used as predictors of cpg_id's methylation
        @ cpg_id: the id of the CpG
        @ samples: the samples to be used
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
        # if for some reason this cpg is not in illumina_cpg_locs_df, return empty list of predictors
        except:
            return []
        # get num_correl_sites correlated CpGs and convert to genomic locations
        corr_cpg_ids = self._select_correl_sites(cpg_id, chrom, num_correl_sites, samples)
        corr_locs = (
            self.illumina_cpg_locs_df.loc[
                self.illumina_cpg_locs_df['#id'].isin(corr_cpg_ids)
                ].assign(location=lambda df: df['chr'] + ':' + df['start'].astype(str))['location']
            .tolist()
            )
        # get sites (and cpg_id itself position) within nearby_window_size of cpg_id
        nearby_site_locs = [chrom + ':' + str(start + i) for i in range(-int(nearby_window_size/2), int(nearby_window_size/2) + 1)]
        # get sites from databases
        db_meqtl_locs = self._get_db_sites(cpg_id)
        # get sites from matrixQTL 
        matrix_meqtl_locs = self._get_matrixQTL_sites(cpg_id, chrom, max_meqtl_sites)
        # return the union of all sources
        predictor_sites = list(set(corr_locs + nearby_site_locs + matrix_meqtl_locs + db_meqtl_locs))
        return predictor_sites
    
    def _create_training_mat(
        self, 
        cpg_id: str, 
        predictor_sites: list,
        samples: list
        ) -> tuple:
        """
        Create the training matrix for the given cpg_id and predictor_sites
        @ cpg_id: the id of the CpG
        @ predictor_sites: list of sites to be used as predictors of cpg_id's methylation
        @ samples: the samples to be included in the training matrix
        @ returns: X, y where X is the training matrix and y is the methylation values of cpg_id across samples
        """
        # for each sample, get the cpg_id methylation values
        y = self.all_methyl_age_df_t.loc[samples, cpg_id]
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
        # add one-hot-encoded gender and tissue type covariate columns
        covariate_df = self.all_methyl_age_df_t.loc[X.index, self.all_methyl_age_df_t.columns[-35:]]
        X = pd.merge(X, covariate_df, left_index=True, right_index=True)
        return X, y
    
    def evaluate_predictor(
        self, 
        cpg_id: str,
        samples: list,
        predictor_sites: list
        ) -> pd.DataFrame:
        """
        Train the predictor for one CpG
        """
        X, y = self._create_training_mat(cpg_id, predictor_sites, samples)
        
        model = ElasticNetCV(cv=3, random_state=0, max_iter=5000, n_jobs=5, selection='random')
        cv = KFold(n_splits=3, shuffle=True, random_state=0)
        # do the cross validation
        maes, r2s, feature_names = [], [], []
        for train_index, test_index in cv.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            maes.append(mean_absolute_error(y_test, preds))
            r2s.append(r2_score(y_test, preds))
            feature_names.append(X_train.columns[model.coef_ != 0].to_list())
        model2 = ElasticNetCV(cv=3, random_state=0, max_iter=5000, n_jobs=5, selection='random')
        # repeat with just the covariates
        base_maes, base_r2s, base_feature_names = [], [], []
        X = X.iloc[:, -35:]
        for train_index, test_index in cv.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model2.fit(X_train, y_train)
            preds = model2.predict(X_test)
            base_maes.append(mean_absolute_error(y_test, preds))
            base_r2s.append(r2_score(y_test, preds))
            base_feature_names.append(X_train.columns[model2.coef_ != 0].to_list())
        # create df with columns cpg_id, mae, r2, feature_names, base_mae, base_r2, base_feature_names, with entries as the lists, not expanding
        result_df = pd.DataFrame({
            'cpg_id': [cpg_id], 'mae': [maes], 'r2': [r2s], 'feature_names': [feature_names],
            'base_mae': [base_maes], 'base_r2': [base_r2s], 'base_feature_names': [base_feature_names]
            })
        return result_df
    
    def train_predictor(
        self, 
        cpg_id: str,
        samples: list,
        num_correl_sites: int,
        max_meqtl_sites: int,
        nearby_window_size: int
        ):
        """
        Build the predictor for one CpG
        @ cpg_id: the id of the CpG to predict
        @ samples: list of samples to use for training
        @ num_correl_sites: number of sites to use as predictors
        @ max_meqtl_sites: maximum number of meqtl db sites to use as predictors
        @ nearby_window_size: window size to use for choosing nearby sites
        """
        # get the sites to be used as predictors
        predictor_sites = self.get_predictor_sites(cpg_id, samples, num_correl_sites, max_meqtl_sites, nearby_window_size)
        # train the model
        X, y = self._create_training_mat(cpg_id, predictor_sites, samples)
        # train one elasticNet model to predict y from X
        model = ElasticNetCV(cv=5, random_state=0, max_iter=5000, selection = 'random')
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
        samples: list = []
        ):
        """
        Train the predictor for all CpGs
        """
        # get the list of all CpGs
        if len(cpg_ids) == 0:
            cpg_ids = self.illumina_cpg_locs_df['#id'].to_list()
        if len(samples) == 0:
            samples = self.all_methyl_age_df_t.index.to_list()
        # for each cpg, train the predictor
        for i, cpg_id in enumerate(cpg_ids):
            self.train_predictor(cpg_id, samples, num_correl_sites, max_meqtl_sites, nearby_window_size) 
            if i % 10 == 0:
                print(i)

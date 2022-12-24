import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate


class mutationClock:
    def __init__(
        self,
        all_mut_w_age_df: pd.DataFrame,
        illumina_cpg_locs_df: pd.DataFrame, 
        all_methyl_age_df_t: pd.DataFrame,
        matrix_qtl_dir: str = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/muts",
        godmc_meqtl_fn: str = "/cellar/users/zkoch/methylation_and_mutation/data/meQTL/goDMC_meQTL/goDMC_meQTLs.parquet",
        pancan_meqtl_fn: str = "/cellar/users/zkoch/methylation_and_mutation/data/meQTL/pancan_tcga_meQTL/pancan_meQTL.parquet"
        ) -> None:

        self.all_mut_w_age_df = all_mut_w_age_df
        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.all_methyl_age_df_t = all_methyl_age_df_t
        # Preprocessing: subset to only mutations that are non X and Y chromosomes and that occured in samples with measured methylation
        self.all_mut_w_age_df['mut_cpg'] = self.all_mut_w_age_df['chr'] + ':' + self.all_mut_w_age_df['start'].astype(str)
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
        
    def _select_correl_sites(
        self,
        cpg_id: str,
        cpg_chr: str,
        num_correl_sites: int
        ) -> list:
        """
        Just in time correlation to find the most correlated sites to the mutation event CpG in matched samples
        """
        # get the  CpG's MF
        cpg_mf = self.all_methyl_age_df_t.loc[:, cpg_id]
        # get the MF of all same chrom CpGs
        same_chrom_cpgs = self.illumina_cpg_locs_df.loc[
            self.illumina_cpg_locs_df['chr'] == cpg_chr, # exclude the mut_cpg
            '#id'].values
        same_chrom_cpgs_mf = self.all_methyl_age_df_t.loc[:, same_chrom_cpgs]
        # get correlation between mut_cpg and all same chrom CpGs
        corrs = same_chrom_cpgs_mf.corrwith(cpg_mf, axis=0)
        # choose the sites with largest absolute correlation
        return corrs.abs().sort_values(ascending=False).index[:num_correl_sites].to_list()
        
    def get_matrixQTL_sites(
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
        # read in the matrixQTL results for this chromosome        
        meqtl_df = pd.read_parquet(os.path.join(self.matrix_qtl_dir, f"chr{chrom}_meqtl.parquet"))
        # get the meQTLs for this CpG
        meqtls = meqtl_df.loc[meqtl_df['#id'] == cpg_id, :]
        # get the max_meqtl_sites meQTLS with smallest p-value
        meqtls = meqtls.sort_values(by='p-value', ascending=True).head(max_meqtl_sites)
        return meqtls['SNP'].to_list()
            
    def get_db_sites(
        self,
        cpg_id:str
        ) -> list:
        """
        Return the meQTL locations in either of the databases for the given CpG
        @ cpg_id: the CpG id
        @ returns: a list of the meQTLs, 'chr:start'
        """
        godmc_metqtls = self.godmc_meqtl_df.loc[self.godmc_meqtl_df['cpg'] == cpg_id, 'snp'].to_list()    
        pancan_meqtls = self.pancan_meqtl_df.loc[self.ancan_meqtl_df['cpg'] == cpg_id, 'snp'].to_list()
        return godmc_metqtls + pancan_meqtls

    def get_predictor_sites(
        self, 
        cpg_id: str,
        num_correl_sites: int,
        max_meqtl_sites: int,
        nearby_window_size: int
        ) -> list:
        """
        Get the sites to be used as predictors of cpg_id's methylation
        @ cpg_id: the id of the CpG
        @ returns: list of genomic locations of the sites to be used as predictors in format chr:start
        """
        # get cpg_id's chromosome and start position
        chrom = self.illumina_cpg_locs_df.loc[
            self.illumina_cpg_locs_df['#id'] == cpg_id, 'chr'
            ].values[0]
        start = self.illumina_cpg_locs_df.loc[
            self.illumina_cpg_locs_df['#id'] == cpg_id, 'start'
            ].values[0]
        # get num_correl_sites correlated CpGs and convert to genomic locations
        corr_cpg_ids = self._select_correl_sites(cpg_id, chrom, num_correl_sites)
        corr_locs = (
            self.illumina_cpg_locs_df.loc[
                self.illumina_cpg_locs_df['#id'].isin(corr_cpg_ids)
                ].assign(location=lambda df: df['chr'] + ':' + df['start'].astype(str))['location']
            .tolist()
            )
        # get sites (and cpg_id position) within nearby_window_size of cpg_id
        nearby_site_locs = [chrom + ':' + str(start + i) for i in range(-nearby_window_size, nearby_window_size + 1)]
        # get sites from databases
        db_meqtl_locs = self.get_db_sites(cpg_id)
        # get sites from matrixQTL 
        matrix_meqtl_locs = self.get_matrixQTL_sites(cpg_id, chrom, max_meqtl_sites)
        # return the union of the two
        predictor_sites = list(set(corr_locs + nearby_site_locs + matrix_meqtl_locs + db_meqtl_locs))
    
        return predictor_sites
    
    def train_one_predictor(
        self, 
        cpg_id: str,
        predictor_sites: list
        ) -> pd.DataFrame:
        """
        Train the predictor for one CpG
        """
        # for each sample, get the cpg_id methylation values
        y = self.all_methyl_age_df_t.loc[:, cpg_id]
        # get the mutation status of predictor sites
        mut_status = self.all_mut_w_age_df.loc[
            self.all_mut_w_age_df['mut_cpg'].isin(predictor_sites),
            ['DNA_VAF', 'case_submitter_id', 'mut_cpg']
            ]
        # create a new dataframe with columns = predictor sites, rows = y.index, and values = mut_status
        X = pd.DataFrame(index=y.index, columns=predictor_sites)
        # populate the X dataframe
        X.loc[:, predictor_sites] = mut_status.pivot(index='case_submitter_id', columns='mut_cpg', values='DNA_VAF')
        X.fillna(0, inplace=True)
        
        # train an linear regression model from sklearn to predict y from X with 5-fold cross validation
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.linear_model import ElasticNetCV
        model = ElasticNetCV(cv=5, random_state=0)
        cv = KFold(n_splits=2, shuffle=True, random_state=0)
        # do the cross validation
        for train_index, test_index in cv.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            # calculate MAE
            print(mean_absolute_error(y_test, model.predict(X_test)))
            # and r2
            print(r2_score(y_test, model.predict(X_test)))
            # print all the nonzero coefficients
            print(model.coef_[model.coef_ != 0])
            # print the number of nonzero coefficients
            print(np.count_nonzero(model.coef_))
            # print the intercept
            print(model.intercept_)
            
            
    def build_one_predictor(
        self, 
        cpg_id: str,
        num_correl_sites: int,
        nearby_window_size: int
        ) -> pd.DataFrame:
        """
        Build the predictor for one CpG
        """
        # get the sites to be used as predictors
        predictor_sites = self.get_predictor_sites(cpg_id, num_correl_sites, nearby_window_size)
        
        # train the model
        
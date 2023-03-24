import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import get_data, analysis, utils, plotting, compute_comethylation, methyl_mut_burden, mutation_features, methylation_pred

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import dask.dataframe as dd
from scipy.stats import spearmanr, pearsonr
from scipy import stats
import dask
import sklearn
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
out_dir = "/cellar/users/zkoch/methylation_and_mutation/output_dirs/030623_output"
dependency_f_dir = "/cellar/users/zkoch/methylation_and_mutation/dependency_files"
data_dir = "/cellar/users/zkoch/methylation_and_mutation/data"
corr_dir = '/cellar/users/zkoch/methylation_and_mutation/dependency_files/chr_dset_corrs'
#methylation_dir = '/cellar/users/zkoch/methylation_and_mutation/data/dropped3SD_qnormed_methylation'
methylation_dir = '/cellar/users/zkoch/methylation_and_mutation/data/processed_methylation'
icgc_dir = "/cellar/users/zkoch/methylation_and_mutation/data/icgc"

illumina_cpg_locs_df, all_mut_df, all_methyl_df, all_methyl_df_t, all_meta_df, dataset_names_list = get_data.main(
    illum_cpg_locs_fn = os.path.join(dependency_f_dir, "illumina_cpg_450k_locations.csv"),
    out_dir = out_dir,
    methyl_dir = methylation_dir,
    mut_fn = os.path.join(data_dir, "PANCAN_mut.tsv.gz"),
    meta_fn = os.path.join(data_dir, "PANCAN_meta.tsv")
    )
# read in other already computed files
mut_in_measured_cpg_w_methyl_age_df = pd.read_parquet(os.path.join(dependency_f_dir, "mut_in_measured_cpg_w_methyl_age_df_5year.parquet"))
all_mut_w_age_df, all_methyl_age_df_t = utils.add_ages_to_mut_and_methyl(all_mut_df, all_meta_df, all_methyl_df_t)


samples_w_mut_and_methyl = set(all_methyl_age_df_t.index) & set(all_mut_w_age_df['case_submitter_id'])

methyl = all_methyl_age_df_t.loc[samples_w_mut_and_methyl]
mut = all_mut_w_age_df.loc[all_mut_w_age_df['case_submitter_id'].isin(samples_w_mut_and_methyl)]

mut['mut_loc'] = mut['chr'].astype(str) + ':' + mut['start'].astype(str)
# rename DNA_VAF to MAF
mut = mut.rename(columns={'DNA_VAF': 'MAF', 'case_submitter_id': 'sample'})
# drop columns
mut.drop(columns=['end','reference', 'alt',  'dataset_r', 'gender_r', 'age_at_index', 'gender', 'dataset', 'mutation'], inplace=True)
mut.reset_index(drop=True, inplace=True)
mut.to_parquet('/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/tcga_muts.parquet')


covariates = all_methyl_age_df_t.loc[samples_w_mut_and_methyl, ['age_at_index', 'dataset', 'gender']].T
covariates = covariates.reindex(sorted(covariates.columns), axis=1)
# map gender row to 0, 1
covariates.loc['gender'] = covariates.loc['gender'].map({'FEMALE': 0, 'MALE': 1})
# map each unique dataset value to a unique integer
dataset_map = {dataset: i for i, dataset in enumerate(covariates.loc['dataset'].unique())}
covariates.loc['dataset'] = covariates.loc['dataset'].map(dataset_map)
covariates.to_csv('/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/tcga_covariates.csv.gz', compression='gzip')

print("Starting methyl", flush=True)
methyl.drop(columns = ['age_at_index', 'dataset', 'gender'], inplace=True)
methyl = methyl.T
methyl = methyl.reindex(sorted(methyl.columns), axis=1)
methyl.to_parquet('/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/tcga_methyl.parquet')
methyl.to_csv('/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/tcga_methyl.csv.gz', compression='gzip')


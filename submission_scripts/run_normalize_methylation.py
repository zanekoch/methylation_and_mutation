import get_data, utils
import os
import pandas as pd
import numpy as np
import dask
import pyarrow


out_dir = "/cellar/users/zkoch/methylation_and_mutation/output_120522"
dependency_f_dir = "/cellar/users/zkoch/methylation_and_mutation/dependency_files"
data_dir = "/cellar/users/zkoch/methylation_and_mutation/data"
corr_dir = '/cellar/users/zkoch/methylation_and_mutation/dependency_files/chr_dset_corrs'

illumina_cpg_locs_df, all_mut_df, all_methyl_df, all_methyl_df_t, all_meta_df, dataset_names_list = get_data.main(
    os.path.join(dependency_f_dir, "illumina_cpg_450k_locations.csv"),
    out_dir,
    os.path.join(data_dir, "processed_methylation"),
    os.path.join(data_dir, "PANCAN_mut.tsv.gz"),
    os.path.join(data_dir, "PANCAN_meta.tsv")
    )

# get mean of each column of all_methyl_df
sample_mean_methyl = all_methyl_df.mean(axis=0)
# compute the mean and standard deviation of the data
mean = sample_mean_methyl.mean()
std = sample_mean_methyl.std()
# drop rows that have values greater than 3 standard deviations from the mean
sample_mean_methyl_dropped = sample_mean_methyl[np.abs(sample_mean_methyl - mean) <= 3 * std]
# get the samples in sample_mean_methyl that are not in sample_mean_methyl_dropped
dropped_samples = sample_mean_methyl[~sample_mean_methyl.index.isin(sample_mean_methyl_dropped.index)].index.to_list()
print("dropped_samples: ", len(dropped_samples))
all_methyl_df_dropped = all_methyl_df[sample_mean_methyl_dropped.index]
# do quantile normalization
normed_df = utils.quantileNormalize(all_methyl_df_dropped)
print("did quantile normalization")
# convert to dask df
normed_methyl_dd = dask.dataframe.from_pandas(normed_df, npartitions=300)
# write to directory
normed_methyl_dd.to_parquet("/cellar/users/zkoch/methylation_and_mutation/data/dropped3SD_qnormed_methylation")
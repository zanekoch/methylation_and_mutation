import get_data, utils, comethylation_distance
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import dask.dataframe as dd
from scipy.stats import spearmanr
from rich.progress import track

out_dir = "/cellar/users/zkoch/methylation_and_mutation/output_110422"
dependency_f_dir = "/cellar/users/zkoch/methylation_and_mutation/dependency_files"
data_dir = "/cellar/users/zkoch/methylation_and_mutation/data"

all_illumina_cpg_locs_df = pd.read_csv(os.path.join(dependency_f_dir, "illumina_cpg_450k_locations.csv"), sep=',', dtype={'CHR': str}, low_memory=False)
# get rows of all_illumina_cpg_locs_df where 'exon' appears in UCSC_RefGene_Group
all_illumina_cpg_locs_df.dropna(subset=['UCSC_RefGene_Group'], inplace=True)
exon_cpg_locs_df = all_illumina_cpg_locs_df[all_illumina_cpg_locs_df['UCSC_RefGene_Group'].str.contains('Body')]
cpg_in_body = exon_cpg_locs_df['Name'].to_list()

illumina_cpg_locs_df, all_mut_df, all_methyl_df, all_methyl_df_t, all_meta_df, dataset_names_list = get_data.main(os.path.join(dependency_f_dir, "illumina_cpg_450k_locations.csv"), 
    out_dir,
    os.path.join(data_dir, "processed_methylation"),
    os.path.join(data_dir, "PANCAN_mut.tsv.gz"),
    os.path.join(data_dir, "PANCAN_meta.tsv"))

# add ages to all_methyl_df_t
all_mut_w_age_df, all_methyl_age_df_t = utils.add_ages_to_mut_and_methyl(all_mut_df, all_meta_df, all_methyl_df_t)
all_methyl_age_df_t.drop(columns=['gender'], inplace=True)
all_mut_w_age_df.drop(columns=['gender'], inplace=True)

mut_scan_distance = comethylation_distance.mutationScanDistance(all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, age_bin_size = 5, max_dist = 100000)

mut_scan_distance.preproc_correls(out_dir = '/cellar/users/zkoch/methylation_and_mutation/dependency_files/chr_dset_corrs')
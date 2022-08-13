import comethylation, get_data, analysis, utils
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
import pickle

# set args
illum_cpg_locs_fn = "/cellar/users/zkoch/methyl_mut_proj/data/illumina_cpg_450k_locations.csv"
out_dir = "/cellar/users/zkoch/methylation_and_mutation/output_080422"
data_dirs = glob.glob( os.path.join("/cellar/users/zkoch/methyl_mut_proj/data", "tcga*data"))

illumina_cpg_locs_df, all_mut_df, all_methyl_df, all_methyl_df_t, all_meta_df, run_name, dataset_names_list = get_data.main(illum_cpg_locs_fn, out_dir, data_dirs)


# only keep sites on chr 1
chr_methyl_df = all_methyl_df.loc[all_methyl_df.index.isin(illumina_cpg_locs_df[illumina_cpg_locs_df.chr == "1"]['#id'].to_list())]
# only keep ages that match to columns of chr_methyl_df
all_meta_df = all_meta_df.loc[all_meta_df.index.isin(chr_methyl_df.columns)]
# only keep columns that have ages
chr_methyl_df = chr_methyl_df[chr_methyl_df.columns[chr_methyl_df.columns.isin(all_meta_df.index)]]
# put samples in same order as methylation
all_meta_df = all_meta_df.loc[chr_methyl_df.columns]

# read in ct_mut_in_measured_cpg_w_methyl_df
ct_mut_in_measured_cpg_w_methyl_df = pd.read_parquet(os.path.join(out_dir, "ct_mut_in_measured_cpg_w_methyl_df.parquet"))
# correlations of each CpG on chr1 with age
ewas_df = pd.read_parquet(os.path.join(out_dir, "all_chr1_EWAS_results.parquet"))
ewas_df.index = chr_methyl_df.index
# sort
ewas_by_abs = np.abs(ewas_df).sort_values(by = 'pearson_corrs')
ewas_df = ewas_df.sort_values(by = 'pearson_corrs') # GOTTA SORT!!

# co-methylation for all CpGs on chr1
chrom_one_corr = pd.read_parquet(os.path.join(out_dir, "all_corrs_chrom1.parquet"))

print("starting analysis", flush=True)
# we only care about mutations in samples with some methylation (doesn't have to be 
all_mut_w_methyl_df = all_mut_df[all_mut_df['sample'].isin(all_methyl_df.columns.to_list())]
# instead of counting nearby mutations, count mutations that are in linked CpG sites
cpg_sample_linked_mut_count_df = comethylation.count_linked_mutations(cpgs_to_count_df = ewas_by_abs.iloc[:1000],
                       all_mut_w_methyl_df = all_mut_w_methyl_df,
                       illumina_cpg_locs_df = illumina_cpg_locs_df,
                       all_methyl_df = all_methyl_df, 
                       corr_df = chrom_one_corr, 
                       num_sites=10,
                       max_dist = 1000,
                       percentile_cutoff=.9999)
cpg_sample_linked_mut_count_df.to_parquet(os.path.join(out_dir, "cpg_sample_linked_mut_count_lowest_1000_tenLinkedSites_1kbp_df.parquet"))
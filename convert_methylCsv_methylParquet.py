import get_data, analysis, utils, methyl_mut_burden
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

illumina_cpg_locs_df, all_mut_df, _, _, all_meta_df, dataset_names_list = get_data.main(os.path.join(dependency_f_dir, "illumina_cpg_450k_locations.csv"),
                                                                                                                  out_dir,
                                                                                                                  os.path.join(data_dir, "processed_methylation"),
                                                                                                                  os.path.join(data_dir, "PANCAN_mut.tsv.gz"),
                                                                                                                  os.path.join(data_dir, "PANCAN_meta.tsv"))


methyl_fn = "/cellar/users/zkoch/methylation_and_mutation/data/PANCAN_methyl.tsv.gz"

get_data.preprocess_methylation(methyl_fn, all_meta_df, illumina_cpg_locs_df, out_dir = "/cellar/users/zkoch/methylation_and_mutation/data/processed_methylation_noDrop")
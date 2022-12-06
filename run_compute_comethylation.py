import compute_comethylation, get_data, utils
import os
import pandas as pd


out_dir = "/cellar/users/zkoch/methylation_and_mutation/output_120522"
dependency_f_dir = "/cellar/users/zkoch/methylation_and_mutation/dependency_files"
data_dir = "/cellar/users/zkoch/methylation_and_mutation/data"
methylation_dir = '/cellar/users/zkoch/methylation_and_mutation/data/dropped3SD_qnormed_methylation'
corr_dir = '/cellar/users/zkoch/methylation_and_mutation/dependency_files/chr_dset_corrs'

illumina_cpg_locs_df, all_mut_df, all_methyl_df, all_methyl_df_t, all_meta_df, dataset_names_list = get_data.main(
    illum_cpg_locs_fn = os.path.join(dependency_f_dir, "illumina_cpg_450k_locations.csv"),
    out_dir = out_dir,
    methyl_dir = methylation_dir,
    mut_fn = os.path.join(data_dir, "PANCAN_mut.tsv.gz"),
    meta_fn = os.path.join(data_dir, "PANCAN_meta.tsv")
    )

# add ages to all_methyl_df_t
all_mut_w_age_df, all_methyl_age_df_t = utils.add_ages_to_mut_and_methyl(all_mut_df, all_meta_df, all_methyl_df_t)

mut_scan = compute_comethylation.mutationScan(
    all_mut_w_age_df, illumina_cpg_locs_df, 
    all_methyl_age_df_t, 
    corr_dir = corr_dir, 
    age_bin_size = 5, max_dist = 1000,
    num_correl_sites = 1000, num_background_events = 0
    )

comparison_sites_df, all_metrics_df = mut_scan.look_for_disturbances(
    min_VAF_percentile = 75, 
    linkage_method='correl', 
    out_dir = out_dir, 
    corr_direction = 'pos'
    )

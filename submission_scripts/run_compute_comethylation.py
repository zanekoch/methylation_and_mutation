import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import compute_comethylation, get_data, utils
import os

out_dir = "/cellar/users/zkoch/methylation_and_mutation/output_dirs/021323_comethylation_disturbance_output"
dependency_f_dir = "/cellar/users/zkoch/methylation_and_mutation/dependency_files"
data_dir = "/cellar/users/zkoch/methylation_and_mutation/data"
methylation_dir = '/cellar/users/zkoch/methylation_and_mutation/data/dropped3SD_qnormed_methylation'
corr_dir = '/cellar/users/zkoch/methylation_and_mutation/dependency_files/chr_dset_corrs'

start_num_mut_to_process = 2500
end_num_mut_to_process = 5000
corr_direction = 'neg'
print(f"Running comethylation disturbance analysis with {start_num_mut_to_process} to {end_num_mut_to_process} mutations, {corr_direction} correlation direction and writing to {out_dir}", flush=True)

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
    all_methyl_age_df_t, corr_dir = corr_dir,
    age_bin_size = 5, max_dist = 1000,
    num_correl_sites = 100, num_background_events = 0,
    matched_sample_num = 20
    )

comparison_sites_df, all_metrics_df = mut_scan.look_for_disturbances(
    start_num_mut_to_process = start_num_mut_to_process,
    end_num_mut_to_process = end_num_mut_to_process,
    linkage_method='correl', 
    out_dir = out_dir, 
    corr_direction = corr_direction
    )

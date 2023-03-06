import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import compute_comethylation, get_data, utils
import os


out_dir = "/cellar/users/zkoch/methylation_and_mutation/output_dirs/022723_comethylation_disturbance_output/bigger_run"
dependency_f_dir = "/cellar/users/zkoch/methylation_and_mutation/dependency_files"
data_dir = "/cellar/users/zkoch/methylation_and_mutation/data"
methylation_dir = '/cellar/users/zkoch/methylation_and_mutation/data/dropped3SD_qnormed_methylation'
corr_dir = '/cellar/users/zkoch/methylation_and_mutation/dependency_files/chr_dset_corrs'

# want 1000 mutevents
# 10 jobs of 100
# 100 with 10 background => (10 with 10 background) * 10 = 20*10 = 200 min = 3.3 hours
# => with 50 background = 16.5 hours
start_num_mut_to_process = int(sys.argv[1])
end_num_mut_to_process = int(sys.argv[2])
linkage_method = 'dist'
print(f"Running comethylation disturbance analysis with {start_num_mut_to_process} to {end_num_mut_to_process} mutations, {linkage_method} linkage method and writing to {out_dir}", flush=True)

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
    age_bin_size = 5, max_dist = 2500,
    num_correl_sites = 100, num_background_events = 100,
    matched_sample_num = 50
    )

comparison_sites_df_test, all_metrics_df_test = mut_scan.look_for_disturbances(
    start_num_mut_to_process = start_num_mut_to_process,
    end_num_mut_to_process = end_num_mut_to_process,
    linkage_method = linkage_method, 
    out_dir = out_dir, 
    corr_direction = 'pos'
    )

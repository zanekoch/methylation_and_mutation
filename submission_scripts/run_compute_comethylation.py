import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import compute_comethylation, get_data, utils
import os
import pandas as pd
import sys


dependency_f_dir = "/cellar/users/zkoch/methylation_and_mutation/dependency_files"
data_dir = "/cellar/users/zkoch/methylation_and_mutation/data"
methylation_dir = '/cellar/users/zkoch/methylation_and_mutation/data/dropped3SD_qnormed_methylation'
#methylation_dir = '/cellar/users/zkoch/methylation_and_mutation/data/processed_methylation'
corr_dir = '/cellar/users/zkoch/methylation_and_mutation/dependency_files/chr_dset_corrs'

######## TO SET #########

meqtl_fn = str(sys.argv[1])
print(f"Using meqtl file {meqtl_fn}", flush=True)
out_dir = str(sys.argv[2])

start_num_mut_to_process = int(sys.argv[3])
end_num_mut_to_process = int(sys.argv[4])
linkage_method = str(sys.argv[5])

# make out_dir if it doesn't exist
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

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
meqtls = pd.read_parquet(meqtl_fn)

mut_scan = compute_comethylation.mutationScan(
    all_mut_w_age_df, illumina_cpg_locs_df, 
    all_methyl_age_df_t, corr_dir = corr_dir,
    age_bin_size = 10, max_dist = 500,
    num_correl_sites = 10, num_background_events = 100,
    matched_sample_num = 20
    )

comparison_sites_df_test, all_metrics_df_test = mut_scan.look_for_disturbances(
    start_num_mut_to_process = start_num_mut_to_process,
    end_num_mut_to_process = end_num_mut_to_process,
    linkage_method = linkage_method, 
    out_dir = out_dir, 
    corr_direction = 'pos',
    meqtl_db_df = meqtls
    )

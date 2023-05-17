import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import get_data, utils
import os

out_dir = "/cellar/users/zkoch/methylation_and_mutation/data/processed_methylation_noDropNaN_qnorm_new"
dependency_f_dir = "/cellar/users/zkoch/methylation_and_mutation/dependency_files"
data_dir = "/cellar/users/zkoch/methylation_and_mutation/data"


print("getting illumina and metadata", flush=True)
illumina_cpg_locs_df = get_data.get_illum_locs(
    illum_cpg_locs_fn = os.path.join(dependency_f_dir, "illumina_cpg_450k_locations.csv")
    )
all_meta_df, dataset_names_list = get_data.get_metadata(
    meta_fn = os.path.join(data_dir, "PANCAN_meta.tsv")
    )

print("starting conversion", flush=True)
utils.preprocess_methylation(
    methyl_fn = "/cellar/users/zkoch/methylation_and_mutation/data/PANCAN_methyl.tsv.gz",
    all_meta_df = all_meta_df, 
    illumina_cpg_locs_df = illumina_cpg_locs_df,
    out_dir = out_dir
    )
print("done")
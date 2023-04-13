import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import compute_comethylation
import os
import pandas as pd
import sys
import glob
import dask.dataframe as dd

### TO SET ###
out_dir = "/cellar/users/zkoch/methylation_and_mutation/output_dirs/032423_comethyl_output/correl_based_1000Top_no_mutMF"
#paths = glob.glob("/cellar/users/zkoch/methylation_and_mutation/output_dirs/032423_comethyl_output/distance_based_100kbMax/all_metrics*")
#c_paths = glob.glob("/cellar/users/zkoch/methylation_and_mutation/output_dirs/032423_comethyl_output/distance_based_100kbMax/comparison_sites*")
paths = glob.glob("/cellar/users/zkoch/methylation_and_mutation/output_dirs/032423_comethyl_output/correl_based_1000Top_no_mutMF/all_metrics*")
c_paths = glob.glob("/cellar/users/zkoch/methylation_and_mutation/output_dirs/032423_comethyl_output/correl_based_1000Top_no_mutMF/comparison_sites*")

# the int of path to the metrics file
i = int(sys.argv[1])

# initialize analyzeComethylation object
analyze_comethylation = compute_comethylation.analyzeComethylation()

print("reading in", paths[i], flush=True)
one_metrics_dd = dd.read_parquet(paths[i])
one_metrics_df = one_metrics_dd.compute()
print("done reading in", paths[i], flush=True)
# get mean metrics for distances 
one_mean_metrics = analyze_comethylation.get_mean_metrics_by_dist(
    one_metrics_df, 
    absolute_distances = [5, 10, 50, 100, 500, 1000]
    #absolute_distances = [100, 500, 1000, 5000, 10000, 50000, 100000]
)   
print("done with mean methyl", paths[i], flush=True)

# get muts number
muts_num = paths[i].split("/")[-1].split("_")[2]

# write to out_dir
one_mean_metrics.to_parquet(os.path.join(out_dir, f"mean_metrics_by_dist_actual_corr_{muts_num}.parquet"))
print(f'Wrote to {os.path.join(out_dir, f"mean_metrics_by_dist_actual_corr_{muts_num}.parquet")}', flush=True)
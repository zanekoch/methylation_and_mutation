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
paths = glob.glob("/cellar/users/zkoch/methylation_and_mutation/output_dirs/032423_comethyl_output/correl_based_1000Top_no_mutMF/all_metrics*")
c_paths = glob.glob("/cellar/users/zkoch/methylation_and_mutation/output_dirs/032423_comethyl_output/correl_based_1000Top_no_mutMF/comparison_sites*")

# read in all metrics files
all_metrics_dfs = []
for path in paths:
    # read in metrics sites from dask paruet
    one_metrics_dd = dd.read_parquet(path)
    one_metrics_df = one_metrics_dd.compute()
    all_metrics_dfs.append(one_metrics_df)
    print("done with", path, flush=True)
all_metrics_df = pd.concat(all_metrics_dfs)
all_metrics_df.reset_index(inplace=True, drop=True)

print("Read in all metrics files", flush=True)

"""all_comparison_site_dfs = []
for path in c_paths:
    # read in metrics sites from dask paruet
    one_comp_dd = dd.read_parquet(path)
    one_comp_df = one_comp_dd.compute()
    all_comparison_site_dfs.append(one_comp_df)
all_comparison_site_df = pd.concat(all_comparison_site_dfs)
all_comparison_site_df.reset_index(inplace=True, drop=True)"""

# initialize analyzeComethylation object
analyze_comethylation = compute_comethylation.analyzeComethylation()
# get mean metrics for distances 
mean_metrics = analyze_comethylation.get_mean_metrics_by_dist(
    all_metrics_df, 
    absolute_distances = [5, 10, 50, 100, 500, 1000]
    #absolute_distances = [100, 500, 1000, 5000, 10000, 50000, 100000]
)
# write to out_dir
mean_metrics.to_parquet(os.path.join(out_dir, "mean_metrics_by_dist.parquet"))
print(f"Wrote to {os.path.join(out_dir, 'mean_metrics_by_dist.parquet')}", flush=True)
import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import compute_comethylation
import os
import pandas as pd
import sys
import glob
import dask.dataframe as dd

### TO SET ###


comp_site_type = sys.argv[1]
if comp_site_type == 'dist':
    paths = glob.glob("/cellar/users/zkoch/methylation_and_mutation/output_dirs/032423_comethyl_output/distance_based_100kbMax/all_metrics*")
    c_paths = glob.glob("/cellar/users/zkoch/methylation_and_mutation/output_dirs/032423_comethyl_output/distance_based_100kbMax/comparison_sites*")
    out_dir = "/cellar/users/zkoch/methylation_and_mutation/output_dirs/032423_comethyl_output/distance_based_100kbMax"
    all_together = False
    absolute_distances = [100, 500, 1000, 5000, 10000, 50000, 100000]
elif comp_site_type == 'corr':
    paths = glob.glob("/cellar/users/zkoch/methylation_and_mutation/output_dirs/032423_comethyl_output/correl_based_1000Top_no_mutMF/all_metrics*")
    c_paths = glob.glob("/cellar/users/zkoch/methylation_and_mutation/output_dirs/032423_comethyl_output/correl_based_1000Top_no_mutMF/comparison_sites*")
    out_dir = "/cellar/users/zkoch/methylation_and_mutation/output_dirs/032423_comethyl_output/correl_based_1000Top_no_mutMF"
    all_together = False
    absolute_distances = [5, 10, 50, 100, 500, 1000]
else:
    # error
    print("Error: comp_site_type must be 'dist' or 'corr'", flush=True)

# sort paths in place
paths.sort(key=lambda x: int(x.split('/')[-1].split('Muts')[0].split('all_metrics_')[1].split('-')[0]))

# initialize analyzeComethylation object
analyze_comethylation = compute_comethylation.analyzeComethylation()

# iterate across each path, get mean metrics, and combine
if all_together: 
    all_mean_metrics_l = []
    for path in paths:
        print("reading in", path, flush=True)
        one_metrics_dd = dd.read_parquet(path)
        one_metrics_df = one_metrics_dd.compute()
        print("done reading in", path, flush=True)
        # get mean metrics for distances 
        one_mean_metrics = analyze_comethylation.get_mean_metrics_by_dist(
            one_metrics_df, 
            absolute_distances = absolute_distances
        )   
        print("done with mean methyl", path, flush=True)
        all_mean_metrics_l.append(one_mean_metrics)
    all_mean_metrics = pd.concat(all_mean_metrics_l)
    # write to out_dir
    all_mean_metrics.to_parquet(os.path.join(out_dir, "mean_metrics_by_dist_w_weighted.parquet"))
else: # get mean metrics for each path separately
    # the int of path to the metrics file
    i = int(sys.argv[2])
    print("reading in", paths[i], flush=True)
    one_metrics_dd = dd.read_parquet(paths[i])
    one_metrics_df = one_metrics_dd.compute()
    print("done reading in", paths[i], flush=True)
    # get mean metrics for distances 
    one_mean_metrics = analyze_comethylation.get_mean_metrics_by_dist(
        one_metrics_df, 
        absolute_distances = absolute_distances
    )   
    print("done with mean methyl", paths[i], flush=True)
    # get muts number to base name on
    muts_num = paths[i].split("/")[-1].split("_")[2]
    # write to out_dir
    one_mean_metrics.to_parquet(os.path.join(out_dir, f"mean_metrics_by_dist_w_weighted_{muts_num}.parquet"))
    print(f'Wrote to {os.path.join(out_dir, f"mean_metrics_by_dist_w_weighted_{muts_num}.parquet")}', flush=True)
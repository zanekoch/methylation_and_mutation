import glob
import dask.dataframe as dd
import pandas as pd
import argparse
import glob

paths = glob.glob("/cellar/users/zkoch/methylation_and_mutation/output_dirs/032423_comethyl_output/correl_based_1000Top_no_mutMF/all_metrics*")

def combine_files(
    all_metrics_l: list,
    output_fn: str
    ):
    all_metrics_dfs = []
    for all_metrics_fn in all_metrics_l:
        # read in metrics sites from dask paruet
        one_metrics_dd = dd.read_parquet(all_metrics_fn, 
                                        columns = ['delta_mf_median', 'mutated_sample', 'mut_event', 'is_background', 'index_event', 'measured_site', 'measured_site_dist']
                                        )
        one_metrics_df = one_metrics_dd.compute()
        one_metrics_df = one_metrics_df.loc[one_metrics_df["mutated_sample"] == True]
        all_metrics_dfs.append(one_metrics_df)
        print("finsihed with {}".format(all_metrics_fn), flush = True)
    corr_all_metrics_df = pd.concat(all_metrics_dfs)
    print("finished combining all metrics files", flush = True)
    
    corr_all_metrics_df.reset_index(inplace=True, drop=True)
    corr_all_metrics_df.to_parquet(output_fn)
    print("finished writing to {}".format(output_fn), flush = True)

def main():
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_metrics_glob", type=str, help="path to all_metrics parquet file")
    parser.add_argument("--output_fn", type=str, help="path to output parquet file")
    # prase
    args = parser.parse_args()
    all_metrics_glob = args.all_metrics_glob
    output_fn = args.output_fn
    print("reading {} and writing to {}".format(all_metrics_glob, output_fn), flush = True)
    all_metrics_list = glob.glob(all_metrics_glob)
    
    combine_files(all_metrics_list, output_fn)
    
if __name__ == "__main__":
    main()
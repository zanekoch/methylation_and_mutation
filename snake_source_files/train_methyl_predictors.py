import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import get_data, utils, somatic_mut_clock
import argparse
import os
import pandas as pd

def train_predictors(
    out_dir: str,
    cpg_start: int,
    cpg_end: int,
    all_mut_w_age_df: pd.DataFrame,
    illumina_cpg_locs_df: pd.DataFrame, 
    all_methyl_age_df_t: pd.DataFrame,
    training_samples: list
    ) -> None:
    """
    Train all the predictors
    @ output_dir: directory to save output files
    @ cpg_start: start index in mut_clock.illumina_cpg_locs of cpg_ids to train on
    @ cpg_end: end index in mut_clock.illumina_cpg_locs of cpg_ids to train on
    """
    mut_clock = somatic_mut_clock.mutationClock(
        all_mut_w_age_df, 
        illumina_cpg_locs_df, 
        all_methyl_age_df_t,
        out_dir
        )
    cpg_ids = mut_clock.illumina_cpg_locs_df.iloc[cpg_start:cpg_end]['#id'].to_list()
    mut_clock.train_all_predictors(
        num_correl_sites = 1000, max_meqtl_sites = 1000,
        nearby_window_size = 5000, cpg_ids = cpg_ids, train_samples = training_samples
        )
    # create an empty file in out_dir called cpg_end.txt
    with open(os.path.join(out_dir, f"{cpg_start}_{cpg_end}.txt"), "w") as f:
        f.write("done")

def read_data() -> tuple:
    # read in data
    out_dir = "./output_dirs/output_120522"
    dependency_f_dir = "./dependency_files"
    data_dir = "./data"
    methylation_dir = '/cellar/users/zkoch/methylation_and_mutation/data/dropped3SD_qnormed_methylation'
    illumina_cpg_locs_df, all_mut_df, all_methyl_df, all_methyl_df_t, all_meta_df, dataset_names_list = get_data.main(
        illum_cpg_locs_fn = os.path.join(dependency_f_dir, "illumina_cpg_450k_locations.csv"),
        out_dir = out_dir,
        methyl_dir = methylation_dir,
        mut_fn = os.path.join(data_dir, "PANCAN_mut.tsv.gz"),
        meta_fn = os.path.join(data_dir, "PANCAN_meta.tsv")
        )
    # add ages to all_methyl_df_t
    all_mut_w_age_df, all_methyl_age_df_t = utils.add_ages_to_mut_and_methyl(all_mut_df, all_meta_df, all_methyl_df_t)
    return all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t

def main():
    # parse arguments cpg_start, cpg_end, and out_dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpg_start', type=int, help='cpg start')
    parser.add_argument('--cpg_end', type=int, help='cpg end')
    parser.add_argument('--out_dir', type=str, help='output directory')
    parser.add_argument('--train_samples_fn', type=str, help='fn containing train samples')
    args = parser.parse_args()
    cpg_start = args.cpg_start
    cpg_end = args.cpg_end
    out_dir = args.out_dir
    train_samples_fn = args.train_samples_fn
    print(f"Training methylation predictors {cpg_start} to {cpg_end}")
    # read training samples from file and convert to list
    with open(train_samples_fn, "r") as f:
        training_samples = f.read().splitlines()
    
    all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t = read_data()

    train_predictors(
        out_dir, cpg_start, cpg_end, all_mut_w_age_df, 
        illumina_cpg_locs_df, all_methyl_age_df_t, training_samples
        )
    
if __name__ == "__main__":
    main()
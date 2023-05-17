import os
import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import get_data, mutation_features, somatic_mut_clock
import argparse
import glob


def create_somage_obj(
    somage_path, 
    directory_glob, 
    file_suffix, 
    mut_feat, 
    illumina_cpg_locs_df, 
    all_methyl_age_df_t,
    out_dir
    ):
    """
    Given a somage path, directory glob, and file suffix, create a somage object from 
    the files in the directory
    """
    print("Creating soMage object", flush = True)
    predicted_methyl_fns = glob.glob(
        os.path.join(somage_path, directory_glob, f"methyl_predictions_{file_suffix}.parquet")
        )
    predicted_perf_fns = glob.glob(
        os.path.join(somage_path, directory_glob, f"prediction_performance_{file_suffix}.parquet")
        )
    trained_model_fns = glob.glob(
        os.path.join(somage_path, directory_glob, f"trained_models_{file_suffix}.pkl")
        )
    feature_mat_fns = glob.glob(
        os.path.join(somage_path, directory_glob, "*features.pkl")
        )
    
    somage = somatic_mut_clock.mutationClock(
            predicted_methyl_fns = predicted_methyl_fns, 
            predicted_perf_fns = predicted_perf_fns,
            all_methyl_age_df_t = all_methyl_age_df_t,
            illumina_cpg_locs_df = illumina_cpg_locs_df,
            output_dir = out_dir,
            train_samples = mut_feat.train_samples,
            test_samples = mut_feat.test_samples,
            tissue_type = "",
            trained_models_fns = trained_model_fns,
            feature_mat_fns = feature_mat_fns,
        )
    return somage
    

def train_clocks(
    somage: somatic_mut_clock.mutationClock,
    out_dir: str,
    ) -> None:
    print("Training clock", flush = True)
    
    # get performance by dataset
    somage.performance_by_dataset()
    # write to output dir
    print(f"writing performance by dataset to {os.path.join(out_dir, f'performance_by_dataset.parquet')}")
    somage.performance_by_dataset_df.to_parquet(
        os.path.join(out_dir, f"performance_by_dataset.parquet")
        )
    # scan across hyperparameters and train clocks
    # 86min fo
    best_clock_scan_results = somage.scan_for_best_clock(
        datasets = somage.performance_by_dataset_df['dataset'].unique().tolist(),
        cpg_choosing_metrics =['train_AvP_methyl_spearman', 'train_Pmethyl_v_Age_spearman_abs', 'train_AvP_methyl_mi', 'train_Pmethyl_v_Age_mi'],
        number_of_cpgs = [10, 50, 500, 2500, 10000, 25000],
        training_methylation_types = ['pred', 'actual'],
        train_sets = ['train'],
        model_types = ['xgboost'],
        train_tissues = ['self']
    )
    print(f"writing best clock scan results to {os.path.join(out_dir, f'best_clock_scan_results.pkl')}")
    best_clock_scan_results.to_pickle(
        os.path.join(out_dir, f"best_clock_scan_results.pkl")
    )
    
def output_methyl_and_dset_perf(
    somage: somatic_mut_clock.mutationClock,
    out_dir: str,
    cv_num: int
    ):
    print("Writing out predicted methylation and performance", flush = True)
    # write out predicted methylation 
    somage.predicted_methyl_df.to_parquet(
        os.path.join(out_dir, f"predicted_methyl_df_CV{cv_num}.parquet")
        )
    # caclulate performance by dataset 
    somage.performance_by_dataset()
    somage.performance_by_dataset_df.to_parquet(
        os.path.join(out_dir, f"performance_by_dataset_CV{cv_num}.parquet")
        )
    

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--somage_path", type=str, 
        help="path to a directory of somage directories including feature matrices, predicted methylations, trained models, and predicted predicted performances"
        )
    parser.add_argument(
        "--directory_glob", type=str, 
        help = "glob to find directories in somage_path"
        )
    parser.add_argument(
        "--file_suffix", type=str,
        help = "suffix to find files in somage_path/directory_glob"
        )
    parser.add_argument(
        '--cross_val', type=int, 
        help='cross val fold number'
        )
    parser.add_argument(
        '--do', type=str, 
        help='train_clocks, output_methyl_and_dset_perf'
        )
    parser.add_argument(
        '--out_dir', type=str, 
        help='output directory'
        )
    # parse arguments
    args = parser.parse_args()
    somage_path = args.somage_path
    directory_glob = args.directory_glob
    file_suffix = args.file_suffix
    cv_num = args.cross_val
    do = args.do
    out_dir = args.out_dir
    # read in data
    all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, matrix_qtl_dir, covariate_fn  = get_data.read_tcga_data()
    # create mut_feat object
    mut_feat = mutation_features.mutationFeatures(
        all_mut_w_age_df = all_mut_w_age_df, illumina_cpg_locs_df = illumina_cpg_locs_df, 
        all_methyl_age_df_t = all_methyl_age_df_t, out_dir = "", 
        consortium = 'TCGA', dataset = '', cross_val_num = cv_num,
        matrix_qtl_dir = matrix_qtl_dir,
        covariate_fn = covariate_fn
        )
    # create somage object
    somage = create_somage_obj(
        somage_path = somage_path, 
        directory_glob = directory_glob, 
        file_suffix = file_suffix, 
        mut_feat = mut_feat, 
        illumina_cpg_locs_df = illumina_cpg_locs_df, 
        all_methyl_age_df_t = all_methyl_age_df_t,
        out_dir = out_dir
        )
    # do stuff
    if do == "train_clocks":
        train_clocks(somage)
    elif do == "output_methyl_and_dset_perf":
        output_methyl_and_dset_perf(somage, out_dir, cv_num)
        
    return

if __name__ == '__main__':
    main()
import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import get_data, utils, methylation_pred, mutation_features
import argparse
import os
import pandas as pd
import numpy as np
import dask.dataframe as dd
import glob

def read_icgc_data() -> tuple:
    dependency_f_dir = "/cellar/users/zkoch/methylation_and_mutation/dependency_files"
    icgc_mut_df = pd.read_parquet("/cellar/users/zkoch/methylation_and_mutation/data/icgc/for_matrixQTL/icgc_mut_df.parquet")
    icgc_meta_df = pd.read_parquet("/cellar/users/zkoch/methylation_and_mutation/data/icgc/for_matrixQTL/icgc_meta_df.parquet")
    illumina_cpg_locs_df = get_data.get_illum_locs(os.path.join(dependency_f_dir, "illumina_cpg_450k_locations.csv"))
    # read in methyl from dask dir
    #icgc_methyl_dd = dd.read_parquet('/cellar/users/zkoch/methylation_and_mutation/data/icgc/for_matrixQTL/icgc_methyl_df_samplesXcpg')
    #icgc_methyl_df = icgc_methyl_dd.compute()
    #icgc_methyl_df_t = icgc_methyl_df.T
    icgc_methyl_dd = dd.read_parquet('/cellar/users/zkoch/methylation_and_mutation/data/icgc/methyl_dir')
    icgc_methyl_df = icgc_methyl_dd.compute()
    icgc_methyl_df_t = icgc_methyl_df.T
    
    shared_samples = set(icgc_methyl_df_t.index) & set(icgc_mut_df['sample'].unique()) & set(icgc_meta_df['sample'].unique())
    icgc_methyl_df_t = icgc_methyl_df_t.loc[shared_samples]
    icgc_methyl_df_t.dropna(how = 'any', axis=1, inplace=True)
    
    # rename columns 
    icgc_mut_df.rename(columns={'chromosome':'chr', 'sample': 'case_submitter_id', 'chromosome_start':'start', 'MAF': 'DNA_VAF'}, inplace=True)
    icgc_meta_df.rename(columns={'sample': 'case_submitter_id'}, inplace=True)
    # merge with mut
    icgc_mut_w_age_df = icgc_mut_df.merge(icgc_meta_df, on='case_submitter_id', how='left')
    # and methyl dfs
    icgc_meta_df_to_merge = icgc_meta_df[['case_submitter_id', 'age_at_index', 'dataset', 'gender']]
    icgc_meta_df_to_merge.set_index('case_submitter_id', inplace=True)
    # make gender column uppercase
    icgc_meta_df_to_merge['gender'] = icgc_meta_df_to_merge['gender'].str.upper()
    icgc_methyl_age_df_t = icgc_methyl_df_t.merge(icgc_meta_df_to_merge, left_index=True, right_index=True, how='left')
    icgc_mi_df = pd.read_parquet('/cellar/users/zkoch/methylation_and_mutation/output_dirs/011723_output/icgc_mi.parquet')
    icgc_mi_df.sort_values(by='mutual_info', ascending=False, inplace=True)
    return icgc_mut_w_age_df, illumina_cpg_locs_df, icgc_methyl_age_df_t, icgc_mi_df

def read_tcga_data(
    dataset: str
    ) -> tuple:
    print("reading in data")
    # read in data
    out_dir = "/cellar/users/zkoch/methylation_and_mutation/output_dirs/output_010423"
    dependency_f_dir = "/cellar/users/zkoch/methylation_and_mutation/dependency_files"
    data_dir = "/cellar/users/zkoch/methylation_and_mutation/data"
    methylation_dir = '/cellar/users/zkoch/methylation_and_mutation/data/processed_methylation'
    illumina_cpg_locs_df, all_mut_df, _, all_methyl_df_t, all_meta_df, _ = get_data.main(
        illum_cpg_locs_fn = os.path.join(dependency_f_dir, "illumina_cpg_450k_locations.csv"),
        out_dir = out_dir,
        methyl_dir = methylation_dir,
        mut_fn = os.path.join(data_dir, "PANCAN_mut.tsv.gz"),
        meta_fn = os.path.join(data_dir, "PANCAN_meta.tsv")
        )
    # add ages to all_methyl_df_t
    all_mut_w_age_df, all_methyl_age_df_t = utils.add_ages_to_mut_and_methyl(all_mut_df, all_meta_df, all_methyl_df_t)
    mi_df = pd.read_parquet('/cellar/users/zkoch/methylation_and_mutation/dependency_files/mutual_informations/tcga_combinedMI_top10MI.parquet')
    if dataset != "":
        mi_df = mi_df[dataset]
    else:
        mi_df = mi_df['combined']
    mi_df = mi_df.to_frame()
    mi_df.columns = ['mutual_info']
    #mi_df.sort_values(by='mutual_info', ascending=False, inplace=True)
    return all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, mi_df

def run(
    generate_features: bool,
    train_models: bool,
    consortium: str,
    dataset: str, 
    cross_val_num: int,
    out_dir: str, 
    start_top_cpgs: int, 
    end_top_cpgs: int,
    aggregate: str,
    mut_feat_store_fns: list
    ) -> None:
    """
    Driver function for generating features or training models
    @ generate_features: bool, whether to generate features
    @ train_models: bool, whether to train models
    @ consortium: str, either "tcga" or "icgc"
    @ dataset: str, either "BRCA", "COAD", ...
    @ train_samples: list of samples to train on
    @ test_samples: list of samples to test on
    @ out_dir: directory to save output to
    @ start_top_cpgs: start of range of top cpgs to use
    @ end_top_cpgs: end of range of top cpgs to use
    @ aggregate: feature aggregation strategy True, False, or Both
    @ mut_feat_store_fns: list of mutation feature store filenames
    @ returns: None
    """
    if consortium == "ICGC":
        # TODO: make ICGC single dataset work
        all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, mi_df = read_icgc_data()
        matrix_qtl_dir = "/cellar/users/zkoch/methylation_and_mutation/output_dirs/icgc_muts_011423"
    elif consortium == "TCGA":
        all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, mi_df = read_tcga_data(dataset)
        matrix_qtl_dir = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/muts"

    if generate_features:
        # create mutation feature generating object
        mut_feat = mutation_features.mutationFeatures(
            all_mut_w_age_df = all_mut_w_age_df, illumina_cpg_locs_df = illumina_cpg_locs_df, 
            all_methyl_age_df_t = all_methyl_age_df_t, out_dir = out_dir, consortium = consortium,
            dataset = dataset, cross_val_num = cross_val_num, 
            matrix_qtl_dir = matrix_qtl_dir
            )
        # choose which CpGs to generate features for, comes back sorted
        cpg_pred_priority = mut_feat.choose_cpgs_to_train(mi_df = mi_df, bin_size=20000, sort_by = ['mutual_info', 'count'])
        chosen_cpgs = cpg_pred_priority.iloc[start_top_cpgs: end_top_cpgs]['#id'].to_list()
        # run the feature generation
        mut_feat.create_all_feat_mats(
            cpg_ids = chosen_cpgs, aggregate=aggregate,
            num_correl_sites=5000, num_correl_ext_sites=100, 
            max_meqtl_sites=100, nearby_window_size=25000
            )
        mut_feat.save_mutation_features(
            start_top_cpgs = start_top_cpgs, cross_val_num = cross_val_num
            )
    elif train_models:
        methyl_pred = methylation_pred.methylationPrediction(
            mut_feat_store_fns = mut_feat_store_fns,
            model_type = 'xgboost'
            )
        methyl_pred.train_all_models()
        methyl_pred.apply_all_models()
        methyl_pred.save_models_and_preds()
    else:
        raise ValueError("Must specify either generate_features or train_models")

def main():
    # parse arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_gen', type=bool, help='whether to generate features', default=False)
    parser.add_argument('--train_models', type=bool, help='whether to train models', default=False)
    parser.add_argument('--mut_feat_store_fns', type=str, help='glob path to feature files', default="")
    parser.add_argument('--consortium', type=str, help='TCGA or ICGC')
    parser.add_argument('--dataset', type=str, help='tissue type to run, e.g. BRCA, COAD, ...', default="")
    parser.add_argument('--cross_val', type=int, help='cross val fold number, assuming 5', default=0)
    parser.add_argument('--out_dir', type=str, help='path to output directory')
    parser.add_argument('--start_top_cpgs', type=int, help='index of top cpgs to start with', default=0)
    parser.add_argument('--end_top_cpgs', type=int, help='index of top cpgs to end with', default=0)
    parser.add_argument('--aggregate', type=str, help='False, True, or Both', default="Both")
    # parse
    args = parser.parse_args()
    # make sure not asking for features and models at the same time
    assert not (args.feat_gen and args.train_models), "cannot generate features and train models at the same time"
    # if training models, need to specify a glob path to the feature files
    assert not (args.train_models and args.mut_feat_store_fns == ""), "must specify glob path to feature files if training models"    
    generate_features = args.feat_gen
    train_models = args.train_models
    # expand the glob path into a list of filenames
    mut_feat_store_fns = glob.glob(args.mut_feat_store_fns)
    print(args.mut_feat_store_fns, flush=True)
    print(mut_feat_store_fns, flush=True)
    consortium = args.consortium
    dataset = args.dataset
    cross_val_num = args.cross_val
    out_dir = args.out_dir
    start_top_cpgs = args.start_top_cpgs
    end_top_cpgs = args.end_top_cpgs
    aggregate = args.aggregate
    print(f"cross val {cross_val_num} running {generate_features} generate features {train_models} train models {consortium} {dataset} and outputting to {out_dir}")
    # run 
    run(
        generate_features = generate_features, train_models = train_models,
        consortium = consortium, dataset = dataset, cross_val_num = cross_val_num,
        out_dir = out_dir, start_top_cpgs = start_top_cpgs,
        end_top_cpgs = end_top_cpgs, aggregate = aggregate,
        mut_feat_store_fns = mut_feat_store_fns
        )
        

        

if __name__ == "__main__":
    main()

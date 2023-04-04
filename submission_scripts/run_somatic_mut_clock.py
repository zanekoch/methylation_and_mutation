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
    return icgc_mut_w_age_df, illumina_cpg_locs_df, icgc_methyl_age_df_t

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
    all_mut_w_age_df, all_methyl_age_df_t = utils.add_ages_to_mut_and_methyl(
        all_mut_df, all_meta_df, all_methyl_df_t
        )
    # mi dfs
    """mi_df = pd.read_parquet('/cellar/users/zkoch/methylation_and_mutation/dependency_files/mutual_informations/tcga_combinedMI_top10MI.parquet')
    if dataset != "":
        mi_df = mi_df[dataset]
    else:
        mi_df = mi_df['combined']
    mi_df = mi_df.to_frame()
    mi_df.columns = ['mutual_info']"""
    return all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t

def run(
    do: str,
    consortium: str,
    dataset: str, 
    cross_val_num: int,
    out_dir: str, 
    start_top_cpgs: int, 
    end_top_cpgs: int,
    mut_feat_params: dict,
    scramble: bool,
    model: str,
    mut_feat_store_fns: list,
    ) -> None:
    """
    Driver function for generating features or training models
    @ consortium: str, either "tcga" or "icgc"
    @ dataset: str, either "BRCA", "COAD", ...
    @ train_samples: list of samples to train on
    @ test_samples: list of samples to test on
    @ out_dir: directory to save output to
    @ start_top_cpgs: start of range of top cpgs to use
    @ end_top_cpgs: end of range of top cpgs to use
    @ mut_feat_params: dictionary of mutation feature parameters
    @ aggregate: feature aggregation strategy True, False, or Both
    @ scramble: whether to scramble the methylation data
    @ model: model to use
    @ mut_feat_store_fns: list of mutation feature store filenames
    @ returns: None
    """
    if consortium == "ICGC":
        # TODO: make ICGC single dataset work
        all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t = read_icgc_data()
        matrix_qtl_dir = "/cellar/users/zkoch/methylation_and_mutation/output_dirs/icgc_muts_011423"
    elif consortium == "TCGA":
        all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t = read_tcga_data(dataset)
        matrix_qtl_dir = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/clumped_muts_CV"
    if do == "Feat_gen":
        generate_features = True
    elif do == "Train_models":
        train_models = True
    elif do == "Both":
        generate_features = True
        train_models = True
    mut_feat_store_fn = ""
    if generate_features:
        print("generating features", flush=True)
        # create mutation feature generating object
        mut_feat = mutation_features.mutationFeatures(
            all_mut_w_age_df = all_mut_w_age_df, illumina_cpg_locs_df = illumina_cpg_locs_df, 
            all_methyl_age_df_t = all_methyl_age_df_t, out_dir = out_dir, 
            consortium = consortium, dataset = dataset, cross_val_num = cross_val_num, 
            matrix_qtl_dir = matrix_qtl_dir
            )
        
        ######## choose CpGs ############
        # get age correlation of CpGs
        age_corr = mut_feat.all_methyl_age_df_t.loc[mut_feat.train_samples].corrwith(
            mut_feat.all_methyl_age_df_t.loc[mut_feat.train_samples, 'age_at_index']
            )
        age_corr.drop(['age_at_index', 'gender_MALE', 'gender_FEMALE'], inplace=True)
        age_corr = age_corr.to_frame()
        age_corr.columns = ['age_corr']
        age_corr['abs_age_corr'] = age_corr['age_corr'].abs()
        # get stdev of CpGs
        methyl_stdev = mut_feat.all_methyl_age_df_t.loc[mut_feat.train_samples, :].std()
        methyl_stdev.drop(['age_at_index', 'gender_MALE', 'gender_FEMALE'], inplace=True)
        methyl_stdev = methyl_stdev.to_frame()
        methyl_stdev.columns = ['methyl_stdev']
        # choose the top cpgs sorted by cpg_pred_priority
        cpg_pred_priority = mut_feat.choose_cpgs_to_train(
            metric_df = age_corr, bin_size = mut_feat_params['bin_size'], 
            sort_by = ['count', 'abs_age_corr']
            )
        cpg_pred_priority = cpg_pred_priority.merge(methyl_stdev, left_on = '#id', right_index=True, how='left')
        # choose the top cpgs
        chosen_cpgs = cpg_pred_priority.iloc[start_top_cpgs: end_top_cpgs]['#id'].to_list()
        
        ##################################
        # run the feature generation
        mut_feat.create_all_feat_mats(
            cpg_ids = chosen_cpgs, 
            aggregate = mut_feat_params['aggregate'],
            num_correl_sites = mut_feat_params['num_correl_sites'], 
            max_meqtl_sites=mut_feat_params['max_meqtl_sites'],
            nearby_window_size = mut_feat_params['nearby_window_size'], 
            extend_amount = mut_feat_params['extend_amount'] 
            )
        mut_feat_store_fn = mut_feat.save_mutation_features(
            start_top_cpgs = start_top_cpgs
            )
        mut_feat_store_fns = [mut_feat_store_fn]
    if train_models:
        print("training models", flush=True)
        methyl_pred = methylation_pred.methylationPrediction(
            mut_feat_store_fns = mut_feat_store_fns,
            model_type = model,
            scramble = False
            )
        methyl_pred.train_all_models()
        methyl_pred.apply_all_models()
        methyl_pred.save_models_and_preds()
        # also do scrambled
        if scramble:
            methyl_pred = methylation_pred.methylationPrediction(
                mut_feat_store_fns = mut_feat_store_fns,
                model_type = model,
                scramble = True
                #trained_models_fns = [trained_models_fn]
                )
            methyl_pred.train_all_models()
            methyl_pred.apply_all_models()
            methyl_pred.save_models_and_preds()
    

def main():
    # parse arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--do', type=str, help='whether to generate features, train models, or both')
    parser.add_argument('--mut_feat_store_fns', type=str, help='glob path to feature files', default="")
    parser.add_argument('--consortium', type=str, help='TCGA or ICGC')
    parser.add_argument('--dataset', type=str, help='tissue type to run, e.g. BRCA, COAD, ...', default="")
    parser.add_argument('--cross_val', type=int, help='cross val fold number, assuming 5', default=0)
    parser.add_argument('--out_dir', type=str, help='path to output directory')
    parser.add_argument('--start_top_cpgs', type=int, help='index of top cpgs to start with', default=0)
    parser.add_argument('--end_top_cpgs', type=int, help='index of top cpgs to end with', default=0)
    parser.add_argument('--aggregate', type=str, help='False, True, or Both', default="Both")
    parser.add_argument('--scramble' , type=bool, help='whether to scramble the mutation data', default=False)
    parser.add_argument('--model', type=str, help='xgboost or elasticNet', default="xgboost")
    parser.add_argument('--burden_bin_size', type=int, help='bin size for burden test', default=25000)
    parser.add_argument('--num_correl_sites', type=int, help='num_correl_sites', default=50)
    parser.add_argument('--max_meqtl_sites', type=int, help='max_meqtl_sites', default=100000)
    parser.add_argument('--nearby_window_size', type=int, help='nearby_window_size', default=50000)
    parser.add_argument('--extend_amount', type=int, help='extend_amount', default=100)
    
    args = parser.parse_args()
    do = args.do
    # assert do has a valid value
    assert (do in ['Feat_gen', 'Train_models', 'Both']), "cannot generate features and train models at the same time"
    # if training models, need to specify a glob path to the feature files
    if do == "train_models":
        assert (args.mut_feat_store_fns != ""), "must specify glob path to feature files if training models"    
    # expand the glob path into a list of filenames
    # may just be one file, but this still makes a list
    mut_feat_store_fns = glob.glob(args.mut_feat_store_fns)
    consortium = args.consortium
    dataset = args.dataset
    cross_val_num = args.cross_val
    out_dir = args.out_dir
    start_top_cpgs = args.start_top_cpgs
    end_top_cpgs = args.end_top_cpgs
    scramble = args.scramble
    model = args.model
    if model not in ['xgboost', 'elasticNet']:
        raise ValueError("model must be xgboost or elasticNet")
    print(f"cross val {cross_val_num}\n doing {do}\n for {consortium} and {dataset}\n and outputting to {out_dir}")
    
    mut_feat_params = {
        'bin_size': args.burden_bin_size, 'aggregate': args.aggregate, 
        'num_correl_sites' : args.num_correl_sites, 'max_meqtl_sites' : args.max_meqtl_sites,
        'nearby_window_size' : args.nearby_window_size, 'extend_amount' : args.extend_amount
    }
    
    run(
        do = do,
        consortium = consortium, dataset = dataset, cross_val_num = cross_val_num,
        out_dir = out_dir, start_top_cpgs = start_top_cpgs,
        end_top_cpgs = end_top_cpgs, mut_feat_params = mut_feat_params,
        scramble = scramble, model = model, mut_feat_store_fns = mut_feat_store_fns
        )
        

if __name__ == "__main__":
    main()

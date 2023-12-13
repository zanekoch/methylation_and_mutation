import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import get_data, utils, methylation_pred, mutation_features
import argparse
import os
import pandas as pd
import numpy as np
import dask.dataframe as dd
import glob

def run(
    do: str,
    consortium: str,
    dataset: str, 
    cross_val_num: int,
    out_dir: str, 
    start_top_cpgs: int, 
    end_top_cpgs: int,
    mut_feat_params: dict,
    train_baseline: str,
    train_actual_model:str,
    model: str,
    mut_feat_store_fns: list,
    agg_only_methyl_pred: bool,
    scale_counts_within_dataset: bool,
    predict_with_random_feat: int,
    use_gpu: bool,
    trained_model_fns: list
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
    @ train_baseline: whether to scramble, cov_only, or none for baseline
    @ train_actual_model: whether to train the actual, non-baseline, model
    @ model: model to use
    @ mut_feat_store_fns: list of mutation feature store filenames
    @ agg_only_methyl_pred: whether to train the models on aggregate features only or not
    @ scale_counts_within_dataset: whether to scale the mutation counts within the dataset
    @ predict_with_random_feat: Nunber of times to predict with random features
    @ use_gpu: whether to use gpu or not
    @ trained_model_fns: list of trained model filenames
    @ returns: None
    """
    if consortium == "ICGC":
        all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, matrix_qtl_dir, covariate_fn = get_data.read_icgc_data()
    elif consortium == "TCGA":
        all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, matrix_qtl_dir, covariate_fn = get_data.read_tcga_data()
    # read in motif occurence df
    motif_occurence_df = pd.read_parquet(
        "/cellar/users/zkoch/methylation_and_mutation/data/methylation_motifs_weiWang/motif_occurences/motif_occurences_combined_15kb.parquet"
        )
    print("read in motif occurence df", flush=True)
    generate_features = False
    train_models = False
    predict = False
    if do == "Feat_gen":
        generate_features = True
    elif do == "Train_models": # predicting here implicitly
        train_models = True
    elif do == "Both": # predicting here implicitly
        generate_features = True
        train_models = True
    elif do == "Predict": # only predicting here, no feature generation or model training
        predict = True
        
    mut_feat_store_fn = ""
    if generate_features:
        print("generating features", flush=True)
        # create mutation feature generating object
        mut_feat = mutation_features.mutationFeatures(
            all_mut_w_age_df = all_mut_w_age_df, illumina_cpg_locs_df = illumina_cpg_locs_df, 
            all_methyl_age_df_t = all_methyl_age_df_t, out_dir = out_dir, 
            consortium = consortium, dataset = dataset, cross_val_num = cross_val_num, 
            matrix_qtl_dir = matrix_qtl_dir,
            covariate_fn = covariate_fn, motif_occurence_df = motif_occurence_df
            )
        ######## choose CpGs ############
        # choose the top cpgs sorted by nearby mutation count and then absolute age correlation
        cpg_pred_priority = mut_feat.choose_cpgs_to_train(
            bin_size = mut_feat_params['bin_size'], 
            sort_by = ['count', 'abs_age_corr']
            )
        # choose the start - end top cpgs
        chosen_cpgs = cpg_pred_priority.iloc[start_top_cpgs: end_top_cpgs]['#id'].to_list()
        
        ######### feature generation ###########
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
        # there are 3 options: train baseline, train actual model, or both
        # both
        if train_actual_model == 'True' and train_baseline != 'none':
            print(f"training actual model and {train_baseline} baseline", flush=True)
            # do actual model
            methyl_pred = methylation_pred.methylationPrediction(
                mut_feat_store_fns = mut_feat_store_fns,
                model_type = model,
                baseline = "none",
                agg_only = agg_only_methyl_pred,
                scale_counts_within_dataset = scale_counts_within_dataset,
                predict_with_random_feat = predict_with_random_feat,
                illumina_cpg_locs_df = illumina_cpg_locs_df,
                all_methyl_age_df_t = all_methyl_age_df_t,
                use_gpu = use_gpu
                )
            methyl_pred.train_all_models()
            methyl_pred.apply_all_models()
            methyl_pred.calc_performance_by_dataset()
            methyl_pred.save_models_and_preds()
            # baseline
            methyl_pred = methylation_pred.methylationPrediction(
                mut_feat_store_fns = mut_feat_store_fns,
                model_type = model,
                baseline = train_baseline,
                agg_only = agg_only_methyl_pred,
                scale_counts_within_dataset = scale_counts_within_dataset,
                all_methyl_age_df_t = all_methyl_age_df_t,
                use_gpu = use_gpu
                )
            methyl_pred.train_all_models()
            methyl_pred.apply_all_models()
            methyl_pred.calc_performance_by_dataset()
            methyl_pred.save_models_and_preds()
        # just actual model
        elif train_actual_model == 'True' and train_baseline == 'none':
            print("training actual model and not baseline ", flush=True)
            methyl_pred = methylation_pred.methylationPrediction(
                mut_feat_store_fns = mut_feat_store_fns,
                model_type = model,
                baseline = "none",
                agg_only = agg_only_methyl_pred,
                scale_counts_within_dataset = scale_counts_within_dataset,
                predict_with_random_feat = predict_with_random_feat,
                illumina_cpg_locs_df = illumina_cpg_locs_df,
                all_methyl_age_df_t = all_methyl_age_df_t,
                use_gpu = use_gpu
                )
            methyl_pred.train_all_models()
            methyl_pred.apply_all_models()
            methyl_pred.calc_performance_by_dataset()
            methyl_pred.save_models_and_preds()
        # just baseline
        elif train_actual_model == 'False' and train_baseline != 'none':
            print(f"training only {train_baseline} baseline", flush=True)
            # do non-scrambled
            methyl_pred = methylation_pred.methylationPrediction(
                mut_feat_store_fns = mut_feat_store_fns,
                model_type = model,
                baseline = train_baseline,
                agg_only = agg_only_methyl_pred,
                scale_counts_within_dataset = scale_counts_within_dataset,
                all_methyl_age_df_t = all_methyl_age_df_t,
                use_gpu = use_gpu
                )
            methyl_pred.train_all_models()
            methyl_pred.apply_all_models()
            methyl_pred.calc_performance_by_dataset()
            methyl_pred.save_models_and_preds()
        else:
            print(f"{train_baseline} and {train_actual_model} are not a valid combination",
                  flush=True
                  )
    if predict:
        print("predicting", flush=True)
        methyl_pred = methylation_pred.methylationPrediction(
            mut_feat_store_fns = mut_feat_store_fns,
            model_type = model,
            trained_models_fns = trained_model_fns,
            baseline = train_baseline,
            agg_only = agg_only_methyl_pred,
            scale_counts_within_dataset = scale_counts_within_dataset,
            predict_with_random_feat = predict_with_random_feat,
            illumina_cpg_locs_df = illumina_cpg_locs_df,
            all_methyl_age_df_t = all_methyl_age_df_t,
            use_gpu = use_gpu
            )
        methyl_pred.apply_all_models()
        methyl_pred.calc_performance_by_dataset()
        methyl_pred.save_models_and_preds()        
            
            
def parse_str_bool(bool_str: str) -> bool:
    """
    Parses a string to a boolean
    @ bool_str: str, either "True" or "False"
    @ returns: bool
    """
    if bool_str == "True":
        return True
    elif bool_str == "False":
        return False
    else:
        raise ValueError("bool_str must be True or False")

def main():
    # parse arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--do', type=str, help="whether to 'Feat_gen', 'Train_models', 'Both' (train models and feat_gen), or 'Predict")
    parser.add_argument('--mut_feat_store_fns', type=str, help='glob path to feature files', default="")
    parser.add_argument('--trained_model_fns', type=str, help='glob path to model files', default="")
    parser.add_argument('--consortium', type=str, help='TCGA or ICGC')
    parser.add_argument('--dataset', type=str, help='tissue type to run, e.g. BRCA, COAD, ...', default="")
    parser.add_argument('--cross_val', type=int, help='cross val fold number, assuming 5', default=0)
    parser.add_argument('--out_dir', type=str, help='path to output directory')
    parser.add_argument('--start_top_cpgs', type=int, help='index of top cpgs to start with', default=0)
    parser.add_argument('--end_top_cpgs', type=int, help='index of top cpgs to end with', default=0)
    parser.add_argument('--aggregate', type=str, help='False, True, or Both', default="Both")
    parser.add_argument('--train_baseline' , type=str, help='whether to use a baseline, if so scrambled or cov_only [scramble, cov_only, both, none]', default='none')
    parser.add_argument('--train_actual_model', type=str, help='whether to train the actual non-baseline model [True, False]', default='True')
    parser.add_argument('--model', type=str, help='xgboost or elasticNet', default="xgboost")
    parser.add_argument('--burden_bin_size', type=int, help='bin size for burden test', default=25000)
    parser.add_argument('--num_correl_sites', type=int, help='num_correl_sites', default=50)
    parser.add_argument('--max_meqtl_sites', type=int, help='max_meqtl_sites', default=100000)
    parser.add_argument('--nearby_window_size', type=int, help='nearby_window_size', default=50000)
    parser.add_argument('--extend_amount', type=int, help='extend_amount', default=100)
    parser.add_argument('--agg_only_methyl_pred', type=str, help='to train the models on aggregate features only or not: "True" or "False" ')
    parser.add_argument('--scale_counts_within_dataset', type=str, help='whether to scale the mutation counts within the dataset: "True" or "False" ')
    parser.add_argument('--predict_with_random_feat', type=int, help='Number of times to predict with random features', default=-1)
    parser.add_argument('--use-gpu', type=str, help='True/False gpu')
    
    args = parser.parse_args()
    do = args.do
    # assert do has a valid value
    assert (do in ['Feat_gen', 'Train_models', 'Both', 'Predict']), "cannot generate features and train models at the same time"
    # if training models, need to specify a glob path to the feature files
    if do == "Train_models":
        assert (args.mut_feat_store_fns != ""), "must specify glob path to feature files if training models" 
    # expand the glob path into a list of filenames
    # may just be one file, but this still makes a list
    mut_feat_store_fns = glob.glob(args.mut_feat_store_fns)
    if do == "Predict":
        assert (args.trained_model_fns != ""), "must specify glob path to trained model files if predicting"
    trained_model_fns = glob.glob(args.trained_model_fns)
    
    consortium = args.consortium
    dataset = args.dataset
    cross_val_num = args.cross_val
    out_dir = args.out_dir
    start_top_cpgs = args.start_top_cpgs
    end_top_cpgs = args.end_top_cpgs
    train_baseline = args.train_baseline
    train_actual_model = args.train_actual_model
    agg_only_methyl_pred = args.agg_only_methyl_pred
    scale_counts_within_dataset = args.scale_counts_within_dataset
    predict_with_random_feat = args.predict_with_random_feat
    use_gpu = args.use_gpu
    model = args.model
    if model not in ['xgboost', 'elasticNet']:
        raise ValueError("model must be xgboost or elasticNet")
    agg_only_methyl_pred = parse_str_bool(agg_only_methyl_pred)
    scale_counts_within_dataset = parse_str_bool(scale_counts_within_dataset)
    use_gpu = parse_str_bool(use_gpu)
    
    print(f"cross val {cross_val_num}\n doing {do}\n for {consortium} and {dataset}\n, {use_gpu} use gpu, and outputting to {out_dir}")
    
    mut_feat_params = {
        'bin_size': args.burden_bin_size, 'aggregate': args.aggregate, 
        'num_correl_sites' : args.num_correl_sites, 'max_meqtl_sites' : args.max_meqtl_sites,
        'nearby_window_size' : args.nearby_window_size, 'extend_amount' : args.extend_amount
    }
    print(mut_feat_params)
    print(model)
    print(f"agg only {agg_only_methyl_pred}")
    print(f"scale counts within dataset {scale_counts_within_dataset}")
    print(f"predict with random feat {predict_with_random_feat}")
    run(
        do = do,
        consortium = consortium, dataset = dataset, cross_val_num = cross_val_num,
        out_dir = out_dir, start_top_cpgs = start_top_cpgs,
        end_top_cpgs = end_top_cpgs, mut_feat_params = mut_feat_params,
        train_baseline = train_baseline, train_actual_model = train_actual_model, model = model, mut_feat_store_fns = mut_feat_store_fns,
        agg_only_methyl_pred = agg_only_methyl_pred, 
        scale_counts_within_dataset = scale_counts_within_dataset,
        predict_with_random_feat = predict_with_random_feat, use_gpu = use_gpu,
        trained_model_fns = trained_model_fns
        )
        
if __name__ == "__main__":
    main()

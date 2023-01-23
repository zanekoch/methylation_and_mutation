import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import get_data, utils, somatic_mut_clock
import argparse
import os
import pandas as pd
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
    print(icgc_methyl_df_t.shape)
    
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

def read_tcga_data() -> tuple:
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
    mi_df = pd.read_parquet('/cellar/users/zkoch/methylation_and_mutation/output_dirs/011723_output/tcga_training_mi.parquet')
    mi_df.sort_values(by='mutual_info', ascending=False, inplace=True)
    return all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, mi_df


def main():
    # parse arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='TCGA or ICGC')
    parser.add_argument('--do', type=str, help='train, evaluate, predict, or feature information')
    parser.add_argument('--out_dir', type=str, help='TCGA or ICGC')    
    parser.add_argument('--num_correl_sites', type=int, help='number of correl sites to use for prediction')
    parser.add_argument('--max_meqtl_sites', type=int, help='max_meqtl_sites')
    parser.add_argument('--nearby_window_size', type=int, help='nearby_window_size')
    parser.add_argument('--num_top_mi_cpgs', type=int, help='num_top_mi_cpgs')
    parser.add_argument('--samples_fn', type=str, help='path to file with samples to use', default="")
    parser.add_argument('--aggregate', type=str, help='aggregate')
    parser.add_argument('--binarize', type=str, help='binarize')
    parser.add_argument('--use_all_muts', type=str, help='use_all_muts')
    parser.add_argument('--trained_model_dir', type=str, help='train_model_dir', default="")
    parser.add_argument('--tissue_type', type=str, help='subset to only train models for one tissue', default="")
    # parse arguments
    args = parser.parse_args()
    dataset = args.dataset
    do = args.do
    out_dir = args.out_dir
    num_correl_sites = args.num_correl_sites
    max_meqtl_sites = args.max_meqtl_sites
    nearby_window_size = args.nearby_window_size
    num_top_mi_cpgs = args.num_top_mi_cpgs
    aggregate = args.aggregate
    binarize = True if  args.binarize == "True" else False
    use_all_muts = True if args.use_all_muts == "True" else False
    samples_fn = args.samples_fn
    trained_model_dir = args.trained_model_dir
    tissue_type = args.tissue_type
    if samples_fn != "":
        with open(samples_fn, 'r') as f:
            samples = f.read().splitlines()
    else:
        samples = []
            
    if dataset == "TCGA":
        all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, mi_df = read_tcga_data()
        mut_clock = somatic_mut_clock.mutationClock(
            all_mut_w_age_df = all_mut_w_age_df, 
            illumina_cpg_locs_df = illumina_cpg_locs_df, 
            all_methyl_age_df_t = all_methyl_age_df_t,
            output_dir = out_dir,
            tissue_type = tissue_type
            )
    elif dataset == "ICGC":
        icgc_mut_w_age_df, illumina_cpg_locs_df, icgc_methyl_age_df_t, mi_df = read_icgc_data()
        # create a mutationClock object with icgc data
        mut_clock = somatic_mut_clock.mutationClock(
            all_mut_w_age_df = icgc_mut_w_age_df, 
            illumina_cpg_locs_df = illumina_cpg_locs_df, 
            all_methyl_age_df_t = icgc_methyl_age_df_t,
            output_dir = out_dir,
            tissue_type = tissue_type, 
            matrix_qtl_dir = "/cellar/users/zkoch/methylation_and_mutation/output_dirs/icgc_muts_011423"
            )
    print("Got data, now training and evaluating")
    if do == 'predict':
        predicted_methyl = mut_clock.predict_all_cpgs(
            cpg_ids = mi_df.head(num_top_mi_cpgs).index.to_list(), test_samples = samples, 
            model_dir = trained_model_dir, aggregate = aggregate, binarize = binarize
            )
        predicted_methyl.to_parquet(
            os.path.join(out_dir, f"predicted_methyl_{dataset}_{num_correl_sites}correl_{max_meqtl_sites}matrixQtl_{nearby_window_size}nearby_{num_top_mi_cpgs}numCpGs_{aggregate}Aggregate_{binarize}binarize_{use_all_muts}AllMuts.parquet")
            )
        print(
            f"Done, wrote results to predicted_methyl_{dataset}_{num_correl_sites}correl_{max_meqtl_sites}matrixQtl_{nearby_window_size}nearby_{num_top_mi_cpgs}numCpGs_{aggregate}Aggregate_{binarize}binarize_{use_all_muts}AllMuts.parquet"
            )
    else: 
        result_df = mut_clock.driver(
            do = do, num_correl_sites = num_correl_sites, max_meqtl_sites = max_meqtl_sites,
            nearby_window_size = nearby_window_size, cpg_ids = mi_df.iloc[:num_top_mi_cpgs, :].index.to_list(), 
            train_samples = samples, aggregate = aggregate, binarize = binarize
            )
        if do == "evaluate" or do == "train":
            result_df.to_parquet(
                os.path.join(out_dir, f"evaluate_results_{dataset}_{num_correl_sites}correl_{max_meqtl_sites}matrixQtl_{nearby_window_size}nearby_{num_top_mi_cpgs}numCpGs_{aggregate}Aggregate_{binarize}binarize_{use_all_muts}AllMuts.parquet")
                )
            print(
                f"Done, wrote results to evaluate_results_{dataset}_{num_correl_sites}correl_{max_meqtl_sites}matrixQtl_{nearby_window_size}nearby_{num_top_mi_cpgs}numCpGs_{aggregate}Aggregate_{binarize}binarize_{use_all_muts}AllMuts.parquet"
                )
    

if __name__ == "__main__":
    main()
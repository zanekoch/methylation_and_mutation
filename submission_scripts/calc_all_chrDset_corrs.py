import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import get_data
import argparse
import os 

def preproc_correls(
    illumina_cpg_locs_df, 
    all_methyl_age_df_t, 
    out_dir
    ) -> None:
    """
    Calculate the correlation matrix for each dataset within each chromosome and output to file
    """
    print("starting preproc_correls", flush=True)
    # get all measured cpgs (nans already dropped)
    all_measured_cpgs = all_methyl_age_df_t.columns.to_list()
    for chrom in illumina_cpg_locs_df['chr'].unique():
        # get the cpgs for this chromosome
        this_chr_illumina_cpgs = illumina_cpg_locs_df.query('chr == @chrom')['#id'].to_list()
        # and the cpgs that are measured in this chromosome
        this_chr_measured_cpgs = list(set(all_measured_cpgs).intersection(set(this_chr_illumina_cpgs)))
        for dset in all_methyl_age_df_t['dataset'].unique():
            # select the cpgs for this dataset
            this_chr_dset_methyl_df = all_methyl_age_df_t.query('dataset == @dset').loc[:, this_chr_measured_cpgs]
            # and calculate the correlation matrix
            corr_df = this_chr_dset_methyl_df.corr()
            corr_df.to_parquet(os.path.join(out_dir, 'chr{}_{}.parquet'.format(chrom, dset)))
            print("Done", chrom, dset, flush=True)

def main():
    # argparse
    parser = argparse.ArgumentParser(description='Calculate correlation between mutation and methylation')
    parser.add_argument('--out_dir', type=str, help='output directory')
    parser.add_argument('--consortium', type=str, help='TCGA or ICGC')
    # parse
    args = parser.parse_args()
    out_dir = args.out_dir
    consortium = args.consortium
    # get data
    if consortium == "ICGC":
        all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, matrix_qtl_dir, covariate_fn = get_data.read_icgc_data()
    elif consortium == "TCGA":
        all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, matrix_qtl_dir, covariate_fn = get_data.read_tcga_data()
    else:
        raise ValueError("consortium must be TCGA or ICGC")
    
    preproc_correls(
        illumina_cpg_locs_df,
        all_methyl_age_df_t,
        out_dir
        )
    
if __name__ == "__main__":
    main()
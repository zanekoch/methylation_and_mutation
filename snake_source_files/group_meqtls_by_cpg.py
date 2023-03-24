import pandas as pd
import glob
import os
import argparse

def group_by_cpg(
    chrom: str,
    out_fn: str, 
    matrix_qtl_dir: str,
    fold_num: int,
    illumina_cpg_locs_fn: str = "/cellar/users/zkoch/methylation_and_mutation/dependency_files/illumina_cpg_450k_locations.csv"
    ) -> None:
    """
    Go through each *metql file in matrix_qtl_dir and get the CpG-meQTL pairs for all the CpGs in chrom according to illumina_cpg_locs_fn
    @ chrom: the name of a chromosome (not including X or Y)
    @ out_fn: the path to the output file
    @ matrix_qtl_fn: the path to the matrix qtl file
    @ fold_num: the fold number
    @ illumina_cpg_locs_fn: the path to the file containing the locations of the CpGs in the 450k array
    """
    # read in the locations of the CpGs in the 450k array
    illumina_cpg_locs_df = pd.read_csv(illumina_cpg_locs_fn, sep=',', dtype={'CHR': str}, low_memory=False)
    illumina_cpg_locs_df = illumina_cpg_locs_df.rename(
        {"CHR": "chr", "MAPINFO":"start", "IlmnID": "#id"}, axis=1
        )
    illumina_cpg_locs_df = illumina_cpg_locs_df[['#id','chr', 'start', 'Strand']]
    # subset to the CpGs in chrom
    illumina_cpg_locs_df = illumina_cpg_locs_df[illumina_cpg_locs_df['chr'] == chrom]
    print("read in illumina_cpg_locs_df", flush=True)
    
    # read in each partition number of this fold
    all_matrix_qtl_dfs = []
    this_fold_matrix_qtl_fns = glob.glob(os.path.join(matrix_qtl_dir, f"muts_fold_{fold_num}_partition_*meqtl"))
    for matrix_qtl_fn in this_fold_matrix_qtl_fns:                         
        meqtl_df = pd.read_csv(matrix_qtl_fn, sep='\t')
        meqtl_df = meqtl_df.rename({'gene': '#id'}, axis=1)
        all_matrix_qtl_dfs.append(meqtl_df)
        print(f"read in meqtl_df {matrix_qtl_fn}", flush=True)
    meqtl_df = pd.concat(all_matrix_qtl_dfs)
    # join with illumina_cpg_locs_df on #id
    this_chr_meqtls = meqtl_df.merge(illumina_cpg_locs_df, on='#id')
    print("merged meqtl_df with illumina_cpg_locs_df", flush=True)
    
    # rename 'chr' column to 'cpg_chr' and 'start' to 'cpg_start'
    this_chr_meqtls = this_chr_meqtls.rename({'chr': 'cpg_chr', 'start': 'cpg_start'}, axis=1)
    this_chr_meqtls.drop('Strand', axis=1, inplace=True)
    this_chr_meqtls[['snp_chr', 'snp_start']] = this_chr_meqtls['SNP'].str.split(':', expand=True)
    # create column specifying if cis and if so the distance
    this_chr_meqtls['cis'] = this_chr_meqtls.apply(
        lambda x: True if x['cpg_chr'] == x['snp_chr'] else False, axis=1
        )
    # make both int
    this_chr_meqtls['snp_start'] = this_chr_meqtls['snp_start'].astype(int)
    this_chr_meqtls['cpg_start'] = this_chr_meqtls['cpg_start'].astype(int)
    this_chr_meqtls['distance'] = this_chr_meqtls.apply(
        lambda x: x['cpg_start'] - x['snp_start'] if x['cis'] else -1, axis=1
        )
    # write out as parquet
    this_chr_meqtls.to_parquet(out_fn)
    print(f"wrote to {out_fn}", flush=True)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrom", type=str, required=True)
    parser.add_argument("--out_fn", type=str, required=True)
    parser.add_argument("--matrix_qtl_dir", type=str, required=True)
    parser.add_argument("--fold", type=int, required=False)
    args = parser.parse_args()
    group_by_cpg(args.chrom, args.out_fn, args.matrix_qtl_dir, args.fold)

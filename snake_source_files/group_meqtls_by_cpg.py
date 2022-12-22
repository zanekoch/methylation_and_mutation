import pandas as pd
import glob
import os
import argparse

def group_by_cpg(
    chrom: str,
    out_fn: str, 
    illumina_cpg_locs_fn: str = "/cellar/users/zkoch/methylation_and_mutation/dependency_files/illumina_cpg_450k_locations.csv",
    matrix_qtl_dir: str = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/muts"
    ) -> None:
    """
    Go through each *metql file in matrix_qtl_dir and get the CpG-meQTL pairs for all the CpGs in chrom according to illumina_cpg_locs_fn
    @ chrom: the name of a chromosome (not including X or Y)
    @ out_dir: the directory to write the output to
    @ illumina_cpg_locs_fn: the path to the file containing the locations of the CpGs in the 450k array
    @ matrix_qtl_dir: path to directory filled with meQTL files
    """
    # get a list of the *meqtl file names in matrix_qtl_dir using glob
    meqtl_fns = glob.glob(os.path.join(matrix_qtl_dir, "*.meqtl"))
    # read in the locations of the CpGs in the 450k array
    illumina_cpg_locs_df = pd.read_csv(illumina_cpg_locs_fn, sep=',', dtype={'CHR': str}, low_memory=False)
    illumina_cpg_locs_df = illumina_cpg_locs_df.rename({"CHR": "chr", "MAPINFO":"start", "IlmnID": "#id"}, axis=1)
    illumina_cpg_locs_df = illumina_cpg_locs_df[['#id','chr', 'start', 'Strand']]
    # subset to the CpGs in chrom
    illumina_cpg_locs_df = illumina_cpg_locs_df[illumina_cpg_locs_df['chr'] == chrom]
    
    all_meqtls = []
    # for each meqtl file
    for i, meqtl_fn in enumerate(meqtl_fns):
        # read in file
        meqtl_df = pd.read_csv(meqtl_fn, sep='\t')
        meqtl_df = meqtl_df.rename({'gene': '#id'}, axis=1)
        # join with illumina_cpg_locs_df on #id
        this_chr_meqtls = meqtl_df.merge(illumina_cpg_locs_df, on='#id')
        all_meqtls.append(this_chr_meqtls)
        if i == 3:
            break
    # concat all_meqtls
    all_meqtls_df = pd.concat(all_meqtls)
    # rename 'chr' column to 'cpg_chr' and 'start' to 'cpg_start'
    all_meqtls_df = all_meqtls_df.rename({'chr': 'cpg_chr', 'start': 'cpg_start'}, axis=1)
    all_meqtls_df.drop('Strand', axis=1, inplace=True)
    all_meqtls_df[['snp_chr', 'snp_start']] = all_meqtls_df['SNP'].str.split(':', expand=True)
    # write out as parquet
    all_meqtls_df.to_parquet(out_fn)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrom", type=str, required=True)
    parser.add_argument("--out_fn", type=str, required=True)
    args = parser.parse_args()
    group_by_cpg(args.chrom, args.out_fn)
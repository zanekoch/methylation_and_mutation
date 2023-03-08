import pandas as pd
import os
import argparse

def piv_and_partit(
    mut_fn: str, 
    out_dir: str, 
    mut_per_file: int
    ) -> None:
    """
    Given a file of mutations (mut_fn), pivot the table so that each row is a mutation and each column is a sample. 
    Then, write out the pivoted table in mut_per_file chunks to .csv.gz files in out_dir
    @ mut_fn: path to the mutation file. Must have columns 'mut_loc', 'sample', and 'MAF'
    @ out_dir: the directory to write the output to
    @ mut_per_file: the number of mutations to write out to each .csv.gz file
    @ returns: None
    """
    # read in the mutation file
    mut_df = pd.read_parquet(mut_fn)
    print(mut_df.shape, flush=True)
    print(len(mut_df['mut_loc'].unique()), flush=True)
    # pivot
    mut_piv = mut_df.pivot_table(
        index='mut_loc', columns='sample', values='MAF', fill_value=0
        )
    # sort sample columns lexicographically
    mut_piv = mut_piv.reindex(sorted(mut_piv.columns), axis=1)
    print(mut_piv.shape, flush=True)
    # write out in mut_per_file chunks to .csv.gz files
    for i in range(0, mut_piv.shape[0], mut_per_file):
        mut_piv.iloc[i:i+mut_per_file, :].to_csv(
            os.path.join(out_dir, f"muts_{i}.csv.gz"), compression='gzip'
            )
    return
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mut_fn", type=str, required=True, help="path to the mutation file. Must hav")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mut_per_file", type=int, required=True)
    args = parser.parse_args()
    mut_fn = args.mut_fn
    out_dir = args.out_dir
    mut_per_file = args.mut_per_file
    # make the output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f"Reading in mutation file {mut_fn} and outputting to {out_dir} in chunks of {mut_per_file} mutations per file")
    piv_and_partit(mut_fn, out_dir, mut_per_file)
    
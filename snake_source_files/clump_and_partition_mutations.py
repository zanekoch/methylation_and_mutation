import pandas as pd
import os
import argparse
import pickle

def read_file(mut_fn, training_samples_fn):
    # read in the mutation file
    mut_df = pd.read_parquet(mut_fn)
    # read in list of training samples from pkl file
    with open(training_samples_fn, 'rb') as fp:
        train_samples = pickle.load(fp)
    # subset to training samples
    mut_df = mut_df[mut_df['sample'].isin(train_samples)]
    print(f"Read in mutation file and subset to {len(train_samples)} training samples")
    return mut_df

def clump_partit_mutations(
    mut_df: str, 
    clump_window_size: int,
    out_dir: str,
    fold_num: int
    ) -> pd.DataFrame:
    """
    Clump mutations in window and drop windows without enough mutations
    @ mut_df: the mutation dataframe
    @ clump_window_size: the size of the window to clump mutations into
    @ out_dir: the directory to write the output to
    @ fold_num: the fold number
    @ returns: None
    """
    def round_down(num, divisor):
        return num - (num%divisor)
    
    if clump_window_size > 1:
        # create binary mut column to be used for counting 
        mut_df['binary_MAF'] = 1
        # clump start is 'start' rounded down to nearest 1000
        mut_df['clump_start'] = mut_df['start'].apply(lambda x: round_down(x, clump_window_size))
        mut_df['clump_loc'] = mut_df['chr'].astype(str) + ':' + mut_df['clump_start'].astype(str)
        # combine rows with the same sample, chr, and clump_start values by adding the MAF values
        grouped_mut_df = mut_df.groupby(
            ['sample', 'clump_loc']
            ).agg({'MAF': 'sum', 'binary_MAF': 'sum'}).reset_index()
        # pivot to be clump_start x samples
        clumped_mut_df = pd.pivot_table(
            grouped_mut_df, index='clump_loc', columns='sample',
            values='binary_MAF', fill_value=0
            )
        # sort sample columns lexicographically
        clumped_mut_df = clumped_mut_df.reindex(sorted(clumped_mut_df.columns), axis=1)
        print("Pivoted and clumped mutations to get matrix of shape: ", clumped_mut_df.shape, flush=True)
        out_fn = os.path.join(out_dir, f"muts_fold_{fold_num}.csv.gz")
        clumped_mut_df.to_csv(out_fn, compression='gzip')
        print(f"Wrote out clumped mutations to {out_fn}", flush=True)
    else: # no clumping
        # if does not contains mut_loc column, create it
        if 'mut_loc' not in mut_df.columns:
            mut_df['mut_loc'] = mut_df['chr'].astype(str) + ':' + mut_df['start'].astype(str)
        # pivot
        mut_piv = mut_df.pivot_table(
            index='mut_loc', columns='sample', values='MAF', fill_value=0
            )
        # sort sample columns lexicographically
        mut_piv = mut_piv.reindex(sorted(mut_piv.columns), axis=1)
        print("Pivoted mutations to get matrix of shape: ", mut_piv.shape, flush=True)
        out_fn = os.path.join(out_dir, f"muts_fold_{fold_num}.csv.gz")
        mut_piv.to_csv(out_fn, compression='gzip')
        print(f"Wrote out mutations to {out_fn}", flush=True)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mut_fn", type=str, required=True, help="path to the mutation file. Must hav")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--clump_window_size", type=int, required=True)
    parser.add_argument("--training_samples_fn", type=str, required=True)
    args = parser.parse_args()
    mut_fn = args.mut_fn
    out_dir = args.out_dir
    clump_window_size = args.clump_window_size
    training_samples_fn = args.training_samples_fn
    fold_num = int(training_samples_fn.split('_')[-1].split('.')[0])
    # make the output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f"Reading in mutation file {mut_fn} and outputting to {out_dir}")
    # read in file
    mut_df = read_file(mut_fn, training_samples_fn)
    # partition and write to files, clumping if specified
    clump_partit_mutations(mut_df, clump_window_size, out_dir, fold_num)
    
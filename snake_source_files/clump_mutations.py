import pandas as pd
import os
import argparse

def clump_mutations(
    mut_fn: str, 
    out_dir: str, 
    window: int
    ) -> None:
    """
    Clump mutations in window and drop windows without enough mutations
    """
    
    
    
    return
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mut_fn", type=str, required=True, help="path to the mutation file. Must hav")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--window", type=int, required=True)
    args = parser.parse_args()
    mut_fn = args.mut_fn
    out_dir = args.out_dir
    window = args.window
    # make the output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f"Reading in mutation file {mut_fn} and outputting to {out_dir} after clumping mutations in {window} window and dropping those without enough mutations")
    clump_mutations(mut_fn, out_dir, window)
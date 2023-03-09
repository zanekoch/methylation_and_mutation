import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold
import os
import pickle

def split(
    covariate_fn: str,
    out_dir: str,
    num_folds: int
    ) -> None:
    # get samples and their ages from cov_fn
    cov_df = pd.read_csv(covariate_fn, header = 0, index_col=0).T
    # get training and testing samples
    skf = StratifiedKFold(n_splits=num_folds, random_state=10, shuffle=True)
    # get first fold training sampples
    for i, (train_index, test_index) in enumerate(skf.split(cov_df.index, cov_df['age_at_index'])):
        train_samples = cov_df.iloc[train_index].index.to_list()
        test_samples = cov_df.iloc[test_index].index.to_list()
        # write out training and testing samples to pkl files
        train_samples_fn = os.path.join(out_dir, f"train_samples_fold_{i}.pkl")
        test_samples_fn = os.path.join(out_dir, f"test_samples_fold_{i}.pkl")
        with open(train_samples_fn, 'wb') as fp:
            pickle.dump(train_samples, fp)
        with open(test_samples_fn, 'wb') as fp:
            pickle.dump(test_samples, fp)
        print(f"wrote to {train_samples_fn} and {test_samples_fn}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_folds", type=int, required=True)
    parser.add_argument("--covariate_fn", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    split(args.covariate_fn, args.out_dir, int(args.num_folds))

import pandas as pd
import numpy as np
import seaborn as sns
from rich.progress import track
import matplotlib.pyplot as plt
from itertools import combinations

class methylomeMutationalBurden:

    def __init__(self, all_mut_df, all_methyl_age_df_t, age_bin_size = 5):
        self.all_mut_df = all_mut_df
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.age_bin_size = age_bin_size

        self.preproc()

    def preproc(self):
        """
        Reformat and fdr correct correlation-based measured sites and distance-based comparison sites
        """
        # get mutation counts by sample
        self.mut_counts_df = self.all_mut_df['sample'].value_counts()
        self.ct_mut_counts_df = self.all_mut_df[self.all_mut_df['mutation'] == 'C>T']['sample'].value_counts()

    def _observed_methyl_change(self, sample, comparison_samples):
        """
        Get the observed change (Manhattan distance) of the methylome between sample and the comparison samples
        """
        # exclude age dataset, and gender columns from all_methyl_age_df_t
        cpg_cols = self.all_methyl_age_df_t.columns[3:]
        # subtract the methylome of sample from comparison samples
        comp_samples_df = self.all_methyl_age_df_t.loc[comparison_samples, :]
        methylome_diffs = comp_samples_df.loc[:, cpg_cols].subtract(self.all_methyl_age_df_t.loc[sample, cpg_cols])
        # switch sign so that neg diff means sample 1 is lower
        neg_diff_sum = methylome_diffs[methylome_diffs > 0].sum(axis=1)
        pos_diff_sum = methylome_diffs[methylome_diffs < 0].abs().sum(axis=1)
        abs_diff_sum = neg_diff_sum.add(pos_diff_sum, fill_value=0)
        # combine        
        diff_sum_df = pd.concat([abs_diff_sum, neg_diff_sum, pos_diff_sum, comp_samples_df['age_at_index'], comp_samples_df.index.to_series()], axis=1)
        diff_sum_df.columns = ['abs_diff_sum', 'neg_diff_sum', 'pos_diff_sum', 'comp_age', 'comp_sample']
        return diff_sum_df

    def compare_all_pairs(self):
        """
        Get the observed methylome manhattan distance, mutation count, and age between all pairs of samples
        """
        pair_comp_dfs = []
        for dset in self.all_methyl_age_df_t['dataset'].unique():
            for gender in ['FEMALE', 'MALE']:
                # get all methylomes that are from the same dataset and gender
                methylome_df = self.all_methyl_age_df_t.loc[(self.all_methyl_age_df_t['dataset'] == dset)
                                                        & (self.all_methyl_age_df_t['gender'] == gender)]
                if len(methylome_df) <= 0:
                    continue
                # remove age, gender, and dataset columns 
                methylome_df = methylome_df.iloc[:, 3:]
                # create iterator of all pairs of indexes in methylome_df
                pairs = combinations(methylome_df.index, 2)
                # iterate across all pairs, calculating the manhattan distance between the two samples
                compare_dict = {}
                for pair in pairs:
                    compare_dict[pair] = (methylome_df.loc[pair[0], :] - methylome_df.loc[pair[1], :]).abs().sum()
                # create df from dict with tuple as key
                compare_df = pd.Series(compare_dict).reset_index()
                compare_df.columns = ['sample1', 'sample2', 'manhattan_diff']
                # get mutation count difference between samples
                compare_df['mut1'] = compare_df['sample1'].map(self.mut_counts_df)
                compare_df['mut2'] = compare_df['sample2'].map(self.mut_counts_df)
                compare_df['mut_diff'] = (compare_df['mut1'] - compare_df['mut2']).abs()
                compare_df['ct_mut1'] = compare_df['sample1'].map(self.ct_mut_counts_df)
                compare_df['ct_mut2'] = compare_df['sample2'].map(self.ct_mut_counts_df)
                compare_df['ct_mut_diff'] = (compare_df['ct_mut1'] - compare_df['ct_mut2']).abs()
                # ages
                compare_df['age1'] = compare_df['sample1'].map(self.all_methyl_age_df_t['age_at_index']) # TODO: change this from all_methyl to smaller
                compare_df['age2'] = compare_df['sample2'].map(self.all_methyl_age_df_t['age_at_index'])
                compare_df['age_diff'] = (compare_df['age1'] - compare_df['age2']).abs()
                # dataset
                compare_df['dataset'] = dset
                pair_comp_dfs.append(compare_df)
                print("done with dataset {}".format(dset), flush=True)
        print("done comparing all pairs", flush=True)
        pair_comp_df = pd.concat(pair_comp_dfs, axis=0)
        return pair_comp_df

    def compare_pairs(self, num_samples = -1, same_age = True):
        """
        For every pair of samples 
        """
        all_results = []
        samples_done = 0
        # iterate across each dataset in order
        for dset in self.all_methyl_age_df_t['dataset'].unique():
            print(dset)
            dset_methyl_age_df_t = self.all_methyl_age_df_t[self.all_methyl_age_df_t['dataset'] == dset]
            # get a list of valid samples from this dataset to choose from, must have at least 1 ct mutation and methylation data
            samples_w_mut_and_methyl = list(set(dset_methyl_age_df_t.index) & set(self.ct_mut_counts_df.index))
            for sample in track(samples_w_mut_and_methyl, total=len(samples_w_mut_and_methyl), description="Comparing pairs"):
            #for sample in samples_w_mut_and_methyl:
                age = dset_methyl_age_df_t.loc[sample, 'age_at_index']
                gender = dset_methyl_age_df_t.loc[sample, 'gender']
                # get all other samples that can be compared to rand_sample (same dataset, same age bin)
                # TODO: add same sex to both of these
                if same_age:
                    same_age_dset_samples = dset_methyl_age_df_t[(dset_methyl_age_df_t['age_at_index'] >= age - self.age_bin_size/2)
                                                                & (dset_methyl_age_df_t['age_at_index'] <= age + self.age_bin_size/2)
                                                                & (dset_methyl_age_df_t['gender'] == gender)]
                    same_age_dset_samples = same_age_dset_samples.drop(sample)
                    comparison_samples = list(set(same_age_dset_samples.index.to_list()) & set(samples_w_mut_and_methyl))
                    if len(comparison_samples) == 0:
                        continue
                else:
                    comparison_samples = list(set(dset_methyl_age_df_t[dset_methyl_age_df_t['gender'] == gender].index.to_list()) & set(samples_w_mut_and_methyl))
                    if len(comparison_samples) == 0:
                        continue
                    comparison_samples.remove(sample)
                # get the observed change in methylome between sample and the comparison samples
                methylome_diffs = self._observed_methyl_change(sample, comparison_samples)
                # get the number of mutations in each sample
                methylome_diffs['mut_diff'] = np.abs(
                    self.mut_counts_df.loc[comparison_samples]
                    - self.mut_counts_df.loc[sample])
                methylome_diffs['ct_mut_diff'] = np.abs(
                    self.ct_mut_counts_df.loc[comparison_samples]
                    - self.ct_mut_counts_df.loc[sample])
                methylome_diffs.reset_index(inplace=True)
                methylome_diffs = methylome_diffs.rename(columns={'index': 'comp_sample'})
                methylome_diffs['age'] = age
                methylome_diffs['age_diff'] = np.abs(methylome_diffs['comp_age'] - methylome_diffs['age'])
                methylome_diffs['sample'] = sample
                methylome_diffs['dataset'] = dset
                all_results.append(methylome_diffs)
                samples_done += 1
                if num_samples > 0 and samples_done >= num_samples:
                    break
        # make a df from all_results
        pair_comp_df = pd.concat(all_results, axis=0)
        return pair_comp_df

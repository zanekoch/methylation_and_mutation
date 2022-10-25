import utils
import pandas as pd
import numpy as np
import seaborn as sns
from rich.progress import track
import random
import matplotlib.pyplot as plt


class methylomeMutationalBurden:
    def __init__(
        self, linked_sites_names_dfs, linked_sites_diffs_dfs, linked_sites_z_pvals_dfs, mut_nearby_measured_df, nearby_diffs_df, all_methyl_age_df_t, all_mut_df,mut_in_measured_cpg_w_methyl_age_df, cpg_in_body, age_bin_size = 10):
        self.linked_sites_names_df = linked_sites_names_dfs[0]
        self.linked_sites_diffs_df = linked_sites_diffs_dfs[0]
        self.linked_sites_z_pvals_df = linked_sites_z_pvals_dfs[0]
        self.mut_nearby_measured_df = mut_nearby_measured_df
        self.nearby_diffs_df = nearby_diffs_df
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.all_mut_df = all_mut_df
        self.mut_in_measured_cpg_w_methyl_age_df = mut_in_measured_cpg_w_methyl_age_df
        self.cpg_in_body = cpg_in_body
        self.age_bin_size = age_bin_size
        # preprocess
        self.preproc()


    def preproc(self):
        """
        Reformat and fdr correct correlation-based measured sites and distance-based comparison sites
        """
        # get mutation counts by sample
        self.mut_counts_df = self.all_mut_df['sample'].value_counts()
        self.ct_mut_counts_df = self.all_mut_df[self.all_mut_df['mutation'] == 'C>T']['sample'].value_counts()
        # combined linked site information
        corr_comp_sites = utils.stack_and_merge(self.linked_sites_diffs_df, self.linked_sites_z_pvals_df, self.linked_sites_names_df)
        # limit to 5 most linked sites 
        corr_comp_sites = corr_comp_sites[corr_comp_sites['comparison_site']<5]
        # get the sample in which each mutation occured
        corr_comp_sites['sample'] = corr_comp_sites.apply(lambda row: self.mut_in_measured_cpg_w_methyl_age_df[self.mut_in_measured_cpg_w_methyl_age_df['#id'] == row['mut_site']]['case_submitter_id'].values[0], axis=1)

        # fdr correct linked and distance-based sites
        corr_comp_sites = utils.fdr_correct(corr_comp_sites, 'pval')
        # fdr correct distance based sites also
        dist_comp_sites = utils.fdr_correct(self.nearby_diffs_df, 'ztest_pval')

        # rename both dfs columns to be consistent
        dist_comp_sites.columns = ['comp_site_name', 'delta_mf', 'ztest_pval', 'mut_site', 'comp_site_dist', 'mut_sample', 'sig', 'fdr_pval']
        corr_comp_sites.columns = ['mut_site', 'comparison_site', 'delta_mf', 'ztest_pval', 'comp_site_name', 'mut_sample', 'sig', 'fdr_pval']

        self.dist_comp_sites = dist_comp_sites
        self.corr_comp_sites = corr_comp_sites

    def mutation_impacts_by_sample(self):
        """
        Form a samples X CpG sites matrix of delta_mf's due to mutations
        """
        # pivot so that mut_sample is the index and mut_site is the columns, and entires are the *significant* delta_mfs
        corr_mut_impact_df = self.corr_comp_sites[self.corr_comp_sites['sig']==True].pivot_table(index='mut_sample', columns='comp_site_name', values='delta_mf')
        dist_mut_impact_df = self.dist_comp_sites[self.dist_comp_sites['sig']==True].pivot_table(index='mut_sample', columns='comp_site_name', values='delta_mf')

        # TODO: change this to take max abs values
        # combine distance and correlation-based sites for each sample
        mut_impact_df = corr_mut_impact_df.join(dist_mut_impact_df, how='outer', lsuffix='_corr', rsuffix='_dist')
        # if columns collide, keep the dist column
        mut_impact_df = mut_impact_df[mut_impact_df.columns[~mut_impact_df.columns.str.endswith('_corr')]]
        mut_impact_df.columns = mut_impact_df.columns.str.replace('_dist', '')

        self.mut_impact_df = mut_impact_df
        return mut_impact_df    

    def observed_methyl_change(self, sample, comparison_samples):
        """
        Get the observed change in methylome between sample and the comparison samples
        """
        # exclude age and dataset columns
        cpg_cols = self.all_methyl_age_df_t.columns[2:]
        # subtract the methylome of sample from comparison samples
        comp_samples_df = self.all_methyl_age_df_t.loc[comparison_samples, :]
        methylome_diffs = comp_samples_df.loc[:, cpg_cols].subtract(self.all_methyl_age_df_t.loc[sample, cpg_cols])
        # save as opposite sign so that neg diff means sample 1 is lower
        neg_diff_sum = methylome_diffs[methylome_diffs > 0].sum(axis=1)
        pos_diff_sum = methylome_diffs[methylome_diffs < 0].abs().sum(axis=1)
        abs_diff_sum = neg_diff_sum.add(pos_diff_sum, fill_value=0)
        # combine
        diff_sum_df = pd.concat([abs_diff_sum, neg_diff_sum, pos_diff_sum, comp_samples_df['age_at_index']], axis=1)
        diff_sum_df.columns = ['abs_diff_sum', 'neg_diff_sum', 'pos_diff_sum', 'comp_age']
        return diff_sum_df

    def compare_pairs(self, num_samples, same_age = True):
        """
        For every pair of samples 
        """
        all_results = []
        # list of valid samples to choose from, must at least 1 ct mutation and methylation data 
        samples_w_mut_and_methyl = list(set(self.all_methyl_age_df_t.index) & set(self.ct_mut_counts_df.index))
        for _ in track(range(num_samples)):
            # choose a random sample from self.methyl_age_df_t.index
            rand_sample = random.choice(samples_w_mut_and_methyl)
            age = self.all_methyl_age_df_t.loc[rand_sample, 'age_at_index']
            dataset = self.all_methyl_age_df_t.loc[rand_sample, 'dataset']
            # get all other samples that can be compared to rand_sample (same dataset, same age bin)
            if same_age:
                same_age_dset_samples = self.all_methyl_age_df_t[(self.all_methyl_age_df_t['age_at_index'] >= age - self.age_bin_size/2) & (self.all_methyl_age_df_t['age_at_index'] <= age + self.age_bin_size/2) & (self.all_methyl_age_df_t['dataset'] == dataset) ]
                same_age_dset_samples = same_age_dset_samples.drop(rand_sample)
                comparison_samples = list(set(same_age_dset_samples.index.to_list()) & set(samples_w_mut_and_methyl))
                if len(comparison_samples) == 0:
                    continue
                # choose at most 50 random samples from comparison_samples
                comparison_samples = random.sample(comparison_samples, min(50, len(comparison_samples)))
            else:
                same_dataset_samples = self.all_methyl_age_df_t[self.all_methyl_age_df_t['dataset'] == dataset]
                comparison_samples = list(set(same_dataset_samples.index.to_list()) & set(samples_w_mut_and_methyl))
                if len(comparison_samples) == 0:
                    continue
                comparison_samples.remove(rand_sample)
                # choose at most 50 random samples from comparison_samples no replacement
                comparison_samples = random.sample(comparison_samples, min(len(comparison_samples), 50))
            # get the observed change in methylome rand_sample and the comparison samples
            methylome_diffs = self.observed_methyl_change(rand_sample, comparison_samples)
            # get the number of mutations in each sample
            mut_counts_comp = self.mut_counts_df.loc[comparison_samples]
            mut_counts_sample = self.mut_counts_df.loc[rand_sample]
            methylome_diffs['mut_diff'] = np.abs(mut_counts_comp - mut_counts_sample)
            # same for ct mutations
            ct_mut_counts_comp = self.ct_mut_counts_df.loc[comparison_samples]
            ct_mut_counts_sample = self.ct_mut_counts_df.loc[rand_sample]
            methylome_diffs['ct_mut_diff'] = np.abs(ct_mut_counts_comp - ct_mut_counts_sample)
            # we want to end up with a df with rows being a pair of samples, their ages, methyl diffs, and mut diffs
            methylome_diffs.reset_index(inplace=True)
            methylome_diffs = methylome_diffs.rename(columns={'index': 'comp_sample'})
            methylome_diffs['age'] = age
            methylome_diffs['age_diff'] = np.abs(methylome_diffs['comp_age'] - methylome_diffs['age'])
            methylome_diffs['sample'] = rand_sample
            methylome_diffs['dataset'] = dataset
            all_results.append(methylome_diffs)
        # make a df from all_results
        pair_comp_df = pd.concat(all_results, axis=0)
        return pair_comp_df
    
    def _same_age_and_tissue_samples(self, sample_name):
        """
        Get the sample that has the mutation in the mutated CpG and the samples of the same age as that sample
        @ all_methyl_age_df_t: dataframe with columns=CpGs and rows=samples and entries=methylation fraction
        @ mut_in_measured_cpg_w_methyl_age_df
        @ mut_cpg: the mutated CpG
        @ returns: the mutated sample name and the samples of the same age and dset as the mutated sample
        """
        # get this sample's age and dataset
        this_age = self.all_methyl_age_df_t.loc[sample_name]['age_at_index']
        this_dset = self.all_methyl_age_df_t.loc[sample_name]['dataset']
        
        # get the mf all other samples of within age_bin_size/2 years of age on either side
        same_age_dset_samples_mf_df = self.all_methyl_age_df_t[(np.abs(self.all_methyl_age_df_t['age_at_index'] - this_age) <= self.age_bin_size/2) & (self.all_methyl_age_df_t['dataset'] == this_dset)]
        # drop the mutated sample itself
        same_age_dset_samples_mf_df = same_age_dset_samples_mf_df.drop(index = sample_name)
        
        return same_age_dset_samples_mf_df

    def _plot_mf_change(self, num_samples):
        """
        Line plot of each starting MF of comparison site and ending MF of comparison site for sites with a signiciant difference
        """
        # self.mut_impact_df already is only sig sites at FDR <.05
        start_mfs = []
        end_mfs = []
        for sample, row in self.mut_impact_df.iloc[:num_samples, :].iterrows():
            same_age_tissue_mean_mf = self._same_age_and_tissue_samples(sample).mean(axis=0)
            # get the column name of non-nan values in row
            sig_change_cpgs = row[row.notna()]
            sig_change_cpgs_names = sig_change_cpgs.index
            sig_change_cpgs_amounts = sig_change_cpgs.to_numpy()
            start_mfs += same_age_tissue_mean_mf[sig_change_cpgs_names].to_list()
            # add the delta_mf to get ending CpG mfs
            end_mfs += list(same_age_tissue_mean_mf[sig_change_cpgs_names].to_numpy() + sig_change_cpgs_amounts)
        # make a df with start_mfs and end_mfs as rows
        mf_change_df = pd.DataFrame({'Non-mutated samples\nmean': start_mfs, 'Mutated sample': end_mfs})
        mf_change_df['Direction'] = np.where(mf_change_df['Mutated sample'] > mf_change_df['Non-mutated samples\nmean'], 'increase', 'decrease')
        mf_change_df.reset_index(inplace=True)
        melted = pd.melt(mf_change_df, id_vars=['Direction', 'index'], value_vars=['Non-mutated samples\nmean', 'Mutated sample'])
        melted.columns = ['Direction of change', 'index', 'Mutation status', 'Methylation Fraction']
        # tight layout
        fig, axes = plt.subplots(figsize=(6, 6), dpi=100, squeeze=True)

        sns.lineplot(data=melted[melted['Direction of change'] == 'decrease'], x='Mutation status', y='Methylation Fraction', estimator=None, units='index', hue='index', palette='Purples', alpha=0.3, legend=False)
        sns.lineplot(data=melted[melted['Direction of change'] == 'increase'], x='Mutation status', y='Methylation Fraction', estimator=None, units='index', hue='index', palette='Blues', alpha=0.3, legend=False)
        # left justify x axes ticks





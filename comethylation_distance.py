import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import utils
import seaborn as sns
from statsmodels.stats.weightstats import ztest as ztest
from rich.progress import track
import sys
from statsmodels.stats.multitest import fdrcorrection



class mutationScanDistanceMulti:
    def __init__(self, all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, age_bin_size = 5, max_dist = 1000):
        self.all_mut_w_age_df = all_mut_w_age_df
        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.age_bin_size = age_bin_size
        self.max_dist = max_dist

    def _join_with_illum(self, in_df):
        """
        Join the dataframe with the illumina_cpg_locs_df
        """
        df = in_df.copy(deep=True)
        # convert start column to int with to_numeric
        df['start'] = pd.to_numeric(df['start'])
        df_w_illum = df.merge(self.illumina_cpg_locs_df, on=['chr', 'start'], how='left')
        return df_w_illum

    def _fdr_correct(self, all_metrics_df):
        # because each of the samples at a comparison site has the same pval, we drop all but on of these so we are not overcorrecting
        unique_tests = all_metrics_df.drop_duplicates(subset=['mut_cpg', 'measured_site'])
        unique_tests.loc[:,'sig'], unique_tests.loc[:,'fdr_pval'] = fdrcorrection(unique_tests.loc[:, 'mwu_pval'], alpha=0.05)
        # merge back in the non-unique tests
        all_metrics_df = all_metrics_df.merge(unique_tests[['mut_cpg', 'measured_site', 'sig', 'fdr_pval']], on=['mut_cpg', 'measured_site'], how='left')
        return all_metrics_df

    def _detect_effect_in_other_samples(self, sites_to_test: list, mut_cpg: str, same_age_samples: list):
        """
        @ sites_to_test: list of sites to test
        @ same_age_samples: list of samples to test
        @ returns: list of samples that do not have any C>T mutations in max_dist from any of the sites in sites_to_test
        """
        # detect samples that have a mutation in the max_dist window of any of the sites in sites_to_test
        sites_to_test_locs = self.illumina_cpg_locs_df.loc[self.illumina_cpg_locs_df['#id'].isin(sites_to_test)]
        # get the mutations which are on the same chromosome and within max_dist of a location in sites_to_test_locs (works because all the sites to test are on the same chromosome)
        have_illegal_muts = self.all_mut_w_age_df.loc[self.all_mut_w_age_df['chr'].isin(sites_to_test_locs['chr']) & (np.abs(self.all_mut_w_age_df['start'] - sites_to_test_locs['start']) <= self.max_dist)]

        # also detect samples that have a mutation in the mut_cpg
        have_illegal_muts = have_illegal_muts.append(self.all_mut_w_age_df.loc[self.all_mut_w_age_df['mut_cpg'] == mut_cpg])

        return have_illegal_muts['case_submitter_id'].to_list()

    def _same_age_and_tissue_samples(self, mut_samples, dataset):
        """
        Get the sample that has the mutation in the mutated CpG and the samples of the same age as that sample
        @ mut_samples: list of samples
        @ returns: the samples of the same age and dset as the mutated sample which are not in the mut_samples list
        """
        # get this sample's age and dataset
        this_age = self.all_methyl_age_df_t.loc[mut_samples, 'age_at_index'].median()
        # get the mf all other samples of within age_bin_size/2 years of age on either side
        same_age_dset_samples = self.all_methyl_age_df_t[(np.abs(self.all_methyl_age_df_t['age_at_index'] - this_age) <= self.age_bin_size/2) & (self.all_methyl_age_df_t['dataset'] == dataset)].index.copy(deep=True)
        # drop the mutated samples
        # TODO: sometimes this raises an error bc of the two methods of age binning
        same_age_dset_samples = same_age_dset_samples.drop(mut_samples, errors='ignore')
        return same_age_dset_samples.to_list()

    def _compare_sites(self, same_age_tissue_methyl_df, mut_samples_methyl_df):
        """
        Get the delta_mf and ztest pvalue for each site in each sample, mutated and non-mutated
        @ returns:
            - all_metrics: a df of sample and site pairs with columns specifying the delta_mf_mean, delta_mf_median, modified zscore, and mwu pvalue of mutated samples vs non-mutated samples
        """
        # create a new dataframe all_samples_at_comparison_sites with same_age_tissue_methyl_df as first rows and mut_samples_methyl_df as last rows
        all_samples_at_comparison_sites = same_age_tissue_methyl_df.append(mut_samples_methyl_df)
        # get difference of each sample from the mean and median
        mean = all_samples_at_comparison_sites.mean(axis=0)
        mean_diff = all_samples_at_comparison_sites.subtract(mean)
        median = all_samples_at_comparison_sites.median(axis=0)
        median_diff = all_samples_at_comparison_sites.subtract(median)
        # median absolute deviation
        CONSISTENCY_CONST = 0.6745
        modified_zscores = all_samples_at_comparison_sites.apply(lambda col: CONSISTENCY_CONST * (col - col.median())/ stats.median_absolute_deviation(col, nan_policy='omit'), axis=0)
        # mann whitney U test of mutated samples vs non-mutated samples in each column
        mwu_pvals = all_samples_at_comparison_sites.apply(lambda col: stats.mannwhitneyu(col[:len(same_age_tissue_methyl_df)], col[len(same_age_tissue_methyl_df):], alternative='two-sided').pvalue, axis=0)
        
        # matrix of z scores of each sample at each site being different from the other samples
        zscores = all_samples_at_comparison_sites.apply(lambda col: stats.zscore(col, nan_policy='omit'), axis=0)
        # convert to 2 sided pvalues
        ztest_pvals = stats.norm.sf(abs(zscores))*2
        ztest_pvals = pd.DataFrame(ztest_pvals, index=all_samples_at_comparison_sites.index, columns=all_samples_at_comparison_sites.columns)

        # stack and combine
        mean_diff_stacked = mean_diff.stack().reset_index()
        mean_diff_stacked.columns = ['sample', 'measured_site', 'delta_mf_mean']
        median_diff_stacked = median_diff.stack().reset_index()
        median_diff_stacked.columns = ['sample', 'measured_site', 'delta_mf_median']
        modified_zscores_stacked = modified_zscores.stack().reset_index()
        modified_zscores_stacked.columns = ['sample', 'measured_site', 'modified_zscore']
        ztest_pvals_stacked = ztest_pvals.stack().reset_index()
        ztest_pvals_stacked.columns = ['sample', 'measured_site', 'ztest_pval']
        all_metrics = mean_diff_stacked.merge(median_diff_stacked, on=['sample', 'measured_site'], how='left')
        all_metrics = all_metrics.merge(modified_zscores_stacked, on=['sample', 'measured_site'], how='left')
        all_metrics = all_metrics.merge(ztest_pvals_stacked, on=['sample', 'measured_site'], how='left')
        all_metrics['mwu_pval'] = mwu_pvals.loc[all_metrics['measured_site']].values
        
        return all_metrics
        
    def effect_on_each_site(self, mut_nearby_measured_df):
        """
        For each mutation, get the effect of the mutation on each measured CpG within max_dist of the mutation in the mutated samples
        """
        num_skipped = 0
        num_dropped = 0
        all_metrics_dfs = []
        #for _, mut_row in track(mut_nearby_measured_df.iterrows(), description="Analyzing each mutation", total=len(mut_nearby_measured_df)):
        for _, mut_row in mut_nearby_measured_df.iterrows():
            # get the same age and dataset samples
            same_age_dset_samples = self._same_age_and_tissue_samples(mut_row['samples'], mut_row['dataset'])
            # exclude samples that have ANY mutations within max_dist of a close_measured site
            samples_to_exclude = self._detect_effect_in_other_samples(mut_row['close_measured'], mut_row['mut_cpg'], same_age_dset_samples)

            # drop entries of same_age_dset_samples that are in samples_to_exclude
            before_drop = len(same_age_dset_samples)
            same_age_dset_samples = [s for s in same_age_dset_samples if s not in samples_to_exclude]
            after_drop = len(same_age_dset_samples)
            num_dropped += before_drop - after_drop

            if len(same_age_dset_samples) <= 10:
                num_skipped += 1
                continue
            # get these mutated samples MFs at comparison sites
            nearby_sites = mut_row['close_measured']
            mut_samples_nearby_mfs = self.all_methyl_age_df_t.loc[mut_row['samples'], nearby_sites]
            same_age_nearby_mfs = self.all_methyl_age_df_t.loc[same_age_dset_samples, nearby_sites]
            # measure the change in methylation between sites in the mutated samples and in other non-mutated samples of the same age
            metrics = self._compare_sites(same_age_nearby_mfs, mut_samples_nearby_mfs)
            metrics['mut_cpg'] = mut_row['mut_cpg']
            # create column called 'mutated' that is True if the sample is one of the mutated samples
            metrics['mutated'] = metrics['sample'].isin(mut_row['samples'])
            cpg_to_dist_dict = dict(zip(mut_row['close_measured'], mut_row['close_measured_dists']))
            metrics['measured_site_dist'] = metrics['measured_site'].map(cpg_to_dist_dict)
            all_metrics_dfs.append(metrics)
        all_metrics_df = pd.concat(all_metrics_dfs)
        print("NOTE: Not enough samples of the same age and tissue to calculate effect of mutation for {} mutations".format(num_skipped), flush=True)
        print("NOTE: Dropped {} samples due to mutations in other samples".format(num_dropped), flush=True)
        return all_metrics_df

    def find_nearby_measured_cpgs(self):
        """
        Find the measured cpgs within max_dist of each mutation
        @ returns: df of mutations that have at least one measured CpG within max_dist of the mutation. 'close_measured' column is a list of the measured cpgs within max_dist of the mutation.
        """
        mut_nearby_measured_l = []
        chroms = self.muts_grouped['chr'].unique()
        #for chrom in track(chroms, description = 'Finding nearby measured cpgs'):
        for chrom in chroms:
            illum_locs = self.illumina_cpg_locs_df[self.illumina_cpg_locs_df['chr'] == chrom]
            mut_locs = self.muts_grouped.loc[self.muts_grouped['chr'] == chrom]
            # for each mutation, get a list of the measured CpGs #id in illum_locs that are within max_dist of the mutation 'start' but 0 distance (the same CpG)
            mut_locs.loc[:, 'close_measured'] = mut_locs.apply(lambda x: list(illum_locs[(np.abs(x['start'] - illum_locs['start']) <= self.max_dist) & (x['start'] -illum_locs['start'] != 0)]['#id']), axis = 1)
            # also get a list of the distances of these sites
            mut_locs.loc[:, 'close_measured_dists'] = mut_locs.apply(lambda x: list(illum_locs[(np.abs(x['start'] - illum_locs['start']) <= self.max_dist) & (x['start'] -illum_locs['start'] != 0)]['start'] - x['start']), axis = 1)
            # drop all rows of mut_locs where close_measured is empty
            mut_locs = mut_locs[mut_locs['close_measured'].apply(lambda x: len(x) > 0)]
            mut_nearby_measured_l.append(mut_locs)
        mut_nearby_measured_df = pd.concat(mut_nearby_measured_l)
        mut_nearby_measured_df['mut_cpg'] = mut_nearby_measured_df['chr'] + ':' + mut_nearby_measured_df['start'].astype(str)
        return mut_nearby_measured_df

    def look_for_disturbances(self, min_VAF_percentile):
        """
        Driver for the analysis. Finds group of samples with mutations with VAF >= min_VAF_percentile that have a measured CpG within max_dist of the mutation and then looks for disturbances in the methylation of these CpGs.
        @ max_dist: maximum distance between mutation and measured CpG to be considered
        @ min_VAF: minimum VAF of mutation to be considered
        """
        # subset illumina_cpg_locs_df to only the CpGs that are measured
        self.illumina_cpg_locs_df = self.illumina_cpg_locs_df[self.illumina_cpg_locs_df['#id'].isin(self.all_methyl_age_df_t.columns)]
        # subset to only mutations that are C>T, non X and Y chromosomes, and that occured in samples with measured methylation
        self.all_mut_w_age_df = self.all_mut_w_age_df[(self.all_mut_w_age_df['mutation'] == 'C>T')
         & (self.all_mut_w_age_df['chr'] != 'X') & (self.all_mut_w_age_df['chr'] != 'Y')
         & (self.all_mut_w_age_df['case_submitter_id'].isin(self.all_methyl_age_df_t.index))
         & (self.all_mut_w_age_df['DNA_VAF'] >= np.percentile(self.all_mut_w_age_df['DNA_VAF'], min_VAF_percentile))]
        # create new column 'age_bin' where every age_at_index is rounded to nearest age_bin_size
        self.all_mut_w_age_df.loc[:,'age_bin'] = self.all_mut_w_age_df['age_at_index'].apply(lambda x: round(x/self.age_bin_size)*self.age_bin_size)

        # group mutations by dataset, location, and age_bin
        muts_grouped = self.all_mut_w_age_df.groupby(['dataset', 'chr', 'start', 'age_bin']).apply(lambda x: x['case_submitter_id'].unique()).reset_index()
        muts_grouped.columns = ['dataset', 'chr', 'start', 'age_bin', 'samples']
        # keep only those mutations that have at least 2 samples with the mutation
        self.muts_grouped = muts_grouped[muts_grouped['samples'].apply(lambda x: len(x) > 1)]

        self.all_mut_w_age_df['mut_cpg'] = self.all_mut_w_age_df['chr'] + ':' + self.all_mut_w_age_df['start'].astype(str)
        # for each mutation, get a list of the CpGs #id in illum_locs that are within max_dist of the mutation 'start'
        mut_nearby_measured_df = self.find_nearby_measured_cpgs()
        # for each mutation with nearby measured site, compare the methylation of the nearby measured sites in the mutated samples to the other samples of same age and dataset
        all_metrics_df = self.effect_on_each_site(mut_nearby_measured_df)
        # fdr correct pvals
        all_metrics_df = self._fdr_correct(all_metrics_df)
        self.all_metrics_df = all_metrics_df

        return mut_nearby_measured_df, all_metrics_df



class mutationScanDistance:
    def __init__(self, all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, age_bin_size = 5, max_dist = 500):
        self.all_mut_w_age_df = all_mut_w_age_df
        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.age_bin_size = age_bin_size
        self.max_dist = max_dist

    def volcano_plot(self, nearby_site_diffs_df):
        """
        Plot a volcano plot of the nearby_site_diffs_df
        """
        # get the log10 of the pvals
        nearby_site_diffs_df['log10_pval'] = nearby_site_diffs_df['fdr_pval'].apply(lambda x: -np.log10(x))
        # get the log2 of the fold change
        # color points orange if they are significant
        # put legend in upper left
        sns.scatterplot(y = 'log10_pval', x = 'delta_mf', data = nearby_site_diffs_df, alpha=0.3, hue = 'sig', palette = {True: 'orange', False: 'grey'})
        plt.legend(loc='upper left')
        plt.xlabel(r"$\Delta$MF")
        plt.ylabel('-log10 FDR pval')
        plt.show()

    def effect_violin(self, nearby_diffs_w_illum_df, mut_in_measured_cpg_w_methyl_age_df, pval,  sig_thresh = .05):
        """
        Make a violin plot of the effect of mutations on the nearby measured cpgs
        """
        fig, axes = plt.subplots(1,2, figsize=(14, 5), dpi=100, gridspec_kw={'width_ratios': [1, 5]}, sharey=True, constrained_layout=True)
        # subset to only significant sites
        nearby_diffs_w_illum_df = nearby_diffs_w_illum_df[nearby_diffs_w_illum_df[pval] < sig_thresh]
        # create 5 equal width bins of distances and assign each row to a distance bin
        nearby_diffs_w_illum_df.loc[:,'measured_site_dist'] = np.abs(nearby_diffs_w_illum_df['measured_site_dist'])
        nearby_diffs_w_illum_df.loc[:,'dist_bin'] = pd.cut(nearby_diffs_w_illum_df['measured_site_dist'], bins=5, labels=['1-500', '500-1000', '1000-1500', '1500-2000', '2000-2500'])
        #nearby_diffs_w_illum_df.loc[:,'dist_bin'] = pd.cut(nearby_diffs_w_illum_df['measured_site_dist'], bins=5, labels=['1-100', '101-200', '201-300', '301-400', '401-500'])

        # create a violin plot of the effect of mutations on the nearby measured cpgs
        sns.violinplot(data=nearby_diffs_w_illum_df, x='dist_bin', y='delta_mf', hue = 'mutated', split=True, cut=0, inner="quartile", palette=['steelblue', 'maroon'], axes=axes[1])
        #_ = sns.stripplot(data=nearby_diffs_w_illum_df, x='dist_bin', y='delta_mf', color="black", edgecolor="black", alpha=0.2,axes=axes[1])
        axes[1].tick_params(axis='y', labelleft=True)
        axes[1].set_ylabel(r"$\Delta$MF significantly (FDR<.05) affected nearby CpGs")
        axes[1].set_xlabel("")

        # select the rows of mut_in_measured_cpg_w_methyl_age_df that 
        # rename case_submitter_id to sample
        mut_in_measured_cpg_w_methyl_age_df_tm = mut_in_measured_cpg_w_methyl_age_df.rename(columns={'case_submitter_id': 'sample'})
        # get tested muts that are in measured cpgs
        tested_mut_in_measured_cpg = nearby_diffs_w_illum_df.loc[~nearby_diffs_w_illum_df['#id'].isna()]
        # merge mut_in_measured with mut_in_measured_cpg_w_methyl_age_df on #id and sample at the same time
        tested_mut_in_measured_cpg = tested_mut_in_measured_cpg.merge(mut_in_measured_cpg_w_methyl_age_df_tm, on=['#id', 'sample'], how='inner')

        tested_mut_in_measured_cpg = tested_mut_in_measured_cpg[tested_mut_in_measured_cpg[pval] < sig_thresh]

        p2 = sns.violinplot(data=tested_mut_in_measured_cpg, y='difference', ax=axes[0], color='maroon', cut=0, inner="quartile")
        #_ = sns.stripplot(data=tested_mut_in_measured_cpg, y='difference', color="black", edgecolor="black", alpha=0.2, ax=axes[0])
        # remove x ticks and x tick labels
        p2.set_xticks([0])
        p2.set_xticklabels(["0"])
        p2.set_ylabel(r"$\Delta$MF of mutated CpGs")

        # set sup x label
        fig.text(0.5, -0.05, 'Distance of affected site from mutation', ha='center')        

    def effect_violin_new(self, cumul_eff_by_dist):
        """
        Make a violin plot of the effect of mutations on the nearby measured cpgs
        """
        _, axes = plt.subplots(figsize=(10, 5), dpi=100)
        # create a violin plot of the effect of mutations on the nearby measured cpgs
        sns.violinplot(
            data=cumul_eff_by_dist, x='dist_bin', y='cumul_delta_mf', hue='mutated', cut=0, inner="quartile",
            split=False, color='steelblue', axes=axes, scale='width'
            )
        axes.tick_params(axis='y', labelleft=True)
        axes.set_ylabel(r"$\Delta$MF significantly (FDR<.05) affected nearby CpGs")
        axes.set_xlabel("")

    def prop_proximal_effected(self, nearby_diffs_w_illum_df):
        """prop_nearby_effected = nearby_diffs_w_illum_df.groupby(['mut_cpg', 'mutated'])['sig'].mean().reset_index()
        prop_nearby_effected.columns = ['mut_cpg', 'Mutated Sample', 'prop_sig']"""
        prop_nearby_effected = nearby_diffs_w_illum_df.groupby(['mut_cpg', 'mutated', 'sample'])['sig'].mean().reset_index()
        prop_nearby_effected.columns = ['mut_cpg', 'Mutated Sample', 'sample', 'prop_sig']
        # plot each as a seaborn distplot
        fig, axes = plt.subplots(1, 2, figsize=(12,5), dpi=100, tight_layout=True)
        sns.kdeplot(
            data=prop_nearby_effected, x="prop_sig", hue="Mutated Sample",
            common_norm=False, ax = axes[0], fill=True, clip=[0,1], palette=['steelblue', 'maroon'], alpha=.5, legend=False, bw_method=0.5
            )
        axes[0].set_ylabel('Density')
        axes[0].legend(title=None, loc='upper right', labels=['Mutated sample', 'Non-mutated sample'])
        # same plot but small y limit
        sns.kdeplot(
            data=prop_nearby_effected, x="prop_sig", hue="Mutated Sample",
            common_norm=False, ax = axes[1], fill=True, clip=[0,1], palette=['steelblue', 'maroon'], alpha=.5, legend=False, bw_method=0.5
            )
        axes[1].set_ylabel('Density')
        axes[1].legend(title=None, loc='upper right', labels=['Mutated sample', 'Non-mutated sample'])
        axes[1].set_ylim(0, .1)
        # set super x label
        fig.text(0.5, -0.05, 'Proportion of CpGs proximal (+- 2.5kbp) to mutation with significant change', ha='center')

    def mut_nearby_methyl_status_effect(self, nearby_site_diffs_df):
        """
        Plot the delta_mf for the 4 possible pairings of hyper and hypo methylated mutated and nearby sites
        """
        # limit to significant sites
        nearby_site_diffs_df = nearby_site_diffs_df[nearby_site_diffs_df['fdr_pval'] < .05]
        measured_site_means = {}
        # get the mean methylation of each 'measured_site' in the same age and tissue samples
        for _, row in track(nearby_site_diffs_df.iterrows(), total=len(nearby_site_diffs_df)):
            same_age_tissue_samples_mf_df = self._same_age_and_tissue_samples(row['sample'])
            # TODO make ignore samples with a mutation nearby
            # get the mean methylation of the measured site and #id (mutated site) in the same age and tissue samples
            measured_site_means[row['measured_site']] = same_age_tissue_samples_mf_df.loc[:, row['measured_site']].mean()
        # join the means to the nearby_site_diffs_w_illum_df on the measured_site and #id
        nearby_site_diffs_df['measured_site_mean'] = nearby_site_diffs_df['measured_site'].map(measured_site_means)
        # classify each site as hypermethylated if >.7, hypomethylated if <.3, and intermediate if between
        nearby_site_diffs_df['Mean methylation status in non-mutated individuals of CpGs nearby mutation'] = nearby_site_diffs_df['measured_site_mean'].apply(lambda x: 'Hypermethylated' if x >= .8 else ('Hypomethylated' if x <= .2 else 'intermediate'))
        # drop rows where the measured site is intermediate
        nearby_site_diffs_df = nearby_site_diffs_df[nearby_site_diffs_df['Mean methylation status in non-mutated individuals of CpGs nearby mutation'] != 'intermediate']
        # plot the delta_mf for the 4 possible pairings of hyper and hypo methylated mutated and nearby sites
        fig, axes = plt.subplots(figsize=(10, 5), dpi=100)
        sns.stripplot(data=nearby_site_diffs_df, x='Mean methylation status in non-mutated individuals of CpGs nearby mutation', y='delta_mf', color="black", edgecolor="black", alpha=0.3, ax=axes)
        sns.violinplot(data=nearby_site_diffs_df, x='Mean methylation status in non-mutated individuals of CpGs nearby mutation', y='delta_mf', cut=0, inner="quartile", palette=['steelblue', 'white'], ax=axes)
        axes.set_ylabel(r"$\Delta$MF")
        return

    def _join_with_illum(self, in_df):
        """
        Join the dataframe with the illumina_cpg_locs_df
        """
        df = in_df.copy(deep=True)
        # split 'mut_cpg' into 'chr' and 'start'
        df[['chr', 'start']] = df['mut_cpg'].str.split(':', expand=True)
        # convert start column to int with to_numeric
        df['start'] = pd.to_numeric(df['start'])
        df_w_illum = df.merge(self.illumina_cpg_locs_df, on=['chr', 'start'], how='left')
        return df_w_illum

    def plot_heatmap(self, mut_sample_cpg, nearby_site_diffs_w_illum_df, mut_nearby_measured_w_illum_df, remove_other_muts=True):
        """
        Given a set of linked sites, nonlinked sites, mutated sample, and mutated site, plots a heatmap of the methylation fraction of same age samples at the linked, nonlinked, and mutated sites
        @ mut_sample_cpg: sample_chr:start
        @ linked_sites_names_df: dataframe of linked sites names
        @ nonlinked_sites_names_df: dataframe of nonlinked sites names
        @ mut_in_measured_cpg_w_methyl_age_df: dataframe of mutations in samples
        @ all_methyl_age_df_t: dataframe of methylation data with ages attached
        """
        # join with illumina_cpg_locs_df to get the CpG names of mutated CpGs if they were in a measured site
        """nearby_site_diffs_w_illum_df = self._join_with_illum(nearby_site_diffs_df)
        mut_nearby_measured_w_illum_df = self._join_with_illum(mut_nearby_measured_df)"""
        # get sample name and position of mutation
        mut_sample_name = mut_sample_cpg.split('_')[0]
        # get the names of the nearby sites
        nearby_sites = mut_nearby_measured_w_illum_df[mut_nearby_measured_w_illum_df['mut_cpg'] == mut_sample_cpg]['close_measured'].values[0]
        # get which of the nearby sites were signficant
        sig_status = nearby_site_diffs_w_illum_df[(nearby_site_diffs_w_illum_df['mut_cpg'] == mut_sample_cpg)]['sig'].to_list()
        # get the same age and dataset samples
        same_age_dset_samples_mf_df = self._same_age_and_tissue_samples(mut_sample_name)

        # get same age and tissue samples  
        same_age_tissue_samples_all = same_age_dset_samples_mf_df.index.values
        # do not include samples that have a mut in the same CpG
        if remove_other_muts:
            nearby_and_mut_sites = np.append(nearby_sites, mut_sample_cpg)
            samples_to_exclude = self._detect_effect_in_other_samples(nearby_and_mut_sites, mut_nearby_measured_w_illum_df)
        # remove samples that have a mutation in the same CpG, a mutation in a nearby CpG, or a mutation in a nearby measured CpG
        same_age_tissue_samples = np.setdiff1d(same_age_tissue_samples_all, samples_to_exclude)
        print("{} samples excluded".format(len(same_age_tissue_samples_all) - len(same_age_tissue_samples)))

        # get the distances of each site from the mutated site
        distances = mut_nearby_measured_w_illum_df[mut_nearby_measured_w_illum_df['mut_cpg'] == mut_sample_cpg]['close_measured_dists'].values[0]
        # sort nearby_sites and sig_status by distances, then sort distances in the same way
        nearby_sites = [x for _, x in sorted(zip(distances, nearby_sites))]
        sig_status = [x for _, x in sorted(zip(distances, sig_status))]
        distances = sorted(distances)

        # if the mutated CpG was measured, add it to sites_to_plot
        if not nearby_site_diffs_w_illum_df[nearby_site_diffs_w_illum_df['mut_cpg'] == mut_sample_cpg]['#id'].isna().any():
            measured_mutated_cpg = nearby_site_diffs_w_illum_df[nearby_site_diffs_w_illum_df['mut_cpg'] == mut_sample_cpg]['#id'].values[0]
            # insert the mutated CpG into the nearby_sites where the distance goes from negative to positive
            for i in range(len(distances)):
                if distances[i] > 0:
                    nearby_sites.insert(i, measured_mutated_cpg)
                    distances.insert(i, 0)
                    sig_status.insert(i, True)
                    mut_pos = i
                    break
            if max(distances) < 0:
                nearby_sites.append(measured_mutated_cpg)
                distances.append(0)
                sig_status.append(True)
                mut_pos = len(distances) - 1
            measured_mut = True
        else:
            measured_mut = False
            for i in range(len(distances)):
                if distances[i] > 0:
                    mut_pos = i
                    break
            if max(distances) < 0:
                mut_pos = len(distances) - 1
        # for each nearby site, check if it is in self.all_methyl_age_df_t.columns
        # if it is, do nothing
        # if it is not, remove it and remove the corresponding distance and sig_status
        for i in range(len(nearby_sites)-1):
            if nearby_sites[i] not in self.all_methyl_age_df_t.columns:
                if i == mut_pos:
                    print("mutated site not in measured sites")
                nearby_sites.pop(i)
                distances.pop(i)
                sig_status.pop(i)
        # list of samples to plot
        samples_to_plot = np.concatenate((utils.half(same_age_tissue_samples, 'first'), [mut_sample_name], utils.half(same_age_tissue_samples, 'second')))
        # select cpgs and samples to plot
        to_plot_df = self.all_methyl_age_df_t.loc[samples_to_plot, nearby_sites]
        _, axes = plt.subplots(figsize=(15,10))
        # make color bar go from 0 to 1 and increase color bar size
        ax = sns.heatmap(to_plot_df, annot=False, center=0.5, xticklabels=False, yticklabels=False, cmap="Blues", cbar_kws={'label': 'Methylation fraction'}, ax=axes, vmin=0, vmax=1)
        ax.figure.axes[-1].yaxis.label.set_size(13)
        # highlight the mutated cpg if it was measured
        if measured_mut:
            # make a dashed rectangle
            ax.add_patch(Rectangle((mut_pos, int(len(utils.half(same_age_tissue_samples, 'first')))), 1, 1, fill=False, edgecolor='red', lw=1, ls='--'))
        else:
            ax.add_patch(Rectangle((mut_pos, int(len(utils.half(same_age_tissue_samples, 'first')))), 0, 1, fill=False, edgecolor='red', lw=3, ls='--'))
        ax.set_xticks(np.arange(.5, len(nearby_sites)+.5, 1))
        ax.set_xticklabels([str(distances[i]) + '*' if sig_status[i] == True else str(distances[i]) for i in range(len(distances))])
        ax.set_yticks(np.arange(.5, len(samples_to_plot)+.5, 1))
        # add a y tick for the mutated sample
        ax.set_yticks([int(len(utils.half(same_age_tissue_samples, 'first')))+.5])
        # make tick label red and rotate 45 degrees
        ax.set_yticklabels([mut_sample_name], color='red', rotation=90)
        axes.set_xlabel("Nearby CpG sites distance (bp)")
        axes.set_ylabel("Samples with same tissue and age (+- 5 years) as mutated sample")
        # increase tick label and axes label size
        ax.tick_params(axis='both', which='major', labelsize=13)
        axes.xaxis.label.set_size(15)
        axes.yaxis.label.set_size(15)
        return to_plot_df


    def _same_age_and_tissue_samples(self, sample_name: str) -> list:
        """
        Get the sample that has the mutation in the mutated CpG and the samples of the same age as that sample
        @ all_methyl_age_df_t: dataframe with columns=CpGs and rows=samples and entries=methylation fraction
        @ mut_cpg: the mutated CpG
        @ returns: the mutated sample name and the samples of the same age and dset as the mutated sample
        """
        # get this sample's age and dataset
        this_age = self.all_methyl_age_df_t.loc[sample_name]['age_at_index']
        this_dset = self.all_methyl_age_df_t.loc[sample_name]['dataset']
        # get the mf all other samples of within age_bin_size/2 years of age on either side
        same_age_dset_samples = self.all_methyl_age_df_t.loc[(np.abs(self.all_methyl_age_df_t['age_at_index'] - this_age) <= self.age_bin_size/2) & (self.all_methyl_age_df_t['dataset'] == this_dset)].index.copy(deep=True)
        # drop the mutated sample itself
        same_age_dset_samples = same_age_dset_samples.drop(sample_name)
        return same_age_dset_samples.to_list()

    def _detect_effect_in_other_samples(self, sites_to_test: list, mut_row: pd.DataFrame):
        """
        @ sites_to_test: list of sites to test, cg######### format
        @ mut_row: the row specifying the mutation
        @ returns: list of samples that do not have any C>T mutations in max_dist from any of the sites in sites_to_test
        """
        # limit to this chromosome
        relevant_mutations = self.all_mut_w_age_df.loc[(self.all_mut_w_age_df['chr'] == mut_row['chr']) & (self.all_mut_w_age_df['dataset'] == mut_row['dataset'])]
        
        # detect samples that have a mutation in the max_dist window of any of the sites in sites_to_test
        # get locations of sites to test
        sites_to_test_locs = self.illumina_cpg_locs_df.loc[self.illumina_cpg_locs_df['#id'].isin(sites_to_test)] 
        # select rows of relevant_mutations that are within max_dist of any of the sites_to_test
        have_illegal_muts = relevant_mutations.loc[relevant_mutations.apply(lambda row: any(np.abs(row['start'] - sites_to_test_locs['start']) <= self.max_dist), axis=1)]

        # detect samples that have a mutation in the mut_cpg or within max_dist of it
        have_illegal_muts = have_illegal_muts.append(relevant_mutations.loc[(relevant_mutations['mut_cpg'] == mut_row['mut_cpg']) | (np.abs(relevant_mutations['start'] - mut_row['start']) <= self.max_dist)])
        
        return have_illegal_muts['case_submitter_id'].to_list()

    def _compare_sites_new(self, same_age_tissue_methyl_df, mut_sample_methyl_df):
        """
        Get the delta_mf and ztest pvalue for each site in each sample, mutated and non-mutated
        @returns:
            - all_samples_diff: samples X sites dataframe with delta_mf for each site and sample
            - pvals: samples X sites dataframe with ztest pvalue for each site and sample
        """
        # create a new dataframe all_samples_at_comparison_sites with same_age_tissue_methyl_df as first rows and mut_sample_methyl_df as last row
        all_samples_at_comparison_sites = same_age_tissue_methyl_df.append(mut_sample_methyl_df)
        # matrix of z scores of each sample at each site being different from the other samples
        zscores = all_samples_at_comparison_sites.apply(lambda col: stats.zscore(col, nan_policy='omit'), axis=0)
        # convert to 2 sided pvalues
        ztest_pvals = stats.norm.sf(abs(zscores))*2
        ztest_pvals = pd.DataFrame(ztest_pvals, index=all_samples_at_comparison_sites.index, columns=all_samples_at_comparison_sites.columns)
        # median absolute deviation
        CONSISTENCY_CONST = 0.6745
        modified_zscores = all_samples_at_comparison_sites.apply(lambda col: CONSISTENCY_CONST * (col - col.median())/ stats.median_absolute_deviation(col, nan_policy='omit'), axis=0)
        # get difference of each sample from the mean and median
        mean = all_samples_at_comparison_sites.mean(axis=0)
        mean_diff = all_samples_at_comparison_sites.subtract(mean)
        median = all_samples_at_comparison_sites.median(axis=0)
        median_diff = all_samples_at_comparison_sites.subtract(median)
        # if any are not dataframes, convert them to dataframes
        if not isinstance(mean_diff, pd.DataFrame):
            mean_diff = pd.DataFrame(mean_diff)
        if not isinstance(median_diff, pd.DataFrame):
            median_diff = pd.DataFrame(median_diff)
        if not isinstance(ztest_pvals, pd.DataFrame):
            ztest_pvals = pd.DataFrame(ztest_pvals)
        if not isinstance(zscores, pd.DataFrame):
            zscores = pd.DataFrame(zscores)
        if not isinstance(modified_zscores, pd.DataFrame):
            modified_zscores = pd.DataFrame(modified_zscores)
        # stack each of these dataframes and then join them together
        mean_diff = mean_diff.stack().reset_index()
        mean_diff.columns = ['sample', 'measured_site', 'delta_mf']
        median_diff = median_diff.stack().reset_index()
        median_diff.columns = ['sample', 'measured_site', 'delta_mf_median']
        ztest_pvals = ztest_pvals.stack().reset_index()
        ztest_pvals.columns = ['sample', 'measured_site', 'ztest_pval']
        zscores = zscores.stack().reset_index()
        zscores.columns = ['sample', 'measured_site', 'zscore']
        modified_zscores = modified_zscores.stack().reset_index()
        modified_zscores.columns = ['sample', 'measured_site', 'modified_zscore']
        # merge all together on sample and measured_site
        all_metrics = mean_diff.merge(median_diff, on=['sample', 'measured_site'])
        all_metrics = all_metrics.merge(ztest_pvals, on=['sample', 'measured_site'])
        all_metrics = all_metrics.merge(zscores, on=['sample', 'measured_site'])
        all_metrics = all_metrics.merge(modified_zscores, on=['sample', 'measured_site'])
        return all_metrics
        

    def effect_on_each_site(self, mut_nearby_measured_df):
        """
        For each mutation, get the effect of the mutation on each measured CpG within max_dist of the mutation
        @ mut_nearby_measured_df
        @ returns: df with columns=[measured_site, mut_cpg, delta_mf, ztest_pval] and rows=[each mut_cpg, measured_site pair]
        """        
        num_skipped = 0
        num_dropped = 0
        all_metrics_dfs = []
        for _, mut_row in track(mut_nearby_measured_df.iterrows(), description="Analyzing each mutation", total=len(mut_nearby_measured_df)):
        #for _, mut_row in mut_nearby_measured_df.iterrows():
            # get the same age and dataset samples
            same_age_dset_samples = self._same_age_and_tissue_samples(mut_row['case_submitter_id'])
            # exclude samples that have ANY mutations within max_dist of a close_measured site
            samples_to_exclude = self._detect_effect_in_other_samples(mut_row['close_measured'], mut_row)
            # drop entries of same_age_dset_samples that are in samples_to_exclude
            before_drop = len(same_age_dset_samples)
            same_age_dset_samples = [s for s in same_age_dset_samples if s not in samples_to_exclude]
            after_drop = len(same_age_dset_samples)
            num_dropped += before_drop - after_drop
            if len(same_age_dset_samples) <= 10:
                num_skipped += 1
                continue       
            nearby_sites = mut_row['close_measured']
            mut_samples_nearby_mfs = self.all_methyl_age_df_t.loc[mut_row['case_submitter_id'], nearby_sites]
            same_age_nearby_mfs = self.all_methyl_age_df_t.loc[same_age_dset_samples, nearby_sites]
            # measure the change in methylation between sites in the mutated samples and in other non-mutated samples of the same age
            metrics = self._compare_sites_new(same_age_nearby_mfs, mut_samples_nearby_mfs)
            metrics['mut_cpg'] = mut_row['mut_cpg']
            # create column called 'mutated' that is True if the sample is the mutated sample
            metrics['mutated'] = metrics['sample'] == mut_row['case_submitter_id']
            cpg_to_dist_dict = dict(zip(mut_row['close_measured'], mut_row['close_measured_dists']))
            metrics['measured_site_dist'] = metrics['measured_site'].map(cpg_to_dist_dict)
            # add to output
            all_metrics_dfs.append(metrics)
    
        all_metrics_df = pd.concat(all_metrics_dfs)
        print("WARNING: Not enough samples of the same age and tissue to calculate effect of mutation for {} mutations".format(num_skipped), flush=True)
        print("WARNING: Dropped {} samples due to colliding mutation".format(num_dropped), flush=True)
        return all_metrics_df

    def find_nearby_measured_cpgs(self):
        """
        Find the measured cpgs within max_dist of each mutation
        @ returns: df of mutations that have at least one measured CpG within max_dist of the mutation. 'close_measured' column is a list of the measured cpgs within max_dist of the mutation.
        """
        mut_nearby_measured_l = []
        for chrom in track(self.all_mut_w_age_df['chr'].unique(), description = 'Finding nearby measured cpgs', total = len(self.all_mut_w_age_df['chr'].unique())):
            illum_locs = self.illumina_cpg_locs_df[self.illumina_cpg_locs_df['chr'] == chrom]
            mut_locs = self.all_mut_w_age_df.loc[self.all_mut_w_age_df['chr'] == chrom]
            # for each mutation, get a list of the measured CpGs #id in illum_locs that are within max_dist of the mutaiton 'start' but 0 distance (the same CpG)
            mut_locs.loc[:, 'close_measured'] = mut_locs.apply(lambda x: list(illum_locs[(np.abs(x['start'] - illum_locs['start']) <= self.max_dist) & (x['start'] -illum_locs['start'] != 0)]['#id']), axis = 1)
            # also get a list of the distances of these sites
            mut_locs.loc[:, 'close_measured_dists'] = mut_locs.apply(lambda x: list(illum_locs[(np.abs(x['start'] - illum_locs['start']) <= self.max_dist) & (x['start'] -illum_locs['start'] != 0)]['start'] - x['start']), axis = 1)
            # drop all rows of mut_locs where close_measured is empty
            mut_locs = mut_locs[mut_locs['close_measured'].apply(lambda x: len(x) > 0)]
            mut_nearby_measured_l.append(mut_locs)
        mut_nearby_measured_df = pd.concat(mut_nearby_measured_l)
        mut_nearby_measured_df.loc[:, 'mut_cpg'] = mut_nearby_measured_df['chr'] + ':' + mut_nearby_measured_df['start'].astype(str)
        return mut_nearby_measured_df 

    def look_for_disturbances(self, min_VAF_percentile):
        """
        Driver for the analysis. Finds mutations with VAF >= min_VAF_percentile that have a measured CpG within max_dist of the mutation and then looks for disturbances in the methylation of these CpGs.
        @ max_dist: maximum distance between mutation and measured CpG to be considered
        @ min_VAF: minimum VAF of mutation to be considered
        """
        # subset to only mutations that are C>T, non X and Y chromosomes, and that occured in samples with measured methylation
        self.all_mut_w_age_df = self.all_mut_w_age_df[(self.all_mut_w_age_df['mutation'] == 'C>T')
         & (self.all_mut_w_age_df['chr'] != 'X') & (self.all_mut_w_age_df['chr'] != 'Y')
         & (self.all_mut_w_age_df['case_submitter_id'].isin(self.all_methyl_age_df_t.index))
         & (self.all_mut_w_age_df['DNA_VAF'] >= np.percentile(self.all_mut_w_age_df['DNA_VAF'], min_VAF_percentile))]
        self.all_mut_w_age_df.loc[:, 'mut_cpg'] = self.all_mut_w_age_df['chr'] + ':' + self.all_mut_w_age_df['start'].astype(str)

        # subset illumina_cpg_locs_df to only the CpGs that are measured
        self.illumina_cpg_locs_df = self.illumina_cpg_locs_df.loc[self.illumina_cpg_locs_df['#id'].isin(self.all_methyl_age_df_t.columns)]

        # for each mutation, get a list of the CpGs #id in illum_locs that are within max_dist of the mutation 'start'
        mut_nearby_measured_df = self.find_nearby_measured_cpgs()

        # move the row with index 1708681 to the start of the dataframe
        #mut_nearby_measured_df = mut_nearby_measured_df.loc[[1708681, 1708282]].append(mut_nearby_measured_df.drop([1708681, 1708282]))

        # for each mutation with nearby measured site, compare the methylation of the nearby measured sites in the mutated sample to the other samples of same age and dataset
        all_metrics_df = self.effect_on_each_site(mut_nearby_measured_df)
        # fdr correct pvals
        all_metrics_df = utils.fdr_correct(all_metrics_df, pval_col_name = 'ztest_pval')
        all_metrics_df.reset_index(inplace=True, drop=True)
        self.all_metrics_df = all_metrics_df

        return mut_nearby_measured_df, all_metrics_df




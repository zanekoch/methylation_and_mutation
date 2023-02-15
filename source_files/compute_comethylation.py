import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import utils
import seaborn as sns
from statsmodels.stats.weightstats import ztest as ztest
from rich.progress import track
import os
import dask
import pyarrow
import sys
from tqdm import tqdm
from collections import defaultdict


class mutationScan:
    def __init__(
        self,
        all_mut_w_age_df: pd.DataFrame,
        illumina_cpg_locs_df: pd.DataFrame, 
        all_methyl_age_df_t: pd.DataFrame,
        corr_dir: str,
        age_bin_size: int, 
        max_dist: int,
        num_correl_sites: float,
        num_background_events: int,
        matched_sample_num: int
        ) -> None:

        self.all_mut_w_age_df = all_mut_w_age_df
        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.corr_dir = corr_dir
        self.age_bin_size = age_bin_size
        self.max_dist = max_dist
        self.num_correl_sites = num_correl_sites
        self.num_background_events = num_background_events
        self.matched_sample_num = matched_sample_num
        # Preprocessing: subset to only mutations that are C>T, non X and Y chromosomes, and that occured in samples with measured methylation
        self.all_mut_w_age_df['mut_cpg'] = self.all_mut_w_age_df['chr'] + ':' + self.all_mut_w_age_df['start'].astype(str)
        self.all_mut_w_age_df = self.all_mut_w_age_df.loc[
            (self.all_mut_w_age_df['mutation'] == 'C>T')
            & (self.all_mut_w_age_df['chr'] != 'X') 
            & (self.all_mut_w_age_df['chr'] != 'Y')
            & (self.all_mut_w_age_df['case_submitter_id'].isin(self.all_methyl_age_df_t.index)),
            :]# (self.all_mut_w_age_df['mutation'] == 'C>T')
        # join self.all_mut_w_age_df with the illumina_cpg_locs_df
        all_mut_w_age_illum_df = self.all_mut_w_age_df.copy(deep=True)
        all_mut_w_age_illum_df['start'] = pd.to_numeric(self.all_mut_w_age_df['start'])
        self.all_mut_w_age_illum_df = all_mut_w_age_illum_df.merge(
                                        self.illumina_cpg_locs_df, on=['chr', 'start'], how='left')
        # subset illumina_cpg_locs_df to only the CpGs that are measured
        self.illumina_cpg_locs_df = self.illumina_cpg_locs_df.loc[self.illumina_cpg_locs_df['#id'].isin(self.all_methyl_age_df_t.columns)]
        # and remove chr X and Y
        self.illumina_cpg_locs_df = self.illumina_cpg_locs_df.loc[(self.illumina_cpg_locs_df['chr'] != 'X') & (self.illumina_cpg_locs_df['chr'] != 'Y')]

    def correct_pvals(
        self, 
        all_metrics_df: pd.DataFrame,
        one_background: bool = False
        ):
        """
        Correct each mutation events pvalue based on the background pvalues
        @ all_metrics_df: a dataframe with all the metrics for each mutation event
        @ returns: The all_metrics_df with the sig columns added
        """
        def get_sigs(mut_row, cutoffs):
            this_mut_cutoffs = cutoffs[mut_row['mut_event']]
            return mut_row[this_mut_cutoffs[0].index] < this_mut_cutoffs[0]

        def one_cutoff(background_pvals, num_mut_events):
            # for each column, get the 5th percentile value and bonf. correct it
            background_pvals_cutoffs = background_pvals[
                ['mf_pval2', 'mf_pval', 'delta_mf_pval2', 'delta_mf_pval','abs_delta_mf_pval']
                ].quantile(0.05) / num_mut_events
            return background_pvals_cutoffs

        def get_cutoffs_one_background(metrics_df, mut_events):
            num_mut_events = len(mut_events)
            background_pvals = metrics_df.loc[
                (metrics_df['is_background'] == True) # get background samples
                & (metrics_df['mutated_sample'] == True) # and get just one of them
                , :].drop_duplicates(subset=['mut_event']) # get just one row per background mut event
            all_background_pvals_cutoff = one_cutoff(background_pvals, num_mut_events)
            return all_background_pvals_cutoff
            
        def get_cutoffs(metrics_df, mut_events):
            num_mut_events = len(mut_events)
            cutoffs = defaultdict(list)
            background_pvals = metrics_df.loc[
                (metrics_df['is_background'] == True) # get background samples
                & (metrics_df['mutated_sample'] == True) # and get just one of them
                , :].drop_duplicates(subset=['mut_event']) # get just one row per background mut event
            # for each mutation event
            for mut_event in mut_events:
                # get this mut event's background sites
                this_background_pvals = background_pvals.loc[
                    (background_pvals['index_event'] == mut_event),
                    :]
                # get the 5th percentile value for each pvalue of this mut event's background sites
                background_pvals_cutoffs = one_cutoff(this_background_pvals, num_mut_events)
                cutoffs[mut_event].append(background_pvals_cutoffs)
            return cutoffs

        # get real mutation events
        real_muts = all_metrics_df.loc[
            (all_metrics_df['index_event'] == 'self') # redundant, makes sure not a background site
            & (all_metrics_df['mutated_sample'] == True) # makes sure is the mutated sample, for uniqueness
            & (all_metrics_df['is_background'] == False) # makes sure not a background site
            , : ]
        # and get the unique mutation events from these
        mut_events = real_muts['mut_event'].unique()
        if one_background:
            cutoffs = get_cutoffs_one_background(all_metrics_df, mut_events)
            all_metrics_df[['mf_pval2_sig', 'mf_pval_sig', 'delta_mf_pval2_sig', 'delta_mf_pval_sig', 'abs_delta_mf_pval_sig']] = all_metrics_df[['mf_pval2', 'mf_pval', 'delta_mf_pval2', 'delta_mf_pval','abs_delta_mf_pval']] < cutoffs
            return all_metrics_df
        else:
            cutoffs = get_cutoffs(all_metrics_df, mut_events)
            all_metrics_df[['mf_pval2_sig', 'mf_pval_sig', 'delta_mf_pval2_sig', 'delta_mf_pval_sig', 'abs_delta_mf_pval_sig']] = all_metrics_df.apply(
                lambda mut_row: get_sigs(mut_row, cutoffs), axis=1
                )
            return all_metrics_df

    def volcano_plot(
        self,
        all_metrics_df: pd.DataFrame,
        pval_col: str,
        ) -> None:
        """
        Plot a volcano plot of the the cumul_delta_mf and pval
        """
        # TODO: deal with pvalues == 0
        # first select only the rows with measurements for the mutated sample
        real_muts_df = all_metrics_df.loc[
            (all_metrics_df['index_event'] == 'self') # redundant, makes sure not a background site
            & (all_metrics_df['mutated_sample'] == True) # makes sure is the mutated sample, for uniqueness
            & (all_metrics_df['is_background'] == False) # makes sure not a background site
            , : ]
        real_muts_df['abs_delta_mf_median'] = real_muts_df['delta_mf_median'].abs()

        # then get median delta_mf for each event
        med_delta_mf = real_muts_df.groupby('mut_event')['delta_mf_median'].median()
        abs_median_delta_mf = real_muts_df.groupby('mut_event')['abs_delta_mf_median'].median()
        # doesn't matter if min, max, or mean cause all pvals are same for a mut event
        pvals = real_muts_df.groupby('mut_event')[pval_col].min()
        sigs = real_muts_df.groupby('mut_event')[pval_col + '_sig'].min()
        grouped_volc_metrics = pd.merge(abs_median_delta_mf, pd.merge(sigs, pd.merge(med_delta_mf, pvals, on='mut_event'), on='mut_event'), on='mut_event')
        grouped_volc_metrics['log10_pval'] = grouped_volc_metrics[pval_col].apply(lambda x: -np.log10(x))
        _, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100, gridspec_kw={'width_ratios': [3, 1]})
        # volcano plot
        sns.scatterplot(
            y = 'log10_pval', x = 'delta_mf_median', data = grouped_volc_metrics, alpha=0.3,
            hue = pval_col + '_sig', palette = {True: 'maroon', False: 'grey'}, ax = axes[0]
            )
        # round the pvalue
        axes[0].legend(loc='upper center', labels=[f'p < sig. threshold', f'p >= sig. threshold'], title='p-value')
        axes[0].set_xlabel(r"Median $\Delta$MF")
        axes[0].set_ylabel('-log10 p-value')
        # barplot of the number of significant and non-significant mutations
        sns.countplot(x = pval_col + '_sig', data = grouped_volc_metrics, palette = {True: 'maroon', False: 'grey'}, ax = axes[1])
        # set x ticks
        axes[1].set_xlabel('Corrected p-value')
        axes[1].set_xticklabels([f'Not significant', f'Significant' ])
        axes[1].set_ylabel('Count of mutation events')

        # same but absolute cumul
        _, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100, gridspec_kw={'width_ratios': [3, 1]})
        # volcano plot
        sns.scatterplot(
            y = 'log10_pval', x = 'abs_delta_mf_median', data = grouped_volc_metrics, alpha=0.3,
            hue = pval_col + '_sig', palette = {True: 'maroon', False: 'grey'}, ax = axes[0]
            )
        axes[0].legend(loc='upper left', labels=[f'p < sig. threshold', f'p >= sig. threshold'], title='Bonferroni p-value')
        axes[0].set_xlabel(r"Median absolute $\Delta$MF")
        axes[0].set_ylabel('-log10 p-value')
        # barplot of the number of significant and non-significant mutations
        sns.countplot(x = pval_col + '_sig', data = grouped_volc_metrics, palette = {True: 'maroon', False: 'grey'}, ax = axes[1])
        axes[1].set_xlabel('Corrected p-value')
        axes[1].set_xticklabels([f'Not significant',f'Significant' ])
        axes[1].set_ylabel('Count of mutation events')

    def extent_of_effect(
        self,
        all_metrics_df: pd.DataFrame,
        ) -> None:
        all_metrics_df['abs_delta_mf_median'] = all_metrics_df['delta_mf_median'].abs()
        # get the cumulative effect in the significant mutated samples first
        sig_mut_events = all_metrics_df.loc[(all_metrics_df['mf_pval_sig'] == True) & (all_metrics_df['mutated_sample'] == True), :]
        abs_cumul_delta_mf_sig_mut = sig_mut_events.groupby('mut_event')['abs_delta_mf_median'].sum().reset_index()
        abs_cumul_delta_mf_sig_mut['Site of mutation event'] = 'Mutated individuals'
        # then do the same for the non-mutated significant samples
        sig_nonmut_events = all_metrics_df[(all_metrics_df['mf_pval_sig'] == True) & (all_metrics_df['mutated_sample'] == False)]
        abs_cumul_delta_mf_sig_nonmut = sig_nonmut_events.groupby(['mut_event', 'sample'])['abs_delta_mf_median'].sum().reset_index()
        abs_cumul_delta_mf_sig_nonmut['Site of mutation event'] = 'Non-mutated matched individuals'
        abs_cumul_delta_mf_sig_nonmut.drop(columns=['sample'], inplace=True)
        # concat the two dataframes
        abs_cumul_delta_mfs = pd.concat([abs_cumul_delta_mf_sig_mut, abs_cumul_delta_mf_sig_nonmut], axis=0)
        fig, axes = plt.subplots(1, 2, figsize=(12,4), dpi=100)
        abs_cumul_delta_mfs.columns= ['mut_event', 'abs_delta_mf_median', 'Site of mutation event']
        sns.violinplot(
            data = abs_cumul_delta_mfs, x= 'Site of mutation event', y = 'abs_delta_mf_median', cut=0, bw=.15,
            ax = axes[0], palette={'Mutated individuals': 'maroon', 'Non-mutated matched individuals': 'steelblue'}
            )
        # set y label
        axes[0].set_ylabel(r"Absolute cumulative $\Delta$MF")
        cumul_delta_mf_sig_mut = sig_mut_events.groupby('mut_event')['delta_mf_median'].sum().reset_index()
        cumul_delta_mf_sig_mut['Site of mutation event'] = 'Mutated individuals'
        hyper_count = cumul_delta_mf_sig_mut[cumul_delta_mf_sig_mut['delta_mf_median'] > 0].shape[0]
        hypo_count = cumul_delta_mf_sig_mut[cumul_delta_mf_sig_mut['delta_mf_median'] < 0].shape[0]
        colors = plt.get_cmap('Reds')(np.linspace(0.2, 0.7, 2))
        axes[1].pie(
            x = [hyper_count, hypo_count], labels=['Hyper-methylation:\nCumulative $\Delta$MF > 0', 'Hypo-methylation:\nCumulative $\Delta$MF < 0'], autopct='%1.1f%%',
            colors=['indianred', 'maroon'], wedgeprops={"alpha": 0.5},
            labeldistance=1.2)
        # move title to bottom
        axes[1].set_title('Direction of comparison site change', y=1.1)

    def corr_heatmap(
        self, 
        mut_event, 
        comparison_sites_df,
        distlim: int
        ) -> None:
        """
        Plot a heatmap of the mutated sample and matched sample MFs at the comparison sites
        """
        # get mutatated sample name and comparison sites
        this_mut_event = comparison_sites_df.loc[comparison_sites_df['mut_event'] == mut_event]
        if len(this_mut_event) == 0:
            print("mut event not present")
            sys.exit(1)
        mut_sample_name = this_mut_event['case_submitter_id'].values[0]
        try:
            comparison_sites = [i.decode('ASCII') for i in this_mut_event['comparison_sites'].values[0]]
        except:
            comparison_sites = this_mut_event['comparison_sites'].values[0]
        mut_cpg_name = this_mut_event['#id'].values[0]
        # get the distances between the mutated site and the comparison sites
        comparison_sites_starts = self.illumina_cpg_locs_df.loc[self.illumina_cpg_locs_df['#id'].isin(comparison_sites), ['#id', 'start']]
        comparison_sites_starts.set_index('#id', inplace=True)
        comparison_sites_starts = comparison_sites_starts.reindex(comparison_sites)
        mut_cpg_start = self.illumina_cpg_locs_df.loc[self.illumina_cpg_locs_df['#id'] == mut_cpg_name, 'start'].values[0]
        comparison_site_distances = comparison_sites_starts['start'] - mut_cpg_start
        # sort the comparison sites by distance
        comparison_site_distances.sort_values(ascending = True, inplace=True)
        comparison_site_distances = comparison_site_distances[(comparison_site_distances > -1*distlim) & (comparison_site_distances < distlim)]
        comparison_sites = comparison_site_distances.index.tolist()

        # get the matched sampes
        matched_samples = self._same_age_and_tissue_samples(mut_sample_name)
        # and drop those with mutations nearby
        samples_to_exclude = self._detect_effect_in_other_samples(
            sites_to_test = comparison_sites, 
            mut_row = this_mut_event
            )
        # drop entries of matched_samples that are in samples_to_exclude
        before_drop = len(matched_samples)
        matched_samples = [s for s in matched_samples if s not in samples_to_exclude]
        print(f"Dropped {before_drop - len(matched_samples)} samples with mutations nearby")
        # get methylation fractions of the all samples at comparison sites, mut sample first
        samples_to_plot = np.concatenate((utils.half(matched_samples, 'first'), [mut_sample_name], utils.half(matched_samples, 'second')))

        # find the index of comparison_sites_distances which is closest to 0 
        magnify_mut_factor =30
        distances = comparison_site_distances.values.tolist()
        for i in range(len(distances)):
            if distances[i] > 0:
                comparison_sites = comparison_sites[:i] + [mut_cpg_name]*magnify_mut_factor + comparison_sites[i:]
                distances = distances[:i] + [0]*magnify_mut_factor + distances[i:]
                mut_pos = i + magnify_mut_factor/2
                break
            if max(distances) < 0:
                comparison_sites.append(mut_cpg_name)
                comparison_sites = comparison_sites + [mut_cpg_name]*magnify_mut_factor
                distances = distances + [0]*magnify_mut_factor
                mut_pos = len(distances) - 1 + magnify_mut_factor/2
        
        all_samples_comp_sites = self.all_methyl_age_df_t.loc[samples_to_plot, comparison_sites]
        # plot as a heatmap
        _, axes = plt.subplots(figsize=(9,6), dpi=100)
        ax = sns.heatmap(
            data = all_samples_comp_sites, annot=False, xticklabels=False, yticklabels=False, 
            cmap="Blues", vmin=0, vmax=1, center=0.5,
            cbar_kws={'label': r'$Delta$MF'}, ax=axes
            )#, cmap="icefire", vmin=-1, vmax=1, center=0
        # label axes
        ax.set_xlabel('Comparison sites')
        ax.set_ylabel('Samples')
        # add a y tick for the mutated sample, make tick label red, and rotate 90 degrees
        ax.set_yticks(np.arange(.5, len(samples_to_plot)+.5, 1))
        ax.set_yticks([int(len(utils.half(matched_samples, 'first')))+.5])
        ax.set_yticklabels(['Mutated sample'], color='red', rotation=90, ha='center', rotation_mode='anchor')
        ax.tick_params(axis='y', which='major', pad=5)
        # slightly seperate mutated CpG
        ax.add_patch(Rectangle((mut_pos-(magnify_mut_factor/2), 0), magnify_mut_factor, len(all_samples_comp_sites), fill=False, edgecolor='white', lw=4))
        # add second around the others to make same height
        """ax.add_patch(Rectangle((100, 0), len(all_samples_comp_sites.columns), len(all_samples_comp_sites), fill=False, edgecolor='white', lw=2))"""
        # add a tick label for the mutated CpG
        tick_locs = [0, mut_pos, len(distances)-1]
        ax.set_xticks(tick_locs)

        ax.set_xticklabels([str(int(-1*distlim/1000000))+'Mbp', 'Mutated site', str(int(distlim/1000000))+'Mbp'], ha='center', rotation_mode='anchor')
        colors = ['black', 'red', 'black']
        for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)
        return all_samples_comp_sites

    def _join_with_illum(self, in_df, different_illum = None):
        """
        Join the dataframe with the illumina_cpg_locs_df
        """
        df = in_df.copy(deep=True)
        # split 'mut_cpg' into 'chr' and 'start'
        df[['chr', 'start']] = df['mut_cpg'].str.split(':', expand=True)
        # convert start column to int with to_numeric
        df['start'] = pd.to_numeric(df['start'])
        if different_illum is None:
            df_w_illum = df.merge(self.illumina_cpg_locs_df, on=['chr', 'start'], how='left')
        else:
            df_w_illum = df.merge(different_illum, on=['chr', 'start'], how='left')
        return df_w_illum

    def preproc_correls(
        self, 
        out_dir
        ) -> None:
        """
        Calculate the correlation matrix for each dataset within each chromosome and output to file
        """
        for chrom in self.all_mut_w_age_df['chr'].unique():
            this_chr_measured_cpgs = self.illumina_cpg_locs_df[self.illumina_cpg_locs_df['chr'] == chrom]
            for dset in self.all_mut_w_age_df['dataset'].unique():
                this_chr_dset_methyl_df = self.all_methyl_age_df_t.loc[:, this_chr_measured_cpgs['#id'].to_list() + ['dataset']]
                this_chr_dset_methyl_df = this_chr_dset_methyl_df[this_chr_dset_methyl_df['dataset'] == dset]
                corr_df = this_chr_dset_methyl_df.corr()
                corr_df.to_parquet(os.path.join(out_dir, 'chr{}_{}.parquet'.format(chrom, dset)))
                print(chrom, dset, flush=True)

    def _same_age_and_tissue_samples(
        self, 
        sample_name: str
        ) -> list:
        """
        Return a list of samples with matched age and tissue as the sample_name
        @ sample_name
        @ returns: list of sample names
        """
        # get this sample's age and dataset
        this_age = self.all_methyl_age_df_t.loc[sample_name, 'age_at_index']
        this_dset = self.all_methyl_age_df_t.loc[sample_name, 'dataset']
        # get the mf all other samples of within age_bin_size/2 years of age on either side
        matched_samples = self.all_methyl_age_df_t.loc[
            (np.abs(self.all_methyl_age_df_t['age_at_index'] - this_age) <= self.age_bin_size/2) 
            & (self.all_methyl_age_df_t['dataset'] == this_dset)
            ].index
        # drop the mutated sample itself
        matched_samples_no_mut = matched_samples.drop(sample_name)
        return matched_samples_no_mut.to_list()

    # TODO: maybe use merge to do this better
    def _detect_effect_in_other_samples(
        self, 
        sites_to_test: list, 
        mut_row: pd.Series
        ) -> list:
        """
        Detect samples that may have been affected by a mutation in the same region as the mutation we are testing, to discard these samples as they are a bad comparison
        @ sites_to_test: list of sites to test, cg######### format
        @ mut_row: the row from comparison_sites_df specifying the mutation
        @ returns: list of samples that do not have any C>T mutations in max_dist from any of the sites in sites_to_test
        """
        # so it works when called from heatmap and comethylation scan 
        mut_row = mut_row.copy(deep=True).squeeze()
        # select rows of self.all_mut_w_age_df that have the same chr and dataset as mut_row
        relevant_mutations = self.all_mut_w_age_df.loc[(self.all_mut_w_age_df['chr'] == mut_row['chr']) & (self.all_mut_w_age_df['dataset'] == mut_row['dataset'])]
        # detect samples that have a mutation in the max_dist window of any of the sites in sites_to_test
        # get locations of sites to test
        sites_to_test_locs = self.illumina_cpg_locs_df.loc[self.illumina_cpg_locs_df['#id'].isin(sites_to_test)] 
        # select rows of relevant_mutations that are within max_dist of any of the sites_to_test
        have_illegal_muts = relevant_mutations.loc[relevant_mutations.apply(lambda row: any(np.abs(row['start'] - sites_to_test_locs['start']) <= self.max_dist), axis=1)]
        # detect samples that have a mutation in the mut_cpg or within max_dist of it
        have_illegal_muts = pd.concat([have_illegal_muts, relevant_mutations.loc[(relevant_mutations['mut_cpg'] == mut_row['mut_cpg']) | (np.abs(relevant_mutations['start'] - mut_row['start']) <= self.max_dist)]])
        return have_illegal_muts['case_submitter_id'].to_list()

    def _compare_sites(
        self, 
        comparison_site_mfs: pd.DataFrame, 
        mut_sample_name: str
        ) -> pd.DataFrame:
        """
        Calculate effect size and pvalue for each comparison
        """
        # get the difference of each sample from the median of the other samples
        median_diffs = comparison_site_mfs.apply(
            lambda row: row - np.nanmedian(comparison_site_mfs.drop(row.name), axis = 0),
            axis=1
            )
        abs_median_diffs = np.abs(median_diffs)

        metrics = median_diffs.stack().reset_index()
        metrics.columns = ['sample', 'measured_site', 'delta_mf_median']
        # create column called 'mutated' that is True if the sample is the mutated sample
        metrics['mutated_sample'] = metrics['sample'] == mut_sample_name
        # test for a difference in methylation fraction
        metrics['mf_pval2'] = stats.mannwhitneyu(
            x = comparison_site_mfs.loc[mut_sample_name].to_numpy().ravel(),
            y = comparison_site_mfs.drop(mut_sample_name).to_numpy().ravel(),
            alternative = 'two-sided'
            ).pvalue
        metrics['mf_pval'] = stats.mannwhitneyu(
            x = comparison_site_mfs.loc[mut_sample_name].to_numpy().ravel(),
            y = comparison_site_mfs.drop(mut_sample_name).to_numpy().ravel(),
            alternative = 'greater'
            ).pvalue
        # test for a difference in delta_mf
        metrics['delta_mf_pval2'] = stats.mannwhitneyu(
            x = median_diffs.loc[mut_sample_name].to_numpy().ravel(),
            y = median_diffs.drop(mut_sample_name).to_numpy().ravel(),
            alternative = 'two-sided'
            ).pvalue
        metrics['delta_mf_pval'] = stats.mannwhitneyu(
            x = median_diffs.loc[mut_sample_name].to_numpy().ravel(),
            y = median_diffs.drop(mut_sample_name).to_numpy().ravel(),
            alternative = 'greater'
            ).pvalue
        # test for a difference in abs_delta_mf
        metrics['abs_delta_mf_pval'] = stats.mannwhitneyu(
            x = abs_median_diffs.loc[mut_sample_name].to_numpy().ravel(),
            y = abs_median_diffs.drop(mut_sample_name).to_numpy().ravel(),
            alternative = 'greater'
            ).pvalue
        return metrics

    def effect_on_each_site(
        self, 
        comparison_sites_df: pd.DataFrame
        ) -> pd.DataFrame:
        """
        For each mutation, get the effect of the mutation on each comparison site
        @ comparison_sites_df
        @ returns: df with statistical test and effect sizes
        """        
        def process_row(mut_row):
            # get the same age and dataset samples
            matched_samples = mut_row['matched_samples']
            # exclude samples that have ANY mutations within max_dist of a comparison site or the mutated site
            samples_to_exclude = self._detect_effect_in_other_samples(mut_row['comparison_sites'], mut_row)
            # drop entries of matched_samples that are in samples_to_exclude 
            matched_samples = [s for s in matched_samples if s not in samples_to_exclude]
            # limit to matched_sample_num samples, no more, no less, 
            # to avoid statistical bias towards samples with more similar samples
            if len(matched_samples) < self.matched_sample_num:
                return None
            else:
                matched_samples = matched_samples[:self.matched_sample_num]
            # get a list of matched and mutated samples to select from methylation 
            all_samples = matched_samples
            all_samples.append(mut_row['case_submitter_id'])
            comparison_site_mfs = self.all_methyl_age_df_t.loc[all_samples, mut_row['comparison_sites']]
            # measure the change in methylation between sites in the mutated samples and in other non-mutated samples of the same age
            metrics = self._compare_sites(comparison_site_mfs, mut_sample_name = mut_row['case_submitter_id'])
            metrics['mut_cpg'], metrics['mut_event'], metrics['is_background'], metrics['index_event'] = mut_row['mut_cpg'], mut_row['mut_event'], mut_row['is_background'], mut_row['index_event']
            cpg_to_dist_dict = dict(zip(mut_row['comparison_sites'], mut_row['comparison_dists']))
            metrics['measured_site_dist'] = metrics['measured_site'].map(cpg_to_dist_dict)
            # add to output
            return metrics
        tqdm.pandas(desc="Calculating effect of mutation on comparison sites", miniters=len(comparison_sites_df)/100)
        # apply process_row across each row of comparison_sites_df
        all_metrics_dfs = comparison_sites_df.progress_apply(process_row, axis=1)
        # drop none values
        all_metrics_dfs = all_metrics_dfs.dropna()
        all_metrics_dfs = all_metrics_dfs.to_list()
        # concat all dfs
        all_metrics_df = pd.concat(all_metrics_dfs)
        print("Done getting effect of mutation on each site", flush=True)
        return all_metrics_df

    def _find_nearby_measured_cpgs(
        self, 
        min_VAF_percentile: float
        ) -> pd.DataFrame:
        """
        Find the measured cpgs within max_dist of each mutation
        @ min_VAF_percentile: the minimum VAF percentile that a mutation must have to be considered
        @ returns: df of mutations that have at least one measured CpG within max_dist of the mutation. 'comparison_sites' column is a list of the measured cpgs within max_dist of the mutation.
        """
        mut_nearby_measured_l = []
        for chrom in track(self.all_mut_w_age_df['chr'].unique(), description = 'Finding nearby measured cpgs', total = len(self.all_mut_w_age_df['chr'].unique())):
            illum_locs = self.illumina_cpg_locs_df[self.illumina_cpg_locs_df['chr'] == chrom]
            # look for only mutations on this chromsome that meet the min_VAF_percentile
            mut_locs = self.all_mut_w_age_df.loc[(self.all_mut_w_age_df['chr'] == chrom)
                                                & (self.all_mut_w_age_df['DNA_VAF'] 
                                                >= np.percentile(self.all_mut_w_age_df['DNA_VAF'], min_VAF_percentile))
                                                ]
            # for each mutation, get a list of the measured CpGs #id in illum_locs that are within max_dist of the mutation 'start' but 0 distance (the same CpG)
            mut_locs.loc[:, 'comparison_sites'] = mut_locs.apply(lambda x: list(illum_locs[(np.abs(x['start'] - illum_locs['start']) <= self.max_dist) & (x['start'] -illum_locs['start'] != 0)]['#id']), axis = 1)
            # also get a list of the distances of these sites
            mut_locs.loc[:, 'comparison_dists'] = mut_locs.apply(lambda x: list(illum_locs[(np.abs(x['start'] - illum_locs['start']) <= self.max_dist) & (x['start'] -illum_locs['start'] != 0)]['start'] - x['start']), axis = 1)
            # drop all rows of mut_locs where comparison_sites is empty
            mut_locs = mut_locs[mut_locs['comparison_sites'].apply(lambda x: len(x) > 0)]
            mut_nearby_measured_l.append(mut_locs)
        comparison_sites_df = pd.concat(mut_nearby_measured_l)
        comparison_sites_df.loc[:, 'mut_cpg'] = comparison_sites_df['chr'] + ':' + comparison_sites_df['start'].astype(str)
        return comparison_sites_df 

    def _find_collisions(
        self,
        muts_df: pd.DataFrame
        ) -> pd.DataFrame:
        # Merge the two DataFrames on the 'chr' column, using an inner join
        merged_df = pd.merge(muts_df, self.illumina_cpg_locs_df, on=['chr'], how='inner')
        # Create a new DataFrame containing only the rows where the 'start' column values
        # are not within 1000 of each other
        no_close_mut = merged_df.loc[np.abs(merged_df['start_x'] - merged_df['start_y']) > self.max_dist]
        return no_close_mut

    def _get_random_sites(
        self,
        mut_row: pd.Series,
        corr_direction: str
        ) -> pd.DataFrame:
        """
        Choose a random set of background sites and comparison sites for these background sites
        """
        
        """# get this sample's mutation events
        this_sample_muts = self.all_mut_w_age_df.loc[self.all_mut_w_age_df['case_submitter_id'] == mut_row['case_submitter_id'], :]
        # get the sites that are not close to a mutation in this sample
        no_close_mut = self._find_collisions(this_sample_muts)
        # randomly select num_background_events #ids from illumina_cpg_locs_df, on different chrom that mutation
        no_close_mut = no_close_mut.loc[no_close_mut['chr'] != mut_row['chr']]"""
        # randomly select num_background_events #ids from illumina_cpg_locs_df, on different chroms than the mutation
        rand_cpgs = self.illumina_cpg_locs_df[self.illumina_cpg_locs_df['chr'] != mut_row['chr']].sample(n=self.num_background_events, random_state=1)
        # assign these the same attributes same as the mutation event
        rand_cpgs['dataset'] = mut_row['dataset']
        rand_cpgs['case_submitter_id'] = mut_row['case_submitter_id']
        rand_cpgs['index_event'] = mut_row['mut_event']
        rand_cpgs['matched_samples'] = [mut_row['matched_samples'] for row in range(len(rand_cpgs))]
        
        """# get comparison sites for each event by randomly sampling from the sites that are on same chrom as the background event
        rand_cpgs['comparison_sites'] = rand_cpgs.apply(
            lambda background_mut_event: self.illumina_cpg_locs_df.loc[
                (self.illumina_cpg_locs_df['chr'] == background_mut_event['chr']), '#id'
                ].sample(n=self.num_correl_sites, random_state=1).to_list(), axis = 1)
        # get comparison sites for each event by choosing the most correlated sites in matched_samples
        rand_cpgs['comparison_sites'] = rand_cpgs.apply(
            lambda mut_event: self._select_correl_sites(mut_event, corr_direction), axis = 1
            )"""
            
        rand_cpgs['comparison_sites'] = rand_cpgs.apply(
            lambda mut_event: self._select_correl_sites_preproc(mut_event, corr_direction), axis = 1
            )
        return rand_cpgs

    def _choose_background_events(
        self,
        comparison_sites_df: pd.DataFrame,
        corr_direction: str
        ) -> pd.DataFrame:
        """
        For each mutation event in comparison_sites_df, choose self.num_background_events locations and associated correlated sites,
        making sure that the chosen sites are not within self.max_dist of a mutation and are on a different chromosome from the
        mutation event
        """
        chosen_background_events_l = []
        # for each mutation event in comparison_sites_df, choose self.num_background_events locations and associated comparison sites
        for i, mut_row in tqdm(comparison_sites_df.iterrows(), desc = "getting background sites", total = len(comparison_sites_df)):
                # returns a DataFrame with the chosen background event and their comparison sites
                rand_mut_events_and_comp_sites = self._get_random_sites(mut_row, corr_direction)
                chosen_background_events_l.append(rand_mut_events_and_comp_sites)
        # combine these randomly chosen background events with the actual mutation events, populating columns
        chosen_background_events = pd.concat(chosen_background_events_l)
        chosen_background_events['comparison_dists'] = [[i for i in range(self.num_correl_sites)] for _ in range(len(chosen_background_events))]
        chosen_background_events['mut_cpg'] = chosen_background_events['chr'] + ':' + chosen_background_events['start'].astype(str)
        chosen_background_events['mut_event'] = chosen_background_events['case_submitter_id'] + '_' + chosen_background_events['mut_cpg']
        chosen_background_events['is_background'] = True
        chosen_background_events['mut_delta_mf'] = np.nan
        # concat with comparison_sites_df
        comparison_sites_df = pd.concat([comparison_sites_df, chosen_background_events])
        comparison_sites_df.reset_index(inplace=True, drop=True)
        return comparison_sites_df

    def _select_correl_sites_preproc(
        self,
        mut_event: pd.Series,
        corr_direction: str
        ) -> list:
        """
        Gets the num sites that are either the most positively or negativelty correlated sites with in_cpg
        @ corr_df: correlation matrix for all sites from one dataset on one chromosome
        @ mut_cpg: the cpg that we are interested in correlating with
        @ corr_direction: 'pos' or 'neg' for positive or negative correlation
        @ returns: list of chosen sites
        """
        # read in the the correlation matrix for this chrom and dataset
        corrs = pd.read_parquet(
            os.path.join(self.corr_dir, 'chr{}_{}.parquet'.format(mut_event['chr'], mut_event['dataset'])),
            columns = [mut_event['#id']]
            )
        # convert corrs to series
        corrs = corrs.iloc[:, 0]
        # get the value of the 'percentile' highest correlation
        if corr_direction == 'pos':
            q = corrs.quantile(1, interpolation='lower')
            # select num sites closest to q, but not including mut_cpg (which will have a correlation of 1)
            return corrs.iloc[(corrs - q).abs().argsort().iloc[1: self.num_correl_sites + 1]].index.to_list()
        elif corr_direction == 'neg':
            q = corrs.quantile(0, interpolation='higher')
            # here do not need to exclude most correlated site, since it will not be the mut_cpg
            return corrs.iloc[(corrs - q).abs().argsort().iloc[:self.num_correl_sites]].index.to_list()

    def _mut_site_delta_mf(
        self,
        mut_event: pd.Series,
        ) -> float:
        """
        @ mut_event: a row from comparison_sites_df
        @ returns: the delta MF of the mutated site
        """
        # get the MF of the mutation in matched samples
        mut_sample_mf = self.all_methyl_age_df_t.loc[mut_event['case_submitter_id'], mut_event['#id']]
        # get the mutated CpG's MF in the matched samples
        matched_mfs = self.all_methyl_age_df_t.loc[mut_event['matched_samples'], mut_event['#id']]
        # delta_mf
        return mut_sample_mf - matched_mfs.median()

    def _select_correl_sites(
        self,
        mut_event: pd.Series,
        corr_direction: str
        ) -> list:
        """
        Just in time correlation to find the most correlated sites to the mutation event CpG in matched samples
        """
        # get the mutated CpG's MF in the matched samples
        mut_cpg_mf = self.all_methyl_age_df_t.loc[mut_event['matched_samples'], mut_event['#id']]
        # get the MF of all same chrom CpGs in matched samples
        same_chrom_cpgs = self.illumina_cpg_locs_df.loc[
            (self.illumina_cpg_locs_df['chr'] == mut_event['chr']) 
            & (self.illumina_cpg_locs_df['#id'] != mut_event['#id']), # exclude the mut_cpg
            '#id'].values
        same_chrom_cpgs_mf = self.all_methyl_age_df_t.loc[mut_event['matched_samples'], same_chrom_cpgs]
        # get correlation between mut_cpg and all same chrom CpGs
        corrs = same_chrom_cpgs_mf.corrwith(mut_cpg_mf, axis=0)
        if corr_direction == 'pos':
            q = corrs.quantile(1, interpolation='lower')
            return corrs.iloc[(corrs - q).abs().argsort().iloc[:self.num_correl_sites]].index.to_list()
        elif corr_direction == 'neg':
            q = corrs.quantile(0, interpolation='higher')
            # here do not need to exclude most correlated site, since it will not be the mut_cpg
            return corrs.iloc[(corrs - q).abs().argsort().iloc[:self.num_correl_sites]].index.to_list()

    def _find_correl_measured_cpgs(
        self, 
        start_num_mut_to_process: int,
        end_num_mut_to_process: int,
        corr_direction: str
        ) -> pd.DataFrame:
        """
        Find the mutation events that meet the criteria (min_VAF, max_delta_mf) and get choose
        comparison sites to be those CpGs which are most correlated with the mutation event CpG 
        in matched samples
        @ min_VAF_percentile: the minimum VAF percentile for a mutation event to be considered
        @ max_delta_mf_percentile: the maximum delta MF percentile for a mutation event to be considered
        @ corr_direction: 'pos' or 'neg' for positive or negative correlation
        """
        pd.options.mode.chained_assignment = None  # default='warn'
        # subset to mutations in the measured CpGs (i.e. in the illumina CpGs)
        valid_muts_w_illum = self.all_mut_w_age_illum_df.loc[
                self.all_mut_w_age_illum_df['#id'].isin(self.illumina_cpg_locs_df['#id'])
                ]
        print("Number mutation events being processed: {}".format(len(valid_muts_w_illum)), flush=True)
        # get same age and tissue samples for each, keeping only mutations with at least self.matched_sample_num
        tqdm.pandas(desc="Getting matched samples", miniters=len(valid_muts_w_illum)/100)
        valid_muts_w_illum['matched_samples'] = valid_muts_w_illum.progress_apply(
            lambda mut_event: self._same_age_and_tissue_samples(mut_event['case_submitter_id']), axis = 1
            )
        valid_muts_w_illum = valid_muts_w_illum.loc[
            valid_muts_w_illum['matched_samples'].apply(len) >= self.matched_sample_num
            ]
        # get the delta MF of the mutated site
        tqdm.pandas(desc="Getting mut site delta MF", miniters=len(valid_muts_w_illum)/100)
        valid_muts_w_illum['mut_delta_mf'] = valid_muts_w_illum.progress_apply(
            lambda mut_event: self._mut_site_delta_mf(mut_event), axis=1
            )
        # sort mutations low to high by mut_delta_mf and and high to low by DNA_VAF
        valid_muts_w_illum = valid_muts_w_illum.sort_values(by=['mut_delta_mf', 'DNA_VAF'], ascending=[True, False])
        # select top mutations for further processing
        valid_muts_w_illum = valid_muts_w_illum.iloc[start_num_mut_to_process:end_num_mut_to_process, :]
        print("Number mutation events being processed after filtering for matched sample number: {}".format(len(valid_muts_w_illum)), flush=True)
        tqdm.pandas(desc="Getting comparison sites", miniters=len(valid_muts_w_illum)/100)
        valid_muts_w_illum['comparison_sites'] = valid_muts_w_illum.progress_apply(
            lambda mut_event: self._select_correl_sites(mut_event, corr_direction), axis = 1
            )
        valid_muts_w_illum['comparison_dists'] = [
            [i for i in range(self.num_correl_sites)] 
            for _ in range(len(valid_muts_w_illum))
            ]
        valid_muts_w_illum['mut_cpg'] = valid_muts_w_illum['chr'] + ':' + valid_muts_w_illum['start'].astype(str)
        valid_muts_w_illum['mut_event'] = valid_muts_w_illum['case_submitter_id'] + '_' + valid_muts_w_illum['mut_cpg']
        valid_muts_w_illum['is_background'] = False
        valid_muts_w_illum['index_event'] = 'self'
        valid_muts_w_illum.reset_index(drop = True, inplace = True)
        pd.options.mode.chained_assignment = 'warn'
        return valid_muts_w_illum

    def look_for_disturbances(
        self, 
        start_num_mut_to_process: int,
        end_num_mut_to_process: int,
        linkage_method: str,
        out_dir: str,
        corr_direction: str,
        comparison_sites_df: pd.DataFrame = None
        ) -> tuple:
        """
        Driver for the analysis. Finds mutations with VAF >= min_VAF_percentile that have a measured CpG within max_dist of the mutation and then looks for disturbances in the methylation of these CpGs.
        @ min_VAF_percentile: minimum VAF percentile of mutation to be considered
        @ max_delta_mf_percentile: maximum delta methylation percentile of mutation to be considered
        @ linkage_method: 'dist' or 'correl'
        @ out_dir: directory to write output files to
        @ comparison_sites_df: optional input of already computed comparison sites
        @ corr_direction: 'pos' or 'neg' for positive or negative correlation
        """
        # PHASE 1: choose mutation events and comparison sites
        ######################################################
        # for each mutation, get a list of the CpGs #id in illum_locs that are 
        # within max_dist of the mutation 'start' or top correlated, depending on linkage_method
        if comparison_sites_df is None:
            if linkage_method == 'dist':
                """comparison_sites_df = self._find_nearby_measured_cpgs(min_VAF_percentile, corr_direction)
                # drop rows of comparison_sites_df where comparison_sites has length 0
                comparison_sites_df = comparison_sites_df[comparison_sites_df['comparison_sites'].apply(lambda x: len(x) > 0)]"""
            elif linkage_method == 'correl':
                comparison_sites_df = self._find_correl_measured_cpgs(
                    start_num_mut_to_process, end_num_mut_to_process, corr_direction
                    )
            else:
                raise ValueError('linkage_method must be "dist" or "correl"')
            # TODO: make background events work with distance
            # for each selected mutation event, choose background sites
            if self.num_background_events > 0:
                # update comparison_sites_df to include background mutation events
                comparison_sites_df = self._choose_background_events(comparison_sites_df, corr_direction = corr_direction)
            # convert comparison_sites_df to dask and write to multiple parquet files
            comparison_sites_dd = dask.dataframe.from_pandas(comparison_sites_df, npartitions = 100)
            comparison_sites_fn = os.path.join(
                out_dir, "comparison_sites_{}-{}Muts_{}corrSites_{}_qnorm3SD_{}background".format(
                    start_num_mut_to_process, end_num_mut_to_process, self.num_correl_sites, corr_direction, self.num_background_events
                    )
                )
            comparison_sites_dd.to_parquet(comparison_sites_fn, engine = 'pyarrow', schema='infer')
            print(f"Wrote comparison sites df to {comparison_sites_fn}", flush=True)
            
        # PHASE 2: compare methylation fractions at comparison sites
        #########################################################
        # for each mutation with nearby measured site, compare the methylation of the nearby measured sites
        # in the mutated sample to the other samples of same age and dataset
        all_metrics_df = self.effect_on_each_site(comparison_sites_df)
        print("got all metrics", flush=True)
        #all_metrics_df.reset_index(inplace=True, drop=True)
        # write out to parquet using dask
        all_metrics_dd = dask.dataframe.from_pandas(all_metrics_df, npartitions = 100)
        all_metrics_fn = os.path.join(
            out_dir, "all_metrics_{}-{}Muts_{}corrSites_{}_qnorm3SD_{}background".format(
                start_num_mut_to_process, end_num_mut_to_process, self.num_correl_sites, corr_direction, self.num_background_events
                )
            )
        all_metrics_dd.to_parquet(all_metrics_fn, engine = 'pyarrow')
        print(f"writing results to {all_metrics_fn}", flush=True)
        return comparison_sites_df, all_metrics_df

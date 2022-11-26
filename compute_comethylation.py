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
import os


class mutationScan:
    def __init__(
        self,
        all_mut_w_age_df: pd.DataFrame,
        illumina_cpg_locs_df: pd.DataFrame, 
        all_methyl_age_df_t: pd.DataFrame,
        corr_dir: str,
        age_bin_size: int, 
        max_dist: int,
        num_correl_sites: float 
        ) -> None:

        self.all_mut_w_age_df = all_mut_w_age_df
        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.corr_dir = corr_dir
        self.age_bin_size = age_bin_size
        self.max_dist = max_dist
        self.num_correl_sites = num_correl_sites
        # Preprocessing: subset to only mutations that are C>T, non X and Y chromosomes, and that occured in samples with measured methylation
        self.all_mut_w_age_df = self.all_mut_w_age_df.loc[(self.all_mut_w_age_df['mutation'] == 'C>T')
         & (self.all_mut_w_age_df['chr'] != 'X') & (self.all_mut_w_age_df['chr'] != 'Y')
         & (self.all_mut_w_age_df['case_submitter_id'].isin(self.all_methyl_age_df_t.index))]
        self.all_mut_w_age_df.loc[:, 'mut_cpg'] = self.all_mut_w_age_df['chr'] + ':' + self.all_mut_w_age_df['start'].astype(str)
        # join self.all_mut_w_age_df with the illumina_cpg_locs_df
        all_mut_w_age_illum_df = self.all_mut_w_age_df.copy(deep=True)
        all_mut_w_age_illum_df.loc[:, 'start'] = pd.to_numeric(self.all_mut_w_age_df.loc[:, 'start'])
        self.all_mut_w_age_illum_df = all_mut_w_age_illum_df.merge(
                                        self.illumina_cpg_locs_df, on=['chr', 'start'], how='left')
        # subset illumina_cpg_locs_df to only the CpGs that are measured
        self.illumina_cpg_locs_df = self.illumina_cpg_locs_df.loc[self.illumina_cpg_locs_df['#id'].isin(self.all_methyl_age_df_t.columns)]

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

    def plot_heatmap(self, mut_sample_cpg, nearby_site_diffs_w_illum_df, mut_nearby_measured_w_illum_df, remove_other_muts=True):
        """
        Given a set of linked sites, nonlinked sites, mutated sample, and mutated site, plots a heatmap of the methylation fraction of same age samples at the linked, nonlinked, and mutated sites
        @ mut_sample_cpg: sample_chr:start
        @ linked_sites_names_df: dataframe of linked sites names
        @ nonlinked_sites_names_df: dataframe of nonlinked sites names
        @ mut_in_measured_cpg_w_methyl_age_df: dataframe of mutations in samples
        @ all_methyl_age_df_t: dataframe of methylation data with ages attached
        """
        # get sample name and position of mutation
        mut_sample_name = mut_sample_cpg.split('_')[0]
        # get the names of the nearby sites
        nearby_sites = mut_nearby_measured_w_illum_df[mut_nearby_measured_w_illum_df['mut_event'] == mut_sample_cpg]['comparison_sites'].values[0]
        # get which of the nearby sites were signficant
        sig_status = nearby_site_diffs_w_illum_df[(nearby_site_diffs_w_illum_df['mut_event'] == mut_sample_cpg)]['sig'].to_list()
        matched_samples = self._same_age_and_tissue_samples(mut_sample_name)
        # exclude samples that have ANY mutations within max_dist of a comparison_sites site
        samples_to_exclude = self._detect_effect_in_other_samples(nearby_sites, mut_row = mut_nearby_measured_w_illum_df[mut_nearby_measured_w_illum_df['mut_event'] == mut_sample_cpg])
        # drop entries of matched_samples that are in samples_to_exclude
        before_drop = len(matched_samples)
        matched_samples = [s for s in matched_samples if s not in samples_to_exclude]
        after_drop = len(matched_samples)
        print(f"dropped {before_drop - after_drop} samples")

        # get the distances of each site from the mutated site
        distances = mut_nearby_measured_w_illum_df[mut_nearby_measured_w_illum_df['mut_event'] == mut_sample_cpg]['comparison_dists'].values[0]
        # sort nearby_sites and sig_status by distances, then sort distances in the same way
        nearby_sites = [x for _, x in sorted(zip(distances, nearby_sites))]
        sig_status = [x for _, x in sorted(zip(distances, sig_status))]
        distances = sorted(distances)

        # if the mutated CpG was measured, add it to sites_to_plot
        if not nearby_site_diffs_w_illum_df[nearby_site_diffs_w_illum_df['mut_event'] == mut_sample_cpg]['#id'].isna().any():
            measured_mutated_cpg = nearby_site_diffs_w_illum_df[nearby_site_diffs_w_illum_df['mut_event'] == mut_sample_cpg]['#id'].values[0]
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
        samples_to_plot = np.concatenate((utils.half(matched_samples, 'first'), [mut_sample_name], utils.half(matched_samples, 'second')))
        # select cpgs and samples to plot
        to_plot_df = self.all_methyl_age_df_t.loc[samples_to_plot, nearby_sites]
        _, axes = plt.subplots(figsize=(15,10))
        # make color bar go from 0 to 1 and increase color bar size
        ax = sns.heatmap(to_plot_df, annot=False, center=0.5, xticklabels=False, yticklabels=False, cmap="Blues", cbar_kws={'label': 'Methylation fraction'}, ax=axes, vmin=0, vmax=1)
        ax.figure.axes[-1].yaxis.label.set_size(13)
        # highlight the mutated cpg if it was measured
        if measured_mut:
            # make a dashed rectangle
            ax.add_patch(Rectangle((mut_pos, int(len(utils.half(matched_samples, 'first')))), 1, 1, fill=False, edgecolor='red', lw=1))
        else:
            ax.add_patch(Rectangle((mut_pos, int(len(utils.half(matched_samples, 'first')))), 0, 1, fill=False, edgecolor='red', lw=2))
        ax.set_xticks(np.arange(.5, len(nearby_sites)+.5, 10))
        ax.set_xticklabels([str(distances[i]) for i in range(0, len(distances), 10)], rotation=45, ha='right', rotation_mode='anchor')

        ax.set_yticks(np.arange(.5, len(samples_to_plot)+.5, 1))
        # add a y tick for the mutated sample
        ax.set_yticks([int(len(utils.half(matched_samples, 'first')))+.5])
        # make tick label red and rotate 90 degrees
        ax.set_yticklabels([mut_sample_name], color='red', rotation=90)
        axes.set_xlabel("CpG site distance (bp)")
        axes.set_ylabel("Samples with same tissue and age (+- 5 years) as mutated sample")
        # increase tick label and axes label size
        ax.tick_params(axis='both', which='major', labelsize=13)
        axes.xaxis.label.set_size(15)
        axes.yaxis.label.set_size(15)
        return to_plot_df

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
        this_age = self.all_methyl_age_df_t.loc[sample_name]['age_at_index']
        this_dset = self.all_methyl_age_df_t.loc[sample_name]['dataset']
        # get the mf all other samples of within age_bin_size/2 years of age on either side
        matched_samples = self.all_methyl_age_df_t.loc[(np.abs(self.all_methyl_age_df_t['age_at_index'] - this_age) <= self.age_bin_size/2) & (self.all_methyl_age_df_t['dataset'] == this_dset)].index.copy(deep=True)
        # drop the mutated sample itself
        matched_samples = matched_samples.drop(sample_name)
        return matched_samples.to_list()

    def _detect_effect_in_other_samples(
        self, 
        sites_to_test: list, 
        mut_row: pd.Series
        ) -> list:
        """
        Detect samples that may have been affected by a mutation in the same region as the mutation we are testing, to discard these samples as they are a bad comparison
        @ sites_to_test: list of sites to test, cg######### format
        @ mut_row: the row from mut_nearby_measured specifying the mutation
        @ returns: list of samples that do not have any C>T mutations in max_dist from any of the sites in sites_to_test
        """
        # so it works when called from heatmap and comethylation scan 
        mut_row = mut_row.copy(deep=True).squeeze()
        # not sure why have to do this to make it not give identically labelled series error
        # select rows of self.all_mut_w_age_df that have the same chr and dataset as mut_row
        relevant_mutations = self.all_mut_w_age_df.loc[(self.all_mut_w_age_df['chr'] == mut_row['chr']) & (self.all_mut_w_age_df['dataset'] == mut_row['dataset'])]
        
        # detect samples that have a mutation in the max_dist window of any of the sites in sites_to_test
        # get locations of sites to test
        sites_to_test_locs = self.illumina_cpg_locs_df.loc[self.illumina_cpg_locs_df['#id'].isin(sites_to_test)] 
        # select rows of relevant_mutations that are within max_dist of any of the sites_to_test
        have_illegal_muts = relevant_mutations.loc[relevant_mutations.apply(lambda row: any(np.abs(row['start'] - sites_to_test_locs['start']) <= self.max_dist), axis=1)]

        # detect samples that have a mutation in the mut_cpg or within max_dist of it
        have_illegal_muts = have_illegal_muts.append(relevant_mutations.loc[(relevant_mutations['mut_cpg'] == mut_row['mut_cpg']) | (np.abs(relevant_mutations['start'] - mut_row['start']) <= self.max_dist)])
        return have_illegal_muts['case_submitter_id'].to_list()

    def _compare_sites(
        self, 
        same_age_tissue_methyl_df, 
        mut_sample_methyl_df
        ) -> pd.DataFrame:
        """
        Calculate effect size and pvalue for each comparison
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

    def effect_on_each_site(
        self, 
        comparison_sites_df: pd.DataFrame
        ) -> pd.DataFrame:
        """
        For each mutation, get the effect of the mutation on each measured CpG within max_dist of the mutation
        @ comparison_sites_df
        @ returns: df with statistical test and effect sizes
        """        
        num_skipped = 0
        num_dropped = 0
        all_metrics_dfs = []
        #for _, mut_row in track(comparison_sites_df.iterrows(), description="Analyzing each mutation", total=len(comparison_sites_df)):
        i = 0
        total = len(comparison_sites_df)
        for _, mut_row in comparison_sites_df.iterrows():
            # get the same age and dataset samples
            matched_samples = self._same_age_and_tissue_samples(mut_row['case_submitter_id'])
            # exclude samples that have ANY mutations within max_dist of a comparison site or the mutated site
            samples_to_exclude = self._detect_effect_in_other_samples(mut_row['comparison_sites'], mut_row)
            # drop entries of matched_samples that are in samples_to_exclude
            before_drop = len(matched_samples)
            matched_samples = [s for s in matched_samples if s not in samples_to_exclude]
            after_drop = len(matched_samples)
            num_dropped += before_drop - after_drop
            if len(matched_samples) <= 10:
                num_skipped += 1
                continue
            mut_sample_comparison_sites = self.all_methyl_age_df_t.loc[mut_row['case_submitter_id'], mut_row['comparison_sites']]
            matched_comparison_sites = self.all_methyl_age_df_t.loc[matched_samples, mut_row['comparison_sites']]
            # measure the change in methylation between sites in the mutated samples and in other non-mutated samples of the same age
            metrics = self._compare_sites(matched_comparison_sites, mut_sample_comparison_sites)
            metrics['mut_cpg'] = mut_row['mut_cpg']
            # create column called 'mutated' that is True if the sample is the mutated sample
            metrics['mutated'] = metrics['sample'] == mut_row['case_submitter_id']
            cpg_to_dist_dict = dict(zip(mut_row['comparison_sites'], mut_row['comparison_dists']))
            metrics['measured_site_dist'] = metrics['measured_site'].map(cpg_to_dist_dict)
            # add to output
            all_metrics_dfs.append(metrics)
            i += 1
            if i % 1000 == 0:
                print("Processed {}% of mutations".format((i/total)*100), flush=True)
        print("WARNING: Not enough samples of the same age and tissue to calculate effect of mutation for {} mutations".format(num_skipped), flush=True)
        print("WARNING: Dropped {} samples due to colliding mutation".format(num_dropped), flush=True)
        all_metrics_df = pd.concat(all_metrics_dfs)
        return all_metrics_df

    def _select_correl_sites(self,
        corr_df: pd.DataFrame,
        mut_cpg: str,
        corr_direction: str
        ) -> list:
        """
        Gets the num sites that are either the most positively or negativelty correlated sites with in_cpg
        @ corr_df: correlation matrix for all sites from one dataset on one chromosome
        @ mut_cpg: the cpg that we are interested in correlating with
        @ corr_direction: 'pos' or 'neg' for positive or negative correlation
        @ returns: list of chosen sites
        """
        corrs = corr_df.loc[mut_cpg]
        # get the value of the 'percentile' highest correlation
        if corr_direction == 'pos':
            q = corrs.quantile(1, interpolation='lower')
            # select num sites closest to q, but not including mut_cpg (which will have a correlation of 1)
            return corrs.iloc[(corrs - q).abs().argsort().iloc[1: self.num_correl_sites + 1]].index.to_list()
        elif corr_direction == 'neg':
            q = corrs.quantile(0, interpolation='higher')
            # here do not need to exclude most correlated site, since it will not be the mut_cpg
            return corrs.iloc[(corrs - q).abs().argsort().iloc[:self.num_correl_sites]].index.to_list()

    def _find_correl_measured_cpgs(
        self, 
        min_VAF_percentile: float, 
        corr_direction: str
        ) -> pd.DataFrame:
        """
        Finds the num_correl_sites most positively or negatively correlated sites with each mutation
        @ min_VAF_percentile: the minimum VAF percentile that a mutation must have to be considered
        @ corr_direction: 'pos' or 'neg' for positive or negative correlation
        @ returns: df of mutations and correlated site pairs
        """
        mut_correl_measured_l = []
        for chrom in self.all_mut_w_age_df['chr'].unique():
            # illum is already subset to measured cpgs, so just choose the chrom
            this_chr_measured_cpgs = self.illumina_cpg_locs_df[self.illumina_cpg_locs_df['chr'] == chrom]
            # get mutations on this chrom, with VAF > min_VAF_percentile
            mut_locs = self.all_mut_w_age_illum_df.loc[
                (self.all_mut_w_age_illum_df['chr'] == chrom) & (self.all_mut_w_age_illum_df['DNA_VAF'] 
                >= np.percentile(self.all_mut_w_age_illum_df['DNA_VAF'], min_VAF_percentile))
                & (self.all_mut_w_age_illum_df['#id'].isin(this_chr_measured_cpgs['#id']))
                ]
            # set empty columns of type list
            mut_locs.loc[:, 'comparison_sites'] = [[] for _ in range(len(mut_locs))]
            mut_locs.loc[:, 'comparison_dists'] = [[] for _ in range(len(mut_locs))]
            for dset in mut_locs['dataset'].unique():
                print(chrom, dset)
                # read in precomputed correlation matrix for this chrom and dataset
                corr_df = pd.read_parquet(
                    os.path.join(self.corr_dir, 'chr{}_{}.parquet'.format(chrom, dset)))
                # find the CpGs on the same chr with highest/lowest correlation of methylation fraction
                correl_sites = []
                for _, mut_row in mut_locs.loc[mut_locs['dataset'] == dset, :].iterrows():
                    correl_sites.append(
                        self._select_correl_sites(
                        corr_df = corr_df,
                        mut_cpg = mut_row['#id'],
                        corr_direction = corr_direction)
                        )
                # why this is the necessary syntax is beyond me
                mut_locs.loc[:, 'comparison_sites'].loc[mut_locs['dataset'] == dset] = correl_sites
                mut_locs.loc[:, 'comparison_dists'].loc[mut_locs['dataset'] == dset] = [
                    [i for i in range(self.num_correl_sites)] 
                    for _ in range(len(mut_locs.loc[mut_locs['dataset'] == dset]))
                    ]
                break
            mut_correl_measured_l.append(mut_locs)
            print("Done chrom {}".format(chrom), flush=True)
            break
        mut_correl_measured_df = pd.concat(mut_correl_measured_l)
        mut_correl_measured_df.loc[:, 'mut_cpg'] = mut_correl_measured_df['chr'] + ':' + mut_correl_measured_df['start'].astype(str)
        return mut_correl_measured_df

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

    def look_for_disturbances(
        self, 
        min_VAF_percentile: float, 
        linkage_method: str,
        comparison_sites_df = None,
        corr_direction: str = 'pos'
        ) -> tuple:
        """
        Driver for the analysis. Finds mutations with VAF >= min_VAF_percentile that have a measured CpG within max_dist of the mutation and then looks for disturbances in the methylation of these CpGs.
        @ max_dist: maximum distance between mutation and measured CpG to be considered
        @ min_VAF: minimum VAF of mutation to be considered
        """
        # for each mutation, get a list of the CpGs #id in illum_locs that are within max_dist of the mutation 'start'
        if comparison_sites_df is None:
            if linkage_method == 'dist':
                comparison_sites_df = self._find_nearby_measured_cpgs(min_VAF_percentile, corr_direction)
            elif linkage_method == 'correl':
                comparison_sites_df = self._find_correl_measured_cpgs(min_VAF_percentile, corr_direction)
            else:
                raise ValueError('linkage_method must be "dist" or "correl"')
        # drop rows of comparison_sites_df where comparison_sites has length 0
        comparison_sites_df = comparison_sites_df[comparison_sites_df['comparison_sites'].apply(lambda x: len(x) > 0)]
        print(comparison_sites_df)
        # for each mutation with nearby measured site, compare the methylation of the nearby measured sites in the mutated sample to the other samples of same age and dataset
        all_metrics_df = self.effect_on_each_site(comparison_sites_df)
        # fdr correct pvals
        all_metrics_df = utils.fdr_correct(all_metrics_df, pval_col_name = 'ztest_pval')
        all_metrics_df.reset_index(inplace=True, drop=True)
        self.all_metrics_df = all_metrics_df

        return comparison_sites_df, all_metrics_df

#####################################################################
#####################################################################
#####################################################################
#####################################################################

def max_prop_effected(
    to_search_in: pd.DataFrame, 
    window_size: int, 
    windows: list
    ) -> pd.DataFrame:
    # calculate a props df for each 1000bp window
    props_dfs = {}
    for window_start in windows:
        this_window_df = to_search_in.loc[(to_search_in['measured_site_dist'] >= window_start)
                        & (to_search_in['measured_site_dist'] < window_start + window_size)]
        props_mut = {}
        delta_mf_step = .2
        delta_mf_bins = [i/10 for i in range(-10, 10, 2)]
        for bin_start in delta_mf_bins:
            num_sites_this_window = len(this_window_df[this_window_df['mutated'] == True]['delta_mf_median'])
            if num_sites_this_window == 0:
                proportion = np.nan
            else: 
                proportion = len(this_window_df[(this_window_df['delta_mf_median'] >= bin_start)
                            & (this_window_df['delta_mf_median'] < bin_start + delta_mf_step)
                            & (this_window_df['mutated'] == True)]) / num_sites_this_window
            props_mut[bin_start] = proportion
        props_nonmut = {}
        for bin_start in delta_mf_bins:
            num_sites_this_window = len(this_window_df[this_window_df['mutated'] == False]['delta_mf_median'])
            if num_sites_this_window == 0:
                proportion = np.nan
            else:
                proportion = len(this_window_df[(this_window_df['delta_mf_median'] >= bin_start)
                            & (this_window_df['delta_mf_median'] < bin_start + delta_mf_step)
                            & (this_window_df['mutated'] == False)]) / num_sites_this_window
            props_nonmut[bin_start] = proportion
        # combine these dicts into a dataframe
        props_df = pd.DataFrame.from_dict(props_mut, orient='index', columns=['mutated'])
        props_df['nonmutated'] = props_nonmut.values()
        props_df['Ratio of probability'] = props_df['mutated'] / props_df['nonmutated']
        props_df.index = [round(i, 1) for i in props_df.index.to_list()]
        props_dfs[window_start] = props_df
    # combine all the props dfs into one
    props_df = pd.concat(props_dfs, axis=0).reset_index()
    props_df.columns = ['Distance', 'Delta MF', 'mutated', 'nonmutated', 'Ratio of probability']
    return props_df


def prop_effected_distr(
    dists: list, 
    metrics_df: pd.DataFrame
    ) -> None:

    axes_int = 0
    fig, axes = plt.subplots(3, len(dists), figsize=(5*len(dists), 10), dpi=100, gridspec_kw={'height_ratios': [2, 1, 1.5]}, sharey='row')
    axes = axes.flatten()

    for i in range(len(dists)):
        close_diffs_df = metrics_df.loc[(metrics_df['measured_site_dist'].abs() >= dists[i][0])
                        & (metrics_df['measured_site_dist'].abs() < dists[i][1])]

        props_mut = {}
        delta_mf_step = .2
        delta_mf_bins = [i/10 for i in range(-10, 10, 2)]
        for bin_start in delta_mf_bins:
            proportion = len(close_diffs_df[(close_diffs_df['delta_mf_median'] >= bin_start)
                            & (close_diffs_df['delta_mf_median'] < bin_start + delta_mf_step)
                            & (close_diffs_df['mutated'] == True)]) / len(close_diffs_df[close_diffs_df['mutated'] == True]['delta_mf_median'])
            props_mut[bin_start] = proportion
        props_nonmut = {}
        for bin_start in delta_mf_bins:
            proportion = len(close_diffs_df[(close_diffs_df['delta_mf_median'] >= bin_start)
                            & (close_diffs_df['delta_mf_median'] < bin_start + delta_mf_step)
                            & (close_diffs_df['mutated'] == False)]) / len(close_diffs_df[close_diffs_df['mutated'] == False]['delta_mf_median'])
            props_nonmut[bin_start] = proportion
        # combine these dicts into a dataframe
        props_df = pd.DataFrame.from_dict(props_mut, orient='index', columns=['mutated'])
        props_df['nonmutated'] = props_nonmut.values()
        props_df['Ratio of probability'] = props_df['mutated'] / props_df['nonmutated']
        props_df.index = [round(j, 1) for j in props_df.index.to_list()]
        sns.histplot(data=close_diffs_df, x='delta_mf_median', hue='mutated', palette=['maroon', 'steelblue'],
            bins=[i/10 for i in range(-10, 11, 2)], log_scale=[False, True], common_norm=False, common_bins=True,
            stat='probability', kde=True, kde_kws={'bw_adjust': 3}, alpha=.3, ax = axes[axes_int])
        axes[axes_int].set_xlim(-1, 1)
        axes[axes_int].set_title(f'Positively correlated sites: {dists[i][0]} - {dists[i][1]}')
        # set x label
        axes[axes_int].set_xlabel(r"$\Delta$MF")
        axes[axes_int].xaxis.set_tick_params(labelbottom=True)
        axes[axes_int].yaxis.set_tick_params(labelleft=True)
        # ratio of pdfs
        sns.lineplot(data=props_df, x=props_df.index + .05, y='Ratio of probability',
                    palette=['maroon'], ax = axes[axes_int + len(dists)],
                    color='black', marker='o')
        # set x lim
        axes[axes_int + len(dists)].set_xlim(-1, 1)
        axes[axes_int + len(dists)].axhline(y=1, color='black', linestyle='--')
        axes[axes_int + len(dists)].set_xlabel(r"$\Delta$MF")
        axes[axes_int + len(dists)].yaxis.set_tick_params(labelleft=True)

        sns.histplot(data=close_diffs_df, x='distance', bins=10, log_scale=[True, False], color='grey', ax = axes[axes_int + 2*len(dists)])
        axes[axes_int + 2*len(dists)].set_xlabel('Distance (bp)')
        axes[axes_int + 2*len(dists)].set_ylabel('Count of sites')
        axes_int += 1

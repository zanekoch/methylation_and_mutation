import pandas as pd
from scipy import stats
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import ticker as mticker
from matplotlib.patches import Rectangle
import utils
import seaborn as sns
from statsmodels.stats.weightstats import ztest as ztest
import ray
from rich.progress import track
import random


PERCENTILES = [1]#np.flip(np.linspace(0, 1, 6))

class mutationScanDistance:
    def __init__(self, all_mut_df, illumina_cpg_locs_df, all_methyl_age_df_t, age_bin_size = 10):
        self.mut_df = all_mut_df
        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.age_bin_size = age_bin_size

    def find_nearby_measured_cpgs(self, max_dist):
        """
        Find the measured cpgs within max_dist of each mutation
        @ returns: df of mutations that have at least one measured CpG within max_dist of the mutation. 'close_measured' column is a list of the measured cpgs within max_dist of the mutation.
        """
        mut_nearby_measured_l = []
        for chrom in self.mut_df['chr'].unique():
            illum_locs = self.illumina_cpg_locs_df[self.illumina_cpg_locs_df['chr'] == chrom]
            mut_locs = self.mut_df.loc[self.mut_df['chr'] == chrom]
            # for each mutation, get a list of the measured CpGs #id in illum_locs that are within max_dist of the mutaiton 'start' but 0 distance (the same CpG)
            mut_locs['close_measured'] = mut_locs.apply(lambda x: list(illum_locs[(np.abs(x['start'] - illum_locs['start']) <= max_dist) & (x['start'] -illum_locs['start'] != 0)]['#id']), axis = 1)
            # also get a list of the distances of these sites
            mut_locs['close_measured_dists'] = mut_locs.apply(lambda x: list(illum_locs[(np.abs(x['start'] - illum_locs['start']) <= max_dist) & (x['start'] -illum_locs['start'] != 0)]['start'] - x['start']), axis = 1)
            # drop all rows of mut_locs where close_measured is empty
            mut_locs = mut_locs[mut_locs['close_measured'].apply(lambda x: len(x) > 0)]
            mut_nearby_measured_l.append(mut_locs)
        mut_nearby_measured_df = pd.concat(mut_nearby_measured_l)
        mut_nearby_measured_df['mut_cpg'] = mut_nearby_measured_df['sample'] + '_' + mut_nearby_measured_df['chr'] + ':' + mut_nearby_measured_df['start'].astype(str)
        return mut_nearby_measured_df

    def volcano_plot(self, nearby_site_diffs_df):
        """
        Plot a volcano plot of the nearby_site_diffs_df
        """
        # get the log10 of the pvals
        nearby_site_diffs_df['log10_pval'] = nearby_site_diffs_df['fdr_pval'].apply(lambda x: -np.log10(x))
        # get the log2 of the fold change
        # color points orange if they are significant
        sns.scatterplot(y = 'log10_pval', x = 'delta_mf', data = nearby_site_diffs_df, alpha=0.3, hue = 'sig', palette = {True: 'orange', False: 'grey'})


        plt.xlabel(r"$\Delta$MF")
        plt.ylabel('-log10 pval')
        plt.show()

    def effect_violin(self, nearby_site_diffs_df, pval,  sig_thresh = .05, groupby_dist = False):
        """
        Make a violin plot of the effect of mutations on the nearby measured cpgs
        """
        fig, axes = plt.subplots(figsize=(10, 5), dpi=100)
        # subset to only significant sites
        nearby_site_diffs_df = nearby_site_diffs_df[nearby_site_diffs_df[pval] < sig_thresh]
        if groupby_dist:
            nearby_site_diffs_df.loc[:,'measured_site_dist'] = np.abs(nearby_site_diffs_df['measured_site_dist'])
            # create 5 equal width bins of distances and assign each row to a distance bin
            nearby_site_diffs_df.loc[:,'dist_bin'] = pd.cut(nearby_site_diffs_df['measured_site_dist'], bins=5)
            # create a violin plot of the effect of mutations on the nearby measured cpgs
            sns.violinplot(data=nearby_site_diffs_df, x='dist_bin', y='delta_mf', cut=0, inner="quartile", scale="count")
            sns.stripplot(data=nearby_site_diffs_df, x='dist_bin', y='delta_mf', color="black", edgecolor="black", alpha=0.3)
        else:
            sns.violinplot(y="delta_mf", data=nearby_site_diffs_df, cut=0, inner="quartile")
            sns.stripplot(y="delta_mf", data=nearby_site_diffs_df, color="black", edgecolor="black", alpha=0.3)
        axes.set_ylabel(r"$\Delta$MF")

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

    def _detect_effect_in_other_samples(self, sites_to_test, mut_nearby_measured_w_illum_df):
        """
        Return the samples that had a mutation in sites_to_test or had a mutations whose nearby sites overlap with a site in sites_to_test
        @ sites_to_test: the sites to see if were mutated or in close_measured in other samples. Can either be measured CpGs (cg#####) or non measured (sample_chr:pos)
        @ mut_nearby_measured_df: df of mutations that have at least one measured CpG within max_dist of the mutation. 'close_measured' column is a list of the measured cpgs within max_dist of the mutation.
        @ df_w_illum_df: 
        @ returns: a list of samples
        """
        samples_to_exclude = np.array([])
        for site_to_test in sites_to_test:
            to_exlude = np.array([])
            if site_to_test[:2] != 'cg':
                # check in which samples site_to_test is in nearby_measured for a mutation
                to_exlude = mut_nearby_measured_w_illum_df[mut_nearby_measured_w_illum_df['close_measured'].apply(lambda x: site_to_test in x)]['sample'].to_numpy()
                # check if site_to_test was mutated in any samples in a measured site
                to_exlude = np.append(to_exlude, mut_nearby_measured_w_illum_df[mut_nearby_measured_w_illum_df['#id'] == site_to_test]['sample'].to_numpy())
            else: # is in form of sample_chr:pos
                # check if site_to_test was mutated in any samples in a measured site
                to_exlude = np.append(to_exlude, mut_nearby_measured_w_illum_df[mut_nearby_measured_w_illum_df['mut_cpg'] == site_to_test]['sample'].to_numpy())
            samples_to_exclude = np.append(samples_to_exclude, to_exlude)
        return samples_to_exclude

    def _join_with_illum(self, df):
        """
        Join the dataframe with the illumina_cpg_locs_df
        """
        df[['sample', 'mut_cpg_chr_start']] = df['mut_cpg'].str.split('_', expand=True)
        df[['chr', 'start']] = df['mut_cpg_chr_start'].str.split(':', expand=True)
        df['start'] = df['start'].astype(int)
        df_w_illum = df.merge(self.illumina_cpg_locs_df, on=['chr', 'start'], how='left')
        return df_w_illum

    def plot_heatmap(self, mut_sample_cpg, nearby_site_diffs_df, mut_nearby_measured_df, remove_other_muts=True):
        """
        Given a set of linked sites, nonlinked sites, mutated sample, and mutated site, plots a heatmap of the methylation fraction of same age samples at the linked, nonlinked, and mutated sites
        @ mut_sample_cpg: sample_chr:start
        @ linked_sites_names_df: dataframe of linked sites names
        @ nonlinked_sites_names_df: dataframe of nonlinked sites names
        @ mut_in_measured_cpg_w_methyl_age_df: dataframe of mutations in samples
        @ all_methyl_age_df_t: dataframe of methylation data with ages attached
        """
        # join with illumina_cpg_locs_df to get the CpG names of mutated CpGs if they were in a measured site
        nearby_site_diffs_w_illum_df = self._join_with_illum(nearby_site_diffs_df)
        mut_nearby_measured_w_illum_df = self._join_with_illum(mut_nearby_measured_df)
        # get sample name and position of mutation
        mut_sample_name = mut_sample_cpg.split('_')[0]
        # get the names of the nearby sites
        nearby_sites = mut_nearby_measured_df[mut_nearby_measured_df['mut_cpg'] == mut_sample_cpg]['close_measured'].values[0]
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
        distances = mut_nearby_measured_df[mut_nearby_measured_df['mut_cpg'] == mut_sample_cpg]['close_measured_dists'].values[0]
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
            rectangle = True
        else:
            rectangle = False
        # list of samples to plot
        samples_to_plot = np.concatenate((utils.half(same_age_tissue_samples, 'first'), [mut_sample_name], utils.half(same_age_tissue_samples, 'second')))
        # select cpgs and samples to plot
        to_plot_df = self.all_methyl_age_df_t.loc[samples_to_plot, nearby_sites]

        _, axes = plt.subplots(figsize=(15,10))
        # make color bar go from 0 to 1
        ax = sns.heatmap(to_plot_df, annot=False, center=0.5, xticklabels=False, yticklabels=False, cmap="Blues", cbar_kws={'label': 'Methylation fraction'}, ax=axes, vmin=0, vmax=1)
        # highlight the mutated cpg if it was measured
        if rectangle:
            # make a dashed rectangle
            ax.add_patch(Rectangle((mut_pos, int(len(utils.half(same_age_tissue_samples, 'first')))), 1, 1, fill=False, edgecolor='red', lw=1, ls='--'))
        ax.set_xticks(np.arange(.5, len(nearby_sites)+.5, 1))
        ax.set_xticklabels([str(distances[i]) + '*' if sig_status[i] == True else str(distances[i]) for i in range(len(distances))], rotation=90)
        ax.set_yticks(np.arange(.5, len(samples_to_plot)+.5, 1))
        # add a y tick for the mutated sample
        ax.set_yticks([int(len(utils.half(same_age_tissue_samples, 'first')))+.5])
        # make tick label red and rotate 45 degrees
        ax.set_yticklabels([mut_sample_name], color='red', rotation=90)
        axes.set_xlabel("Nearby CpG sites distance (bp)")
        axes.set_ylabel("Samples with same tissue and age (+- 5 years) as mutataed sample")
        return

    def effect_on_each_site(self, mut_nearby_measured_df):
        """
        For each mutation, get the effect of the mutation on each measured CpG within max_dist of the mutation
        @ mut_nearby_measured_df
        @ returns: df with columns=[measured_site, mut_cpg, delta_mf, ztest_pval] and rows=[each mut_cpg, measured_site pair]
        """
        mut_nearby_measured_w_illum_df = self._join_with_illum(mut_nearby_measured_df)

        nearby_site_diffs = []
        for _, mut_row in track(mut_nearby_measured_df.iterrows(), description="Analyzing each mutation", total=len(mut_nearby_measured_df)):
        #for _, mut_row in mut_nearby_measured_df.iterrows():
            # get the mutated sample  and the MF's of the same age samples
            same_age_tissue_samples_mf_df = self._same_age_and_tissue_samples(mut_row['sample'])
            # exclude samples that have a mutation in the same CpG, a mutation in a nearby CpG, or a mutation in a nearby measured CpG  
            samples_to_exclude = self._detect_effect_in_other_samples(mut_row['close_measured'], mut_nearby_measured_w_illum_df)
            same_age_tissue_non_mut_samples = np.setdiff1d(same_age_tissue_samples_mf_df.index.values, samples_to_exclude)
            if (len(same_age_tissue_samples_mf_df.index.values) - len(same_age_tissue_non_mut_samples)) > 0:
                print("{} samples excluded".format(len(same_age_tissue_samples_mf_df.index.values) - len(same_age_tissue_non_mut_samples)))
            same_age_tissue_samples_mf_df = same_age_tissue_samples_mf_df.loc[same_age_tissue_non_mut_samples]

            nearby_sites = mut_row['close_measured']
            # get this mutated sample's MF at comparison sites
            mut_sample_nearby_mfs = self.all_methyl_age_df_t.loc[mut_row['sample'], nearby_sites] 
            # measure the change in methylation between sites in the mutated sample and in other non-mutated samples of the same age
            # returns df with rows being comparison sites, columns delta_mf and ztest_pval 
            # TODO: change to also exlude other samples with a mutaiton in a CpG nearby
            nearby_diff = compare_sites(same_age_tissue_samples_mf_df[nearby_sites], mut_sample_nearby_mfs)
            # add to output
            nearby_diff['mut_cpg'] = mut_row['sample'] + '_' + mut_row['chr'] + ':' + str(mut_row['start'])
            nearby_diff['measured_site_dist'] = mut_row['close_measured_dists']
            nearby_diff['sample'] = mut_row['sample']
            nearby_site_diffs.append(nearby_diff)
        # concat all the dfs in nearby_site_diffs into one df
        nearby_site_diffs_df = pd.concat(nearby_site_diffs)
        # reset index and make old index a column called measured_site
        nearby_site_diffs_df = nearby_site_diffs_df.reset_index().rename(columns = {'index': 'measured_site'})
        return nearby_site_diffs_df

    def look_for_disturbances(self, max_dist):
        # subset to only mutations that are C>T, non X and Y chromosomes, mutations only those that occured in samples with measured methylation, and select rows in largest 20 percentile of DNA_VAF
        self.mut_df = self.mut_df[self.mut_df['mutation'] == 'C>T']
        self.mut_df = self.mut_df[(self.mut_df['chr'] != 'X') & (self.mut_df['chr'] != 'Y')]
        self.mut_df = self.mut_df[self.mut_df['sample'].isin(self.all_methyl_age_df_t.index)]
        """self.mut_df = self.mut_df[self.mut_df['DNA_VAF'] >= np.percentile(self.mut_df['DNA_VAF'], 75)]"""
        # subset illumina_cpg_locs_df to only the CpGs that are measured
        self.illumina_cpg_locs_df = self.illumina_cpg_locs_df[self.illumina_cpg_locs_df['#id'].isin(self.all_methyl_age_df_t.columns)]

        # for each mutation, get a list of the CpGs #id in illum_locs that are within max_dist of the mutation 'start'
        mut_nearby_measured_df = self.find_nearby_measured_cpgs(max_dist)

        # for each mutation with nearby measured site, compare the methylation of the nearby measured sites in the mutated sample to the other samples of same age and dataset
        nearby_site_diffs_df = self.effect_on_each_site(mut_nearby_measured_df)
        self.nearby_site_diffs_df = nearby_site_diffs_df

        return mut_nearby_measured_df, nearby_site_diffs_df


class MethylChangeByAgeCorr:
    """
    Class that assigns samples to age bins and then calculates observed and predicted mutational methylome difference between age bins
    """
    def __init__(self, methyl_age_df_t, illumina_cpg_locs_df, dset, num_age_bins = 10):
        # subset to given dataset
        methyl_age_df_t = methyl_age_df_t[methyl_age_df_t['dataset'] == dset]
        # drop X, Y, MT chroms
        methyl_age_df_t = utils.drop_cpgs_by_chrom(methyl_age_df_t, ['X', 'Y'], illumina_cpg_locs_df)
        self.methyl_age_df_t = methyl_age_df_t
        self.dset = dset
        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.num_age_bins = num_age_bins

    def split_into_age_bins(self):
        """
        Assign each sample in self.methyl_age_df_t to an age bin
        """
        self.methyl_age_df_t["age_bin"] = pd.qcut(self.methyl_age_df_t['age_at_index'], q=self.num_age_bins, labels=False)

    def get_observed_methylome_diffs(self):
        """
        Get sum of positive differences and negative differences between the mean of each consecutive age bin
        """
        # add age_bins to methyl_age_df_t
        self.split_into_age_bins()
        # get the difference between the mean methylome of each age bin
        obs_methylome_diffs = self.methyl_age_df_t.groupby(['age_bin']).mean().diff(axis=0)
        # get sum of positive differences and negative differences between each consecutive age bin
        obs_methylome_diffs['pos_diff'] = obs_methylome_diffs[obs_methylome_diffs > 0].sum(axis=1)
        obs_methylome_diffs['neg_diff'] = obs_methylome_diffs[obs_methylome_diffs < 0].sum(axis=1)
        # drop first row (which is nan) because nothing to compare it to
        obs_methylome_diffs = obs_methylome_diffs.iloc[1:]
        self.obs_methylome_diffs = obs_methylome_diffs
        return obs_methylome_diffs

    def get_predicted_methylome_diffs(self, all_mut_df, linked_sites_diffs_dfs, linked_sites_pvals_dfs, pval_cutoff=0.05):
        """
        Get the predicted methylome diff for each age bin. 
        - this is given by the difference in average number of mutations in each age bin and then converted to expected methylome change by MutationImpact
        """
        # initialize a MutationImpact object
        mi = MutationImpactCorr(all_mut_df, linked_sites_diffs_dfs, linked_sites_pvals_dfs)
        # get the expected methylome change for each sample based on mutations
        expected_mut_eff = mi.expected_mut_induced_mf_change(pval_cutoff = pval_cutoff)
        
        # add ages and age bins to expected_mut_eff by joining to self.methyl_age_df_t
        expected_mut_eff_w_age = expected_mut_eff.join(self.methyl_age_df_t[['age_at_index', 'age_bin']], on='sample')

        # group by age bin and calc mean of pos_expected_change, pos_standard_err, neg_expected_change, neg_standard_err
        pred_methylome_diffs = expected_mut_eff_w_age.groupby(['age_bin']).mean()
        # drop first row/age bin to match observed methylome diffs
        pred_methylome_diffs = pred_methylome_diffs.iloc[1:]

        self.pred_methylome_diffs = pred_methylome_diffs
        return pred_methylome_diffs

    def plot_observed_vs_predicted(self):
        """
        Create a barplot of observed vs predicted methylome differences for each age bin
        """
        # unroll and stack actual_methylome_diffs[['pos_diff', 'neg_diff']]
        obs_methylome_diffs_unstacked = self.obs_methylome_diffs[['pos_diff', 'neg_diff']].unstack().reset_index()
        obs_methylome_diffs_unstacked.columns = ['Direction', 'Age bin', 'Methylome difference']
        obs_methylome_diffs_unstacked['Type'] = "Observed"
        obs_methylome_diffs_unstacked['Error'] = 0
        # same for predicted_methylome_diffs
        pred_methylome_diffs_unstacked = self.pred_methylome_diffs[['pos_expected_change', 'neg_expected_change']].unstack().reset_index()
        pred_methylome_diffs_unstacked.columns = ['Direction', 'Age bin', 'Methylome difference']
        pred_methylome_diffs_unstacked['Type'] = "Predicted"
        # get standard errors for predicted to plot as error bars
        pred_methylome_diffs_se = self.pred_methylome_diffs[['pos_standard_err', 'neg_standard_err']].unstack().reset_index()
        pred_methylome_diffs_se.columns = ['Direction', 'Age bin', 'Standard error']
        # add error column to pred_methylome_diffs_unstacked
        pred_methylome_diffs_unstacked['Error'] = pred_methylome_diffs_se['Standard error'].values
        # concat
        all_methylome_diffs = pd.concat([obs_methylome_diffs_unstacked, pred_methylome_diffs_unstacked])
        # plot
        def grouped_barplot(df, cat, subcat, val , err):
            fig, axes = plt.subplots(1, 1, figsize=(10, 5), dpi=150)
            u = df[cat].unique()
            x = np.arange(len(u))
            subx = df[subcat].unique()
            offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
            width= np.diff(offsets).mean()
            for i,gr in enumerate(subx):
                dfg = df[df[subcat] == gr]
                axes.bar(x+offsets[i], dfg[val].values, width=width, 
                        label="{} {}".format(subcat, gr), yerr=dfg[err].values)
            axes.set_xlabel(cat)
            axes.set_ylabel(val)
            axes.set_xticks(x)
            axes.set_xticklabels(u)
            axes.legend()
        grouped_barplot(all_methylome_diffs, 'Age bin', 'Direction', 'Methylome difference', 'Error')
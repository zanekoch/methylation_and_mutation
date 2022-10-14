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

def select_corr_sites(in_cpg,
                            corr_df,
                            num,
                            percentile):
    """
    Gets the num sites that are in the top Percentile percentile of most positively correlated sites with in_cpg
    @ in_cpg: cpg to get correlated sites for
    @ corr_df: df of correlation between sites and in_cpg
    @ num: number of sites to select
    @ percentile: percentile of sites to select
    @ returns: list of sites to select
    """
    # get the value of the qth percentile cpg site in absolute value so negatively correlated sites are included
    q = np.abs(corr_df[in_cpg]).quantile(percentile, interpolation='lower')
    # select num sites closest to q
    comparison_sites = corr_df.iloc[(np.abs(corr_df[in_cpg]) - q).abs().argsort().iloc[:num], 0].index
    return comparison_sites

def select_random_sites(cpgs_names, num):
    return np.random.choice(cpgs_names, num, replace=False)

def select_random_same_mean_mf_sites(linked_sites, all_methyl_age_df_t, same_age_tissue_samples):
    """
    NOTE: Depending on random variation, may end up returning nonlinked sites with some duplicates.
    Select num_linked_sites nonlinked sites with the same mean methylation fraction as the linked sites
    @ linked_sites: list of linked sites
    @ all_methyl_age_df_t: df of methylation fraction for all sites
    @ same_age_tissue_samples: list of samples with same age and tissue as mutated sample
    @ returns: list of nonlinked sites
    """
    # get mean methylation fraction of the chosen linked sites
    linked_sites_mean_mfs = all_methyl_age_df_t.loc[same_age_tissue_samples, linked_sites].mean(axis=0, skipna=True)
    # subset all_methyl_age_df_t to only include sites not in linked sites
    all_nonlinked_sites_means = all_methyl_age_df_t.loc[same_age_tissue_samples, (~all_methyl_age_df_t.columns.isin(linked_sites)) & (~all_methyl_age_df_t.columns.isin(['age_at_index', 'dataset']))].mean(axis=0, skipna=True)
    # for each of the linked sites, select the site with closest mean methylation fraction
    chosen_nonlinked_sites = []
    for mean_mf in linked_sites_mean_mfs.values:
        # select the site from all_nonlinked_sites with the closest mean methylation fraction to mean_mf
        min_diff_site = np.abs(all_nonlinked_sites_means - mean_mf).idxmin()
        chosen_nonlinked_sites.append(min_diff_site)
        # drop the chosen site from all_nonlinked_sites_means
        #all_nonlinked_sites_means = all_nonlinked_sites_means.drop(min_diff_site)
    return chosen_nonlinked_sites

def select_closest_sites_2d_distance(in_cpg,
                         dist_df,
                         num,
                         percentile):
    """
    Gets the num sites that are in the top Percentile percentile of closet sites to in_cpg
    @ in_cpg: cpg to get closest sites for
    @ dist_df: df of distance between sites and in_cpg
    @ num: number of sites to select
    @ percentile: percentile of sites to select
    @ returns: list of sites to select
    """
    # get the value of the qth percentile cpg site
    q = dist_df[in_cpg].quantile(percentile, interpolation='lower')
    # select num sites closest to q
    comparison_sites = dist_df.iloc[(dist_df[in_cpg] - q).abs().argsort().iloc[:num], 0].index
    return comparison_sites

def select_closest_sites_hic_distance(in_cpg,
                         dist_df,
                         num,
                         percentile):
    """
    Gets the num sites that are in the top Percentile percentile of closet sites to in_cpg
    @ in_cpg: cpg to get closest sites for
    @ dist_df: df of distance between sites and in_cpg based on HIC data (largest number = clostest)
    @ num: number of sites to select
    @ percentile: percentile of sites to select
    @ returns: list of sites to select
    """
    # get the value of the qth percentile cpg site
    q = dist_df[in_cpg].quantile(percentile, interpolation='lower')
    # select num sites closest to q
    comparison_sites = dist_df.iloc[(dist_df[in_cpg] - q).abs().argsort().iloc[:num], 0].index
    return comparison_sites


def test_linked_vs_nonlinked(linked_diff, nonlinked_diff):
    """
    Tests if the mean difference of linked and nonlinked sites is significantly different
    @ linked_mean_diff: mean difference of linked sites between mutated and non-mutated same age samples
    @ nonlinked_mean_diff: mean difference of nonlinked sites between mutated and non-mutated same age samples
    @ returns: list containing p-value of wilcoxon test and barlett's test
    """
    # test for diff in distr. of differences of mutated sample and other same age samples at linked vs non inked sites (we expect differences to be larger in magntiude in linked sites)
    pval_wilc = stats.ranksums(linked_diff, nonlinked_diff, alternative='two-sided').pvalue
    _, pval_barlett = stats.bartlett(linked_diff['delta_mf'].values, nonlinked_diff['delta_mf'].values)
    return [pval_wilc, pval_barlett]

def compare_sites(same_age_samples_mf_df, mut_sample_comparison_mfs_df):
    """
    For a given mutation, compare the methylation of input sites (linked or nonlinked) between mutated and non-mutated sample
    @ same_age_samples_mf_df: dataframe of methylation values for non-mutated samples of same age as mutated sample at comparison sites (either linked or nonlinked)
    @ mut_sample_comparison_mfs_df: dataframe of methylation values for mutated sample at comparison sites (either linked or nonlinked)
    @ returns: Dataframe with rows being comparison sites, columns delta_mf (average difference of mutated sample and non mutated sample at the comparison site that is the row) and ztest_pval (which says if the mutated sample was significantly different from the other samples at that site)
    """
    # subtract mut_sample_comparison_mfs_df from every row (sample) in same_age_samples_mf_df
    difference_at_comparison_sites = same_age_samples_mf_df.subtract(mut_sample_comparison_mfs_df.iloc[0])
    # switch the sign to make delta_mf = mut_sample_mf - same_age_sample_mf
    difference_at_comparison_sites = difference_at_comparison_sites.mul(-1)
    # get mean average difference (delta_mf) at each site
    mean_diff_each_comparison_site = pd.DataFrame(difference_at_comparison_sites.mean(axis = 0), columns=['delta_mf'])
    # add mut_sample_comparison_mfs_df as last row of same_age_samples_mf_df
    all_samples_at_comparison_sites = same_age_samples_mf_df.append(mut_sample_comparison_mfs_df)
    # calculate a z score pvalue for the mut_sample site being different from the non-mutated samples
    mean_diff_each_comparison_site['zscore'] = all_samples_at_comparison_sites.apply(lambda x: stats.zscore(x, nan_policy='omit')[-1], axis=0)
    # get 2-sided pvalues
    mean_diff_each_comparison_site['ztest_pval'] = stats.norm.sf(abs(mean_diff_each_comparison_site['zscore']))*2
    return mean_diff_each_comparison_site[['delta_mf', 'ztest_pval']]

def effect_one_mutation(mut_linkage_df,
                            linkage_type,
                            all_methyl_age_df_t,
                            mut_in_measured_cpg_w_methyl_age_df,
                            illumina_cpg_locs_df,
                            percentile,
                            num_linked_sites,
                            age_bin_size,
                            mut_cpg):
    # get chrom of this site
    mut_cpg_chr = illumina_cpg_locs_df[illumina_cpg_locs_df['#id'] == mut_cpg]['chr'].iloc[0]
    # limit comparison sites to cpgs on same chrom that are in all_methyl_age_df_t
    same_chr_cpgs = illumina_cpg_locs_df[(illumina_cpg_locs_df['chr'] == mut_cpg_chr) & (illumina_cpg_locs_df['#id'].isin(all_methyl_age_df_t.columns))]['#id'].to_list()
    same_chr_linkage_df = mut_linkage_df.loc[mut_linkage_df.index.isin(same_chr_cpgs)]
    # drop mut_cpg from same_chr_linkage_df so it is not selected as a comparison site 
    same_chr_linkage_df.drop(mut_cpg, axis=0, inplace=True)
    # get the mutated sample  and the MF's of the same age samples
    mut_sample, same_age_tissue_samples_mf_df = utils.get_same_age_and_tissue_samples(all_methyl_age_df_t, mut_in_measured_cpg_w_methyl_age_df, age_bin_size, mut_cpg)
    this_sample_name = mut_sample['case_submitter_id'].to_numpy()[0]
    # get comparison sites based on method
    if linkage_type == 'methylation_corr':
        # get the positively correlated sites that are on the same chromosome as the mutated CpG
        linked_sites = select_corr_sites(mut_cpg, same_chr_linkage_df, num_linked_sites, percentile)
        # select a random set of non-linked sites from all sites on the same chromosome
        nonlinked_sites = select_random_sites(same_chr_linkage_df.drop(linked_sites, axis=0).index.to_list(), num_linked_sites)
        """# select randomly chosen nonlinked with approximately the same mean methylation as the linked sites, to control for this factor
        nonlinked_sites = select_random_same_mean_mf_sites(linked_sites, all_methyl_age_df_t, same_age_tissue_samples_mf_df.index.to_list())
        # farthest sites are 0th percentile
        nonlinked_sites = select_corr_sites(mut_cpg, same_chr_linkage_df, num_linked_sites, 0)"""
    elif linkage_type == '2d_distance':
        # get the closest sites that are on the same chromosome as the mutated CpG
        linked_sites = select_closest_sites_2d_distance(mut_cpg, same_chr_linkage_df, num_linked_sites, percentile)
        """# farthest sites are 100th percentile
        nonlinked_sites = select_closest_sites_2d_distance(mut_cpg, same_chr_linkage_df, num_linked_sites, 1)"""
        # select a random set of non-linked sites from all sites on the same chromosome
        nonlinked_sites = select_random_sites(same_chr_linkage_df.drop(linked_sites, axis=0).index.to_list(), num_linked_sites)
    elif linkage_type == 'hic_distance':
        # get the closest sites that are on the same chromosome as the mutated CpG
        linked_sites = select_closest_sites_hic_distance(mut_cpg, same_chr_linkage_df, num_linked_sites, percentile)
        # farthest sites are 0th percentile
        nonlinked_sites = select_closest_sites_hic_distance(mut_cpg, same_chr_linkage_df, num_linked_sites, 0)
    else:
        raise ValueError("linkage_type must be either 'methylation_corr' or '2d_distance' or 'hic_distance'")
    # if there are not enough samples of the same age and tissue, warn and skip this site
    if len(same_age_tissue_samples_mf_df) < 10:
        print("WARNING: Not enough samples of the same age and tissue to calculate effect of mutation at site: ".format(mut_cpg), flush=True)
        return None
    # get this mutated sample's MF at comparison sites
    mut_sample_linked_mfs_df = all_methyl_age_df_t.loc[this_sample_name, linked_sites] 
    # measure the change in methylation between linked sites in the mutated sample and in other non-mutated samples of the same age
    # linked_result_df is a dataframe with columns: MabsErr, MavgErr and rows: same age samples and entries the respective metric measuring distance between that sample and mutated sample across all linked sites
    linked_diff = compare_sites(same_age_tissue_samples_mf_df[linked_sites], mut_sample_linked_mfs_df)
    # do same comparison but seeing if unlinked sites also changed same amount
    mut_sample_nonlinked_mfs_df = all_methyl_age_df_t.loc[this_sample_name, nonlinked_sites]
    # nonlinked_result_df is a dataframe with columns: MabsErr, MavgErr and rows: same age samples and entries the respective metric measuring distance between that sample and mutated sample across all nonlinked sites
    nonlinked_diff = compare_sites(same_age_tissue_samples_mf_df[nonlinked_sites], mut_sample_nonlinked_mfs_df)
    # compare mut_result_df to nonlinked_result_df to see if there are less significant differences in less linked CpGs
    this_mut_results = test_linked_vs_nonlinked(linked_diff[['delta_mf']], nonlinked_diff[['delta_mf']])
    # return lists to add to dictionaries
    return mut_cpg, this_mut_results, linked_sites, linked_diff['delta_mf'].to_list(), linked_diff['ztest_pval'].to_list(), nonlinked_sites, nonlinked_diff['delta_mf'].to_list(), nonlinked_diff['ztest_pval'].to_list()

def measure_mut_eff(mut_linkage_df,
                                linkage_type,
                                all_methyl_age_df_t, 
                                mut_in_measured_cpg_w_methyl_age_df,
                                illumina_cpg_locs_df,
                                percentile,
                                num_linked_sites,
                                age_bin_size):
    """
    Given a correlation matrix, calculate the effect of a mutation on the methylation of linked sites (num_linked_sites starting from specified percentile) 
    @ mut_linkage_df: dataframe with columns corresponding to mutated sites we are measured effect of and rows the linkage of that site to all other sites on same chrom
    @ linkage_type: type of linkage in mut_linkage_df either 'methylation_corr' or 'distance'
    @ all_methyl_age_df_t: methylation dataframe with age information
    @ mut_in_measured_cpg_w_methyl_age_df: methylation dataframe with age information for mutated sites
    @ illumina_cpg_locs_df:
    @ percentile: percentile to draw comparison sites from
    @ num_linked_sites: number of comparison sites to draw
    @ age_bin_size: age bin size to use for selected samples to compare at comparison sites
    @ parallel: if 0, run in serial, otherwise let parallel be num_cpus
    @ returns: a dataframe of pvals and effect sizes comparing linked sites in mutated sample to linked sites in non-mutated (within age_bin_size/2 year age) samples, for: MabsErr, MavgErr, pearson r, and WilcoxonP 
    """
    # for each mutated CpG that we have correlation for
    all_results = []
    linked_sites_names_dict, linked_site_diffs_dict, linked_site_z_pvals_dict, nonlinked_sites_names_dict, nonlinked_site_diffs_dict, nonlinked_site_z_pvals_dict = {}, {}, {}, {}, {}, {}
    
    each_perc_result_lists = []
    # iterate across CpGs we are testing
    for mut_cpg in track(mut_linkage_df.columns, description="Analyzing each mutation"):
        each_perc_result_lists.append(
            effect_one_mutation(mut_linkage_df, linkage_type, all_methyl_age_df_t, mut_in_measured_cpg_w_methyl_age_df, illumina_cpg_locs_df, percentile, num_linked_sites, age_bin_size, mut_cpg)
            )
    # go through each result list and if it is == [None], remove it
    each_perc_result_lists = [x for x in each_perc_result_lists if x != None]
    # put the result lists into dictionaries with key being mut_cpg
    for this_perc_result_list in each_perc_result_lists:
        mut_cpg = this_perc_result_list[0]
        all_results.append(this_perc_result_list[1])
        linked_sites_names_dict[mut_cpg] = this_perc_result_list[2]
        linked_site_diffs_dict[mut_cpg] = this_perc_result_list[3]
        linked_site_z_pvals_dict[mut_cpg] = this_perc_result_list[4]
        nonlinked_sites_names_dict[mut_cpg] = this_perc_result_list[5]
        nonlinked_site_diffs_dict[mut_cpg] = this_perc_result_list[6]
        nonlinked_site_z_pvals_dict[mut_cpg] = this_perc_result_list[7]
    
    result_df = pd.DataFrame(all_results, columns = ['p_wilcoxon', 'p_barlett'] )
    linked_sites_names_df = pd.DataFrame.from_dict(linked_sites_names_dict, orient='index')
    linked_sites_diffs_df = pd.DataFrame.from_dict(linked_site_diffs_dict, orient='index')
    linked_sites_z_pvals_df = pd.DataFrame.from_dict(linked_site_z_pvals_dict, orient='index')
    nonlinked_sites_names_df = pd.DataFrame.from_dict(nonlinked_sites_names_dict, orient='index')
    nonlinked_sites_diffs_df = pd.DataFrame.from_dict(nonlinked_site_diffs_dict, orient='index')
    nonlinked_sites_z_pvals_df = pd.DataFrame.from_dict(nonlinked_site_z_pvals_dict, orient='index')
    return result_df, linked_sites_names_df, linked_sites_diffs_df, linked_sites_z_pvals_df, nonlinked_sites_names_df, nonlinked_sites_diffs_df, nonlinked_sites_z_pvals_df

def mutation_eff_varying_linkage_perc(mut_linkage_df,
                                linkage_type,
                                mut_in_measured_cpg_w_methyl_age_df,
                                all_methyl_age_df_t,
                                illumina_cpg_locs_df,
                                num_linked_sites,
                                age_bin_size,
                                parallel=0):
    """
    For each percentile of comparison sites, get the effect size of the mutation in the mutated sample compared to the effect size of the mutation in the non-mutated samples of the same age.
    @ mut_linkage_df: dataframe with columns corresponding to mutated sites we are measured effect of and rows the linkage of that site to all other sites on same chrom
    @ linkage_type: type of linkage in mut_linkage_df either 'methylation_corr', 'distance', or 'hic_distance'
    @ mut_in_measured_cpg_w_methyl_age_df: methylation dataframe with age information for mutated sites
    @ all_methyl_age_df_t: methylation dataframe with age information for all samples
    @ illumina_cpg_locs_df: dataframe of CpG locations
    @ num_linked_sites: number of comparison sites to draw
    @ age_bin_size: age bin size to use for selected samples to compare at comparison sites
    @ parallel: if 0, run in serial, otherwise let parallel be num_cpus
    """ 
    # num cpu's limits the max number of tasks at a given time, by giving each process 2 cpus out of total number
    if parallel != 0:
        ray.init(num_cpus=parallel, ignore_reinit_error=True)
    # calculate results varying percentile of linked CpG sites, only chr1 sites
    result_dfs = []
    linked_sites_names_dfs = []
    linked_sites_diffs_dfs = []
    linked_sites_z_pvals_dfs = []
    nonlinked_sites_names_dfs = []
    nonlinked_sites_diffs_dfs = []
    nonlinked_sites_z_pvals_dfs = []
    for percentile in PERCENTILES:
        print("Starting percentile: {}".format(percentile))
        result_df, linked_sites_names_df, linked_sites_diffs_df, linked_sites_z_pvals_df, nonlinked_sites_names_df, nonlinked_sites_diffs_df, nonlinked_sites_z_pvals_df = measure_mut_eff(mut_linkage_df, linkage_type, all_methyl_age_df_t, mut_in_measured_cpg_w_methyl_age_df, illumina_cpg_locs_df, percentile, num_linked_sites, age_bin_size)
        result_dfs.append(result_df)
        linked_sites_names_dfs.append(linked_sites_names_df)
        linked_sites_diffs_dfs.append(linked_sites_diffs_df)
        linked_sites_z_pvals_dfs.append(linked_sites_z_pvals_df)
        nonlinked_sites_names_dfs.append(nonlinked_sites_names_df)
        nonlinked_sites_diffs_dfs.append(nonlinked_sites_diffs_df)
        nonlinked_sites_z_pvals_dfs.append(nonlinked_sites_z_pvals_df)
    return result_dfs, linked_sites_names_dfs, linked_sites_diffs_dfs, linked_sites_z_pvals_dfs, nonlinked_sites_names_dfs, nonlinked_sites_diffs_dfs, nonlinked_sites_z_pvals_dfs
 

def get_hic_distances(in_cpgs, illumina_cpg_locs_df, hic_fn, this_chr_cpgs_all):
    import cooler
    """
    Get the distance between a set of cpgs and all other cpgs in the HIC map
    @ in_cpgs: list of cpgs to get distances for
    @ illumina_cpg_locs_df: df of cpg locations
    @ hic_fn: path to .cool file of HiC map for one chromosome (user must give the right chromosome file)
    @ this_chr_cpgs_all: list of all cpgs on the same chromosome as the cpgs in in_cpgs, for formatting output
    @ returns: df of distances between in_cpgs and all other cpgs in the HIC map (len(other cpgs) x (in_cpgs)) with np.nan in positions where there is no defined distance betwee sites
    """
    # subset to only cpgs in this_chr_cpgs_all (which are cpgs on chr1 which have methylation data)
    illumina_cpg_locs_df = illumina_cpg_locs_df[illumina_cpg_locs_df['#id'].isin(this_chr_cpgs_all)]
    # get the locations of the input cpgs
    in_cpg_locs = illumina_cpg_locs_df[illumina_cpg_locs_df['#id'].isin(in_cpgs)]
    # read in cool file
    hi_c = cooler.Cooler(hic_fn)
    all_dists_dict = {}
    # for each cpg in in_cpg_locs get the distance from this CpG's region to all other hi_c regions
    for _, row in in_cpg_locs.iterrows():
        this_cpg_name = row['#id']
        cpg_loc = ('chr' + row['chr'], row['start'], row['start'] + 1)
        # a df of the distance from this cpg region to all other cpg regions 
        bins = hi_c.bins()  # fetch all the bins
        # select some pixels with unannotated bins
        pix = hi_c.pixels().fetch(cpg_loc)
        # annotate these bins
        both_dists_w_weight = cooler.annotate(pix, bins)
        # create balanced chromatin contact metric
        both_dists_w_weight['balanced'] = both_dists_w_weight['weight1'] * both_dists_w_weight['weight2'] * both_dists_w_weight['count']
        # drop rows with nan for balanced
        both_dists_w_weight.dropna(subset=['balanced'], how='any', inplace=True)
        # create a dictionary with this_chr_cpgs_all as keys and 0 as values
        one_cpg_dists_dict = dict.fromkeys(this_chr_cpgs_all, 0)
        # for each other CpG region, check if there is a measured CpG site in that region
        for _, other_row in both_dists_w_weight.iterrows():
            # get the cpgs in this region
            other_cpgs_in_region = illumina_cpg_locs_df[(illumina_cpg_locs_df['start'] >= other_row['start2']) 
                                                        & (illumina_cpg_locs_df['start'] < other_row['end2'])]['#id'].values
            # for each cpg in this region (if any), add the distance (count) to the dict
            for other_cpg in other_cpgs_in_region:
                one_cpg_dists_dict[other_cpg] = other_row['balanced'] # TODO: change this to weight when we have weights
        # add this cpg's distances to the output dict
        all_dists_dict[this_cpg_name] = one_cpg_dists_dict
        print(this_cpg_name, flush=True)
    # create dists_df from dict of dicts
    dists_df = pd.DataFrame.from_dict(all_dists_dict, orient='columns')
    return dists_df


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
            sns.violinplot(data=nearby_site_diffs_df, x='dist_bin', y='delta_mf', cut=0, inner="quartile")
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
                # check if site_to_test was mutated in any samples in a non measured site
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
        ax.set_xticklabels([str(distances[i]) + '*' if sig_status[i] == True else str(distances[i]) for i in range(len(distances))])
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

class MutationImpactCorr: 
    """
    Class to calculate information the expected impact of mutations on methyome of the set of samples in sample_mut_df.
    - This the expected impact relative to the same individual with no mutations. 
    """
    def __init__(self, sample_mut_df, linked_sites_diffs_dfs, linked_sites_pvals_dfs, methyl_age_df_t, dset):
        """
        @ sample_mut_df: df of all mutations
        @ linked_sites_diffs_dfs: list of dfs of differences in methylation between linked sites
        @ linked_sites_mwu_pvals_dfs: list of dfs of mwu pvals between linked sites
        @ pval_cutoff: pval cutoff for significance
        """
        # TODO: limit to dset     

        # TODO: remove X and Y chromosomes

        self.sample_mut_df = sample_mut_df
        self.linked_sites_diffs_dfs = linked_sites_diffs_dfs
        self.linked_sites_pvals_dfs = linked_sites_pvals_dfs
        self.methyl_age_df_t = methyl_age_df_t


    def calc_avg_mut_impact(self, pval_cutoff=0.05):
        """
        Calculates the total impact each of the tested mutations in result_dfs has on the methylation of the linked sites
        """
        # for now only consider the top percentile
        # copy so we don't change original
        linked_sites_diffs_df = self.linked_sites_diffs_dfs[0].copy(deep=True)
        linked_sites_pvals_df = self.linked_sites_pvals_dfs[0]
        # set each delta mf to 0 if the corresponding pval in linked_site_mwu_pval_df is above the cutoff
        linked_sites_diffs_df[linked_sites_pvals_df >= pval_cutoff] = 0
        # the sum of all diffs across linked sites of a mutation is the amount of methylation change that is due to the mutation
        # to preserve directionality, sum all positive values and all negative values in each row of linked_sites_diffs_df
        linked_sites_diffs_df['pos_eff'] = linked_sites_diffs_df[linked_sites_diffs_df > 0].sum(axis=1)
        linked_sites_diffs_df['neg_eff'] = linked_sites_diffs_df[linked_sites_diffs_df < 0].sum(axis=1)
        # get median effect and range that contains 95% of the data
        effect_range_df = linked_sites_diffs_df[['pos_eff', 'neg_eff']].quantile([0.025, 0.5, 0.975], axis=0)
        # get mean effect and standard error
        effect_range_df.loc['mean'] = linked_sites_diffs_df[['pos_eff', 'neg_eff']].mean(axis=0)
        effect_range_df.loc['standard_err'] = linked_sites_diffs_df[['pos_eff', 'neg_eff']].sem(axis=0)

        self.effect_range_df = effect_range_df

    def count_mutation_per_sample(self):
        """
        Count the number of mutations in CpGs each sample has
        """
        # TODO: change so actually queries if site is a CpG or not
        # subset to C>T mutations
        ct_mut_df = self.sample_mut_df[self.sample_mut_df['mutation'] == 'C>T']
        # count occurences of each sample
        sample_counts = ct_mut_df['sample'].value_counts()
        # create df of sample counts
        sample_mut_counts_df = pd.DataFrame({'sample': sample_counts.index, 'num_mutations': sample_counts.values})

        self.sample_mut_counts_df = sample_mut_counts_df

    def expected_mut_induced_mf_change(self, pval_cutoff=0.05):
        """
        Calculates the expected change in methylation for each sample due to mutations
        @ pval_cutoff: pval cutoff to use for determining if a mutation has an effect on a linked site
        @ returns: dataframe of the expected change in methylation for each sample due to mutations
        """
        # count the number of mutations in CpGs each sample has
        self.count_mutation_per_sample()
        # calculate the range of effects mutations had on linked sites
        self.calc_avg_mut_impact(pval_cutoff)
        # calculate the expected change in methylation for each sample due to mutations
        self.sample_mut_counts_df['pos_expected_change'] =  self.sample_mut_counts_df['num_mutations'] *  self.effect_range_df.loc['mean','pos_eff']
        self.sample_mut_counts_df['pos_standard_err'] =  self.sample_mut_counts_df['num_mutations'] *  self.effect_range_df.loc['standard_err','pos_eff']
        self.sample_mut_counts_df['neg_expected_change'] =  self.sample_mut_counts_df['num_mutations'] *  self.effect_range_df.loc['mean','neg_eff']
        self.sample_mut_counts_df['neg_standard_err'] =  self.sample_mut_counts_df['num_mutations'] *  self.effect_range_df.loc['standard_err','neg_eff']

        # get samples in self.methyl_age_df_t that are not in self.sample_mut_counts_df
        samples_with_no_mutations = self.methyl_age_df_t[~self.methyl_age_df_t.index.isin(self.sample_mut_counts_df['sample'])].index
        # add each of samples_with_no_mutations to sample_mut_counts_df 0 each column
        to_add_dict = {}
        for i in range(len(samples_with_no_mutations)):
            to_add_dict[i] = [samples_with_no_mutations[i],0, 0, 0, 0, 0]
        to_add_df = pd.DataFrame.from_dict(to_add_dict, orient='index', columns=['sample', 'num_mutations', 'pos_expected_change', 'pos_standard_err', 'neg_expected_change', 'neg_standard_err'])
        self.sample_mut_counts_df = self.sample_mut_counts_df.append(to_add_df)
        # reset index
        self.sample_mut_counts_df = self.sample_mut_counts_df.reset_index(drop=True)

        return self.sample_mut_counts_df


class methylChange:
    """
    Class that compares the observed and predicted methylome change between samples
    """
    def __init__(self, methyl_age_df_t, all_mut_df ,linked_sites_diffs_dfs, linked_sites_pvals_dfs, pval_cutoff=0.05, n_pairs=1000, age_window=5, cancer_type='all'):
        """
        Initialize a methylChange object
        @ methyl_age_df_t: a pandas dataframe of samples x methylation with age column
        @ all_mut_df: a pandas dataframe of all mutations
        @ n_pairs: number of pairs of samples to compare
        @ age_window: the age window to compare samples in [default: 5 years]
        @ cancer_type: the cancer type to compare samples from [default: all cancers]
        """
        self.methyl_age_df_t = methyl_age_df_t
        self.all_mut_df = all_mut_df
        self.linked_sites_diffs_dfs = linked_sites_diffs_dfs
        self.linked_sites_pvals_dfs = linked_sites_pvals_dfs
        self.pval_cutoff = pval_cutoff
        self.n_pairs = n_pairs
        self.age_window = age_window
        self.cancer_type = cancer_type

    def predict_methylome_diffs(self):
        """
        Get the predicted methylome diff for each sample given by number of mutations and their average effecct
        """
        # initialize a MutationImpact object
        mi = MutationImpact(self.all_mut_df, self.linked_sites_diffs_dfs, self.linked_sites_pvals_dfs, self.methyl_age_df_t)
        # get the expected methylome change for each sample based on mutations
        expected_mut_eff = mi.expected_mut_induced_mf_change(pval_cutoff = self.pval_cutoff)
        
        # add ages and age bins to expected_mut_eff by joining to self.methyl_age_df_t
        expected_mut_eff_w_age = expected_mut_eff.join(self.methyl_age_df_t[['age_at_index']], on='sample')

        self.expected_mut_eff_w_age = expected_mut_eff_w_age
        return expected_mut_eff_w_age

    def methylome_diff(self, sample1, sample2):
        """
        Get the positive and negative methylome difference between two samples
        """
        cpg_cols = self.methyl_age_df_t.columns[2:]
        # get methylome diff
        methylome_diff = self.methyl_age_df_t.loc[sample1, cpg_cols] - self.methyl_age_df_t.loc[sample2, cpg_cols]
        # get positive and negative methylome diff
        pos_methylome_diff = methylome_diff[methylome_diff > 0].sum()
        neg_methylome_diff = methylome_diff[methylome_diff < 0].sum()
        return pos_methylome_diff, neg_methylome_diff

    def compare_pairs(self):
        """
        For n_pairs of samples within age_window of eachother, compare their observed and predicted methylome differences
        """
        import random

        diff_dict = {}
        for i in range(self.n_pairs):
            # choose a random sample from self.methyl_age_df_t.index
            rand_sample1 = random.choice(self.methyl_age_df_t.index)
            # get the age of the sample at that index
            age1 = self.methyl_age_df_t.loc[rand_sample1]['age_at_index']
            # get all other samples within age_window of that age
            age_window_samples = self.methyl_age_df_t[(self.methyl_age_df_t['age_at_index'] >= age1 - self.age_window/2) & (self.methyl_age_df_t['age_at_index'] <= age1 + self.age_window/2)]
            # drop the sample we chose so we don't choose it again
            age_window_samples = age_window_samples.drop(rand_sample1)
            # choose a random sample from the age_window_samples to compare with
            rand_sample2 = random.choice(age_window_samples.index)
            # get second age
            age2 = self.methyl_age_df_t.loc[rand_sample2]['age_at_index']
            # get the methylome diff between the two samples
            pos_diff, neg_diff = self.methylome_diff(rand_sample1, rand_sample2)
            # add to results
            diff_dict[i] = {'sample1': rand_sample1, 'sample2': rand_sample2, 'age1': age1, 'age2': age2, 'obs_pos_diff': pos_diff, 'obs_neg_diff': neg_diff}
        # convert to dataframe
        diff_df = pd.DataFrame.from_dict(diff_dict, orient='index')
        
        # pre-compute prediction methylome diff for all samples
        pred_methylome_diffs = self.predict_methylome_diffs()
    
        # join diff_df and pred_methylome_diffs on sample1 and sample2
        pred_methylome_diffs = pred_methylome_diffs.set_index('sample')
        # join diff_df to pred_methylome_diffs on sample1
        diff_df = diff_df.join(pred_methylome_diffs, on='sample1')
        # join diff_df2 to pred_methylome_diffs on sample2
        diff_df = diff_df.join(pred_methylome_diffs, on='sample2', rsuffix='_2')

        self.diff_df = diff_df
        return diff_df

    def plot_methylome_diffs(self):
        diff_df_unstacked = self.diff_df[['obs_pos_diff', 'obs_neg_diff', 'pos_expected_change', 'neg_expected_change']].unstack().reset_index()
        diff_df_unstacked.columns = ['Difference type', 'Pair', 'Difference']

        # seaborn bar chart of differences
        _, axes = plt.subplots(figsize=(6,4), dpi=150)
        sns.barplot(x='Difference type', y='Difference', data=diff_df_unstacked, ax=axes)

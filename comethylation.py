import pandas as pd
from scipy import stats
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import ticker as mticker
import utils
import seaborn as sns
from statsmodels.stats.weightstats import ztest as ztest
import ray
from rich.progress import track
import random


PERCENTILES = np.flip(np.linspace(0, 1, 6))

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
    @ returns: Dataframe with rows being comparison sites, columns MAvgErr (average difference of mutated sample and non mutated sample at the comparison site that is the row) and ztest_pval (which says if the mutated sample was significantly different from the other samples at that site)
    """
    # subtract mut_sample_comparison_mfs_df from every row (sample) in same_age_samples_mf_df
    difference_at_comparison_sites = same_age_samples_mf_df.subtract(mut_sample_comparison_mfs_df.iloc[0])
    # switch the sign of every entry in difference_at_comparison_sites so that delta_mf is positive if the mutated sample has higher methylation than the other samples and vica versa
    difference_at_comparison_sites = difference_at_comparison_sites.mul(-1)
    # take mean to get mean average difference at each site
    mean_diff_each_comparison_site = pd.DataFrame(difference_at_comparison_sites.mean(axis = 0), columns=['delta_mf'])
    # add mut_sample_comparison_mfs_df as last row of same_age_samples_mf_df
    all_samples_at_comparison_sites = same_age_samples_mf_df.append(mut_sample_comparison_mfs_df)
    # also calculate a z score pvalue for the mut_sample site being different from the non-mutated samples
    mean_diff_each_comparison_site['zscore'] = all_samples_at_comparison_sites.apply(lambda x: stats.zscore(x, nan_policy='omit')[-1], axis=0)
    # get 2-sided pvalues
    mean_diff_each_comparison_site['ztest_pval'] = stats.norm.sf(abs(mean_diff_each_comparison_site['zscore']))*2
    
    return mean_diff_each_comparison_site[['delta_mf','ztest_pval']]

#@ray.remote
def effect_one_mutation(mut_linkage_df,
                            linkage_type,
                            all_methyl_age_df_t,
                            mut_in_measured_cpg_w_methyl_age_df,
                            illumina_cpg_locs_df,
                            percentile,
                            num_linked_sites,
                            age_bin_size,
                            mut_cpg):
    # get chrom of this site and which sample had this cpg mutated
    mut_cpg_chr = illumina_cpg_locs_df[illumina_cpg_locs_df['#id'] == mut_cpg]['chr'].iloc[0]
    mut_sample = mut_in_measured_cpg_w_methyl_age_df[mut_in_measured_cpg_w_methyl_age_df['#id'] == mut_cpg]
    # limit comparison sites to cpgs on same chrom that are in all_methyl_age_df_t
    same_chr_cpgs = illumina_cpg_locs_df[(illumina_cpg_locs_df['chr'] == mut_cpg_chr) & (illumina_cpg_locs_df['#id'].isin(all_methyl_age_df_t.columns))]['#id'].to_list()
    same_chr_linkage_df = mut_linkage_df.loc[mut_linkage_df.index.isin(same_chr_cpgs)]
    # drop mut_cpg from same_chr_linkage_df so it is not selected as a comparison site
    same_chr_linkage_df.drop(mut_cpg, axis=0, inplace=True)
    # get comparison sites based on method
    if linkage_type == 'methylation_corr':
        # get the positively correlated sites that are on the same chromosome as the mutated CpG
        linked_sites = select_corr_sites(mut_cpg, same_chr_linkage_df, num_linked_sites, percentile)
        # farthest sites are 0th percentile
        nonlinked_sites = select_corr_sites(mut_cpg, same_chr_linkage_df, num_linked_sites, 0)#np.abs(same_chr_linkage_df[mut_cpg]).nsmallest(num_linked_sites).index.to_list()
    elif linkage_type == '2d_distance':
        # get the closest sites that are on the same chromosome as the mutated CpG
        linked_sites = select_closest_sites_2d_distance(mut_cpg, same_chr_linkage_df, num_linked_sites, percentile)
        # farthest sites are 100th percentile
        nonlinked_sites = select_closest_sites_2d_distance(mut_cpg, same_chr_linkage_df, num_linked_sites, 1)
    elif linkage_type == 'hic_distance':
        # get the closest sites that are on the same chromosome as the mutated CpG
        linked_sites = select_closest_sites_hic_distance(mut_cpg, same_chr_linkage_df, num_linked_sites, percentile)
        # farthest sites are 0th percentile
        nonlinked_sites = select_closest_sites_hic_distance(mut_cpg, same_chr_linkage_df, num_linked_sites, 0)
    else:
        raise ValueError("linkage_type must be either 'methylation_corr' or '2d_distance' or 'hic_distance'")

    # get this sample's age and name
    this_age = mut_sample['age_at_index'].to_list()[0]
    this_sample_name = mut_sample['case_submitter_id']

    # get the mf all other samples of within age_bin_size/2 years of age on either side
    same_age_samples_mf_df = all_methyl_age_df_t[np.abs(all_methyl_age_df_t['age_at_index'] - this_age) <= age_bin_size/2]
    # get this mutated sample's MF at comparison sites
    mut_sample_linked_mfs_df = all_methyl_age_df_t.loc[this_sample_name, linked_sites] 

    # measure the change in methylation between linked sites in the mutated sample and in other non-mutated samples of the same age
    # linked_result_df is a dataframe with columns: MabsErr, MavgErr and rows: same age samples and entries the respective metric measuring distance between that sample and mutated sample across all linked sites
    linked_diff = compare_sites(same_age_samples_mf_df.drop(index = this_sample_name)[linked_sites], mut_sample_linked_mfs_df)

    # do same comparison but seeing if unlinked sites also changed same amount
    mut_sample_nonlinked_mfs_df = all_methyl_age_df_t.loc[this_sample_name, nonlinked_sites]
    # nonlinked_result_df is a dataframe with columns: MabsErr, MavgErr and rows: same age samples and entries the respective metric measuring distance between that sample and mutated sample across all nonlinked sites
    nonlinked_diff = compare_sites(same_age_samples_mf_df.drop(index = this_sample_name)[nonlinked_sites], mut_sample_nonlinked_mfs_df)

    # compare mut_result_df to nonlinked_result_df to see if there are less significant differences in less linked CpGs
    this_mut_results = test_linked_vs_nonlinked(linked_diff[['delta_mf']], nonlinked_diff[['delta_mf']])

    # return lists to add to dictionaries
    return mut_cpg, this_mut_results, linked_sites, linked_diff['delta_mf'].to_list(), linked_diff['ztest_pval'].to_list(), nonlinked_sites, nonlinked_diff['delta_mf'].to_list(), nonlinked_diff['ztest_pval'].to_list()

def measure_mut_eff_on_module(mut_linkage_df,
                                linkage_type,
                                all_methyl_age_df_t, 
                                mut_in_measured_cpg_w_methyl_age_df,
                                illumina_cpg_locs_df,
                                percentile,
                                num_linked_sites,
                                age_bin_size,
                                parallel=0):
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
    linked_sites_names_dict = {}
    linked_site_diffs_dict = {}
    linked_site_z_pvals_dict = {}
    nonlinked_sites_names_dict = {}
    nonlinked_site_diffs_dict = {}
    nonlinked_site_z_pvals_dict = {}
    # run in parallel or not
    if parallel != 0:
        each_perc_result_lists = [
            effect_one_mutation.remote(mut_linkage_df, linkage_type, all_methyl_age_df_t, mut_in_measured_cpg_w_methyl_age_df, illumina_cpg_locs_df, percentile, num_linked_sites, age_bin_size, mut_cpg) for mut_cpg in mut_linkage_df.columns
        ]
        result_lists = ray.get(each_perc_result_lists)
    else:
        each_perc_result_lists = []
        for mut_cpg in track(mut_linkage_df.columns, description="Analyzing each mutation"):
            each_perc_result_lists.append(effect_one_mutation(mut_linkage_df, linkage_type, all_methyl_age_df_t, mut_in_measured_cpg_w_methyl_age_df, illumina_cpg_locs_df, percentile, num_linked_sites, age_bin_size, mut_cpg))
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
    # create linked sites df from linked_sites_dict with keys as index
    linked_sites_names_df = pd.DataFrame.from_dict(linked_sites_names_dict, orient='index')
    linked_sites_diffs_df = pd.DataFrame.from_dict(linked_site_diffs_dict, orient='index')
    linked_sites_z_pvals_df = pd.DataFrame.from_dict(linked_site_z_pvals_dict, orient='index')
    nonlinked_sites_names_df = pd.DataFrame.from_dict(nonlinked_sites_names_dict, orient='index')
    nonlinked_sites_diffs_df = pd.DataFrame.from_dict(nonlinked_site_diffs_dict, orient='index')
    nonlinked_sites_z_pvals_df = pd.DataFrame.from_dict(nonlinked_site_z_pvals_dict, orient='index')
    return result_df, linked_sites_names_df, linked_sites_diffs_df, linked_sites_z_pvals_df, nonlinked_sites_names_df, nonlinked_sites_diffs_df, nonlinked_sites_z_pvals_df

def mutation_eff_varying_linkage(mut_linkage_df,
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
        result_df, linked_sites_names_df, linked_sites_diffs_df, linked_sites_z_pvals_df, nonlinked_sites_names_df, nonlinked_sites_diffs_df, nonlinked_sites_z_pvals_df = measure_mut_eff_on_module(mut_linkage_df, linkage_type, all_methyl_age_df_t, mut_in_measured_cpg_w_methyl_age_df, illumina_cpg_locs_df, percentile, num_linked_sites, age_bin_size)
        result_dfs.append(result_df)
        linked_sites_names_dfs.append(linked_sites_names_df)
        linked_sites_diffs_dfs.append(linked_sites_diffs_df)
        linked_sites_z_pvals_dfs.append(linked_sites_z_pvals_df)
        nonlinked_sites_names_dfs.append(nonlinked_sites_names_df)
        nonlinked_sites_diffs_dfs.append(nonlinked_sites_diffs_df)
        nonlinked_sites_z_pvals_dfs.append(nonlinked_sites_z_pvals_df)
    return result_dfs, linked_sites_names_dfs, linked_sites_diffs_dfs, linked_sites_z_pvals_dfs, nonlinked_sites_names_dfs, nonlinked_sites_diffs_dfs, nonlinked_sites_z_pvals_dfs

def read_correlations(corr_fns, illumina_cpg_locs_df):
    # switch order
    #corr_fns = [corrs_fns[1]] + [corrs_fns[0]] + corrs_fns[2:]
    corr_dfs = []
    for corr_fn in corr_fns:
        this_corr_df = pd.read_parquet(corr_fn)
        corr_dfs.append(this_corr_df)
    # concat and remove duplicate column names from minor overlap
    corr_df = pd.concat(corr_dfs, axis=1)
    corr_df = corr_df.loc[:,~corr_df.columns.duplicated()].copy()
    # remove cpgs on X and Y chromosomes
    corr_df = corr_df[corr_df.columns[~corr_df.columns.isin(illumina_cpg_locs_df[(illumina_cpg_locs_df['chr'] == 'X') | (illumina_cpg_locs_df['chr'] == 'Y')]['#id'])]]
    return corr_df

def add_ages_to_methylation(ct_mut_in_measured_cpg_w_methyl_df, all_meta_df, all_methyl_df_t):
    to_join_ct_mut_in_measured_cpg_w_methyl_df = ct_mut_in_measured_cpg_w_methyl_df.rename(columns={'sample':'case_submitter_id'})
    mut_in_measured_cpg_w_methyl_age_df =  to_join_ct_mut_in_measured_cpg_w_methyl_df.join(all_meta_df, on =['case_submitter_id'], rsuffix='_r',how='inner')
    # join ages with methylation
    all_methyl_age_df_t = all_meta_df.join(all_methyl_df_t, on =['sample'], rsuffix='_r',how='inner')
    return mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t
    

def plot_eff_line(result_dfs, mut_in_measured_cpg_w_methyl_age_df, mut_linkage_df, sig_only, linkage_type):
    """
    Plot the effect of mutation on methylation at mutated sites
    @ result_dfs: list of dataframes, each containing the results for a different percentile
    @ mut_in_measured_cpg_w_methyl_age_df: dataframe containing the methylation and mutation data for the all sites
    @ mut_linkage_df: dataframe containing the linkage data for the tested sites
    @ sig_only: boolean, whether to plot only significant results
    @ linkage_type: string, the type of linkage used
    """
    # reverse order of result_dfs if 2d_distance
    if linkage_type == '2d_distance':
        result_dfs = result_dfs[::-1]
    # limited to significant sites
    if sig_only:
        sig_result_effs_dict = utils.test_sig(result_dfs)
    # make single line chart for top 100
    fig, axes = plt.subplots()
    linked_heights = []
    nonlinked_heights = []
    # across each percentile result
    for i in range(len(PERCENTILES)):
        if sig_only:
            # get the MAvgErr of linked sites
            linked_heights.append(sig_result_effs_dict['m_linked_mean_avg_err'][i])
            # and non-linked sites
            nonlinked_heights.append(sig_result_effs_dict['m_non_linked_mean_avg_err'][i])
        else:
            linked_heights.append(np.abs(result_dfs[i]['linked_mean_avg_err']).mean())
            nonlinked_heights.append(np.abs(result_dfs[i]['non_linked_mean_avg_err']).mean())

    # get the average MF change of the mutated sites
    avg_mut_mf_change = np.abs(mut_in_measured_cpg_w_methyl_age_df[mut_in_measured_cpg_w_methyl_age_df['#id'].isin(mut_linkage_df.columns)]['difference']).mean()
    # plot these values as a lines across each PERCENTILE iteration
    x_pos = np.arange(0,1.5*len(linked_heights),1.5)  
    axes.plot(x_pos, linked_heights, color='steelblue', marker='o', label='Linked')
    axes.plot(x_pos, nonlinked_heights, color='skyblue', marker='o', label='Non-linked')
    axes.plot(x_pos, [avg_mut_mf_change]*len(x_pos),  color='goldenrod',marker='o', label='Mutated sites')
    plt.legend()
    ticks_pos = np.arange(0,1.5*(len(linked_heights)),1.5)    
    plt.xticks(ticks_pos, [str(i)[:4] for i in PERCENTILES])
    if sig_only:
        axes.set_ylabel(r"Mean $\Delta$MF across mutations with significant effects")
    else:
        axes.set_ylabel(r"Mean $\Delta$MF of each tested mutation (n=100)")
    if linkage_type == '2d_distance':
        plt.xlabel("Distance-based linkage percentile")
    elif linkage_type == 'methylation_corr':
        plt.xlabel("Correlation-based linkage percentile")
    elif linkage_type == 'hic_distance':
        plt.xlabel("Hi-C distance-based linkage percentile")

def plot_sig_bars(result_dfs, linkage_type, test='p_wilcoxon'):
    """
    Plot the sig bars for mean avg err
    @ result_dfs: list of dataframes, each containing the results for a different percentile
    @ linkage_type: string, the type of linkage used
    @ test: string, the test used to determine significance (p_wilcoxon or p_barlett)
    """
    # reverse order of result_dfs if 2d_distance
    if linkage_type == '2d_distance':
        result_dfs = result_dfs[::-1]
        
    sig_result_effs_dict = utils.test_sig(result_dfs)
    # plot abs error
    r = [i for i in range(len(PERCENTILES))]
    raw_data = {'sig': sig_result_effs_dict[test], 'non_sig': [len(result_dfs[0]) - i for i in sig_result_effs_dict[test]] }
    plot_df = pd.DataFrame(raw_data)
    totals = [i+j for i,j in zip(plot_df['sig'], plot_df['non_sig'])]
    sig = [i / j * 100 for i,j in zip(plot_df['sig'], totals)]
    non_sig = [i / j * 100 for i,j in zip(plot_df['non_sig'], totals)]
    # plot
    barWidth = 0.85
    names = [str(int(round(i, 1)*100))+'%' for i in  PERCENTILES]
    # Create blue Bars
    plt.bar(r, sig, color='steelblue', edgecolor='white', width=barWidth, label="Significant")
    # Create red Bars
    plt.bar(r, non_sig, bottom=sig, color='maroon', edgecolor='white', width=barWidth, label="Not significant")
    # Custom x axis
    plt.xticks(r, names)
    plt.ylabel("Percent of mutations with significant effect")
    plt.xlabel("Linkage percentile")
    plt.legend()

def stack_and_merge(diffs_df, pvals_df):
    """
    Take one diffs and one pvals df, stack them and merge
    """
    # stack the dfs
    diffs_df = diffs_df.stack().reset_index()
    pvals_df = pvals_df.stack().reset_index()
    # rename the columns
    diffs_df.columns = ['comparison_site', 'mut_site', 'delta_mf']
    pvals_df.columns = ['comparison_site', 'mut_site', 'pval']
    # merge
    merged_df = pd.merge(diffs_df, pvals_df, on=['comparison_site', 'mut_site'])
    return merged_df

def plot_eff_violin(result_dfs, mut_in_measured_cpg_w_methyl_age_df, mut_linkage_df, linkage_type, linked_sites_pvals_dfs, linked_sites_diffs_dfs, nonlinked_sites_pvals_dfs, nonlinked_sites_diffs_dfs,  sig_thresh=1):
    """
    Plot the violin plots for mean avg err
    @ result_dfs: list of dataframes, each containing the results for a different percentile
    @ mut_in_measured_cpg_w_methyl_age_df: dataframe containing the methylation and mutation data for the all sites
    @ mut_linkage_df: dataframe containing the linkage data for the tested sites
    @ linkage_type: string, the type of linkage used
    """
    # reverse order of result_dfs if 2d_distance
    if linkage_type == '2d_distance':
        result_dfs = result_dfs[::-1]
        linked_sites_diffs_dfs = linked_sites_diffs_dfs[::-1]
        linked_sites_pvals_dfs = linked_sites_pvals_dfs[::-1]

    # make a df with columns: linkage percentile, delta_mf, pval, and linkage_status
    to_plot_dfs = []
    for i in range(len(linked_sites_diffs_dfs)):
        linked_diffs_pvals = stack_and_merge(linked_sites_diffs_dfs[i], linked_sites_pvals_dfs[i])
        # add a column for the linkage percentile
        linked_diffs_pvals['Linkage percentile'] = int(round(PERCENTILES[i], 1)*100)
        # also stack and merge the nonlinked_sites_diffs_dfs and nonlinked_sites_pvals_dfs
        nonlinked_diffs_pvals = stack_and_merge(nonlinked_sites_diffs_dfs[i], nonlinked_sites_pvals_dfs[i])
        nonlinked_diffs_pvals['Linkage percentile'] = int(round(PERCENTILES[i], 1)*100)
        # concat these two dfs
        to_plot_df = pd.concat([linked_diffs_pvals, nonlinked_diffs_pvals], axis=0)
        # add a column for the linkage status
        to_plot_df['Linkage status'] = ["Linked CpGs" for _ in range(len(linked_diffs_pvals))] + ["Non-linked CpGs" for _ in range(len(nonlinked_diffs_pvals))]
        to_plot_dfs.append(to_plot_df)
    # concat all the dfs
    to_plot_df = pd.concat(to_plot_dfs, axis=0)
    # subset to only the significant sites
    to_plot_df = to_plot_df[to_plot_df['pval'] < sig_thresh]

    # violin plot of the mean avg err for each linkage percentile
    _, axes = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 6]}, sharey=True, constrained_layout=True)
    my_pal = {"Linked CpGs": "steelblue", "Non-linked CpGs": "skyblue"}
    p = sns.violinplot(x="Linkage percentile", y="delta_mf", hue="Linkage status", data=to_plot_df, scale="count", palette=my_pal, ax=axes[1], split=True, cut=0, linewidth=2)
    p.set_ylabel(r"$\Delta$MF")
    axes[1].invert_xaxis()
    # add a % sign after each x tick label
    axes[1].set_xticklabels(reversed([str(int(round(i, 1)*100))+'%' for i in PERCENTILES]))
    # move legend to the bottom right
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[::-1], labels[::-1], loc='lower right')

    # violin plot of the mean avg err for all mutated sites
    mut_mf_change = mut_in_measured_cpg_w_methyl_age_df[mut_in_measured_cpg_w_methyl_age_df['#id'].isin(mut_linkage_df.columns)]['difference'].reset_index(drop=True)
    # violin plot but stop at end of data points
    p2 = sns.violinplot(data=mut_mf_change, ax=axes[0], color='maroon', cut=0, linewidth=2)
    # add legend to violin plot
    patch = mpatches.Patch(color='maroon', label='Mutated CpGs')
    axes[0].legend(handles=[patch])
    # remove x ticks and x tick labels
    p2.set_xticks([])
    p2.set_xticklabels([])
    p2.set_ylabel(r"$\Delta$MF")
    # show y axis labels
    axes[1].tick_params(axis='y', labelleft=True)
    return 


def plot_linked_site_distances(linked_sites_dfs, distances_df, log=True):
    """
    Plot the distances between linked sites for each linkage percentile
    @ linked_sites_dfs: list of dataframes, each containing the linked sites for a different percentile for every mutated site
    @ distances_df: dataframe containing the distances between each pair of sites
    @ returns: a df with columns percentile, mutated_cpg_site, linked_cpg_site, distance
    """
    # for each percentile, get the distances between linked sites
    linked_distances = {}
    for i, percent in zip(range(len(linked_sites_dfs)), PERCENTILES):
        # create output dict
        linked_distances[percent] = {}
        # iterate across each mutated site (aka row) of linked_df
        linked_df = linked_sites_dfs[i]
        for mut_site, row in linked_df.iterrows():
            # get the distances between this mutated site and all linked sites
            mut_linked_distances = distances_df.loc[mut_site, row.to_list()].values
            # add to linked_distances
            linked_distances[percent][mut_site] = mut_linked_distances

    # want a df with columns: percentile, mutated_cpg_site, linked_cpg_site, distance
    # convert doubly nested dict to df
    to_plot_dfs = []
    for percent, mut_site_dict in linked_distances.items():
        percent_df = pd.DataFrame.from_dict(mut_site_dict, orient='index')
        # unroll the df
        percent_df_stacked = percent_df.unstack().reset_index()
        # rename columns
        percent_df_stacked.columns = ['linked_site_num', 'mutated_site', 'distance']
        # add a column for the percentile
        percent_df_stacked['percentile'] = np.round(percent,1)*100
        to_plot_dfs.append(percent_df_stacked)
    to_plot_df = pd.concat(to_plot_dfs)
    # plot the distances as a violion plot for each percentile
    fig, axes = plt.subplots(figsize=(10, 4), dpi=175)
    # log scale y axis
    to_plot_df['log_distance'] = np.log10(to_plot_df['distance'])
    # change to megabases
    to_plot_df['mbp_distance'] = to_plot_df['distance']/1000000
    # plot
    if log:
        sns.violinplot(data=to_plot_df, ax=axes, cut=0, x='percentile', y='log_distance', color='steelblue')
    else:
        sns.violinplot(data=to_plot_df, ax=axes, cut=0, x='percentile', y='mbp_distance', color='steelblue')
    axes.invert_xaxis()
    axes.set_xlabel("Linkage percentile")
    if log:
        axes.set_ylabel("Distance of linked site from mutated site (bp)")
    else:
        axes.set_ylabel("Distance of linked site from mutated site (Mbp)")
    axes.set_xticklabels(reversed([str(int(round(i, 1)*100))+'%' for i in PERCENTILES]))
    # make nice logscale y axis ticks
    if log:
        axes.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        ymin, ymax = axes.get_ylim()
        tick_range = np.arange(np.floor(ymin), ymax)
        axes.yaxis.set_ticks(tick_range)
        axes.yaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(10 ** p, 10 ** (p + 1), 10)], minor=True)
    # angle x ticks
    return to_plot_df

def plot_distance_vs_eff(linked_distances_df, linked_sites_diffs_df, log=True):
    """
    Plot the distance between linked sites vs the delta_mf
    @ linked_distances_df: df with columns percentile, mutated_cpg_site, linked_cpg_site, distance (from running plot_linked_site_distances)
    @ linked_sites_diffs_dfs: dataframe,  containing the delta_mf for a certain percentile for every mutated site
    """
    # only do for top percentile
    diffs_df = linked_sites_diffs_df
    distances_df = linked_distances_df[linked_distances_df['percentile'] == 100]
    # unroll diffs_df
    diffs_df_stacked = diffs_df.unstack().reset_index()
    diffs_df_stacked.columns = ['linked_site_num', 'mutated_site', 'delta_mf']
    diffs_df_stacked['linked_site_num'] = diffs_df_stacked['linked_site_num'].astype(int)
    # merge the two dfs on the mutated site and linked site num
    merged_df = pd.merge(distances_df, diffs_df_stacked, on=['mutated_site', 'linked_site_num'])
    # plt hexbin plot
    fig, axes = plt.subplots(figsize=(7, 5), dpi=100)
    if log:
        axes.hexbin(merged_df['log_distance'], merged_df['delta_mf'], gridsize=50, cmap='Reds', bins='log')
        axes.set_xlabel("Distance between linked site and mutated site (bp)")
    else:
        axes.hexbin(merged_df['mbp_distance'], merged_df['delta_mf'], gridsize=50, cmap='Reds', bins='log')
        axes.set_xlabel("Distance between linked site and mutated site (Mbp)")
    axes.set_ylabel(r"$\Delta$MF")
    # add colorbar
    cbar = fig.colorbar(axes.collections[0], ax=axes)
    cbar.set_label("Number of sites")
    return

def site_characteristics(comparison_sites_df, all_methyl_age_df_t, mut_in_measured_cpg_w_methyl_age_df):
    """
    Return the average methylation fraction of each site in sites_df
    @ sites_df: dataframe with rows corresponding the linked or nonlinks sites for a given mutated site (index value), columns the linked site index, and entries the site name
    """
    mean_methyls = {}
    for mut_site, comparison_sites in comparison_sites_df.iterrows():
        # get the mean methylation across all samples at the comparison sites
        mean_comp_methyl = all_methyl_age_df_t.loc[:, comparison_sites].mean(axis=0, skipna=True)
        mean_methyls[mut_site] = mean_comp_methyl.values
    mean_methyls_df = pd.DataFrame.from_dict(mean_methyls, orient='index')
    # set mut_in_measured_cpg_w_methyl_age_df index to be #id
    mut_in_measured_cpg_w_methyl_age_df = mut_in_measured_cpg_w_methyl_age_df.set_index('#id')
    # merge the two dfs on index
    mean_methyls_df = pd.merge(mut_in_measured_cpg_w_methyl_age_df['avg_methyl_frac'], mean_methyls_df, left_index=True, right_index=True)

    return mean_methyls_df
    
    

def plot_hyper_vs_hypo_linked_site_eff(linked_sites_diffs_dfs, linked_sites_names_dfs, linked_sites_pvals_dfs, mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t, sig_thresh=0.05):
    """
    Plot the delta_mf for all combinations of hyper and hypo methylated mutated and linked sites
    @ linked_sites_diffs_dfs: list of dataframes, containing the delta_mf for a certain percentile for every mutated site
    @ linked_sites_names_dfs: list of dataframes, containing the names of the linked sites for a certain percentile for every mutated site
    @ linked_sites_pvals_dfs: list of dataframes, containing the pvals of the linked sites for a certain percentile for every mutated site
    @ mut_in_measured_cpg_w_methyl_age_df: dataframe, containing the age and average methylation of every mutated site
    @ all_methyl_age_df_t: dataframe, containing the age and average methylation of every cpg site
    @ age_bin_size: int, size of the age bins to use
    """
    # only do for top percentile
    diffs_df = linked_sites_diffs_dfs[0]
    linked_names_df = linked_sites_names_dfs[0]
    linked_pvals_df = linked_sites_pvals_dfs[0]
    # change indexes
    mut_in_measured_cpg_w_methyl_age_df = mut_in_measured_cpg_w_methyl_age_df.set_index('#id', drop=True)
    all_methyl_age_df_t = all_methyl_age_df_t.reset_index(drop=True)
    all_methyl_age_df_t['age_at_index'] = all_methyl_age_df_t['age_at_index'].astype(int)

    # get the methylation status of linked sites
    hyper_sites = {}
    hypo_sites = {}
    # for each row in linked_df
    for mut_site, linked_sites in linked_names_df.iterrows():
        # get the mean methylation across all samples of the linked sites
        mean_linked_methyl = all_methyl_age_df_t.loc[:, linked_sites].mean(axis=0)
        # add the name of each site with mean_linked_methyl > .5 to hyper_methylated
        hyper_sites[mut_site] = (mean_linked_methyl > .5).values
        # add the name of each site with mean_linked_methyl < .5 to hypo_methylated
        hypo_sites[mut_site] = (mean_linked_methyl <= .5).values
    hyper_sites_df = pd.DataFrame.from_dict(hyper_sites, orient='index')
    hypo_sites_df = pd.DataFrame.from_dict(hypo_sites, orient='index')

    # for each row in hyper_sites_df, get the delta_mf for each hypermethylated linked site
    hyper_df = hyper_sites_df.apply(lambda x: diffs_df.loc[x.name, x.values], axis=1)
    # same for hypo
    hypo_df = hypo_sites_df.apply(lambda x: diffs_df.loc[x.name, x.values], axis=1)

    # unroll the dfs and concat, adding a column for hyper or hypo
    hyper_df_stacked = hyper_df.unstack().reset_index().dropna(how='any', axis=0)
    hyper_df_stacked.columns = ['linked_site_num', 'mutated_site', 'delta_mf']
    hyper_df_stacked['linked_site_num'] = hyper_df_stacked['linked_site_num'].astype(int)
    hyper_df_stacked['Linked site status'] = 'hyper'
    hypo_df_stacked = hypo_df.unstack().reset_index().dropna(how='any', axis=0)
    hypo_df_stacked.columns = ['linked_site_num', 'mutated_site', 'delta_mf']
    hypo_df_stacked['linked_site_num'] = hypo_df_stacked['linked_site_num'].astype(int)
    hypo_df_stacked['Linked site status'] = 'hypo'
    delta_mf_by_methy_status = pd.concat([hyper_df_stacked, hypo_df_stacked])

    # get methylation status of the mutated sites
    # subset mut_in_measured_cpg_w_methyl_age_df to only tested sites
    mut_in_measured_cpg_w_methyl_age_df = mut_in_measured_cpg_w_methyl_age_df[mut_in_measured_cpg_w_methyl_age_df.index.isin(diffs_df.index)]
    # get the hyper and hypo sites
    hyper_mut_sites = mut_in_measured_cpg_w_methyl_age_df[mut_in_measured_cpg_w_methyl_age_df['avg_methyl_frac'] > .5].index
    # add a column to delta_mf_by_methy_status for hyper or hypo mutated site
    delta_mf_by_methy_status['mutated_site_status'] = delta_mf_by_methy_status['mutated_site'].apply(lambda x: 'Hyper-' if x in hyper_mut_sites else 'Hypo-')
    # combine Linked site status and mutated site status columns
    delta_mf_by_methy_status['Methylation status'] = delta_mf_by_methy_status['mutated_site_status'] + delta_mf_by_methy_status['Linked site status']

    # subset to only linked sites that pass the pval threshold
    pvals_df_stacked = linked_pvals_df.unstack().reset_index().dropna(how='any', axis=0)
    pvals_df_stacked.columns = ['linked_site_num', 'mutated_site', 'pval']
    # make column dtypes match
    delta_mf_by_methy_status['mutated_site'] = delta_mf_by_methy_status['mutated_site'].astype(str)
    delta_mf_by_methy_status['linked_site_num'] = delta_mf_by_methy_status['linked_site_num'].astype(int)
    pvals_df_stacked['mutated_site'] = pvals_df_stacked['mutated_site'].astype(str)
    pvals_df_stacked['linked_site_num'] = pvals_df_stacked['linked_site_num'].astype(int)
    # join the pvals df to the delta_mf_by_methy_status on mutated site and linked site number
    delta_mf_by_methy_status = delta_mf_by_methy_status.merge(pvals_df_stacked, on=['mutated_site', 'linked_site_num'])
    # subset to only rows where pval <= sig_thresh
    delta_mf_by_methy_status = delta_mf_by_methy_status[delta_mf_by_methy_status['pval'] <= sig_thresh]

    # make axes with first axes in 3 to 1 ration to second
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100, gridspec_kw={'width_ratios': [1.25, 1]})
    # plot seaborn violin plot
    _ = sns.violinplot(x='Methylation status', y='delta_mf', data=delta_mf_by_methy_status, palette = ['#1434A4', 'steelblue',  'white', '#2060A7'], ax=axes[0], order = ['Hyper-hyper', 'Hyper-hypo', 'Hypo-hypo', 'Hypo-hyper'], cut=0)
    # label axes
    axes[0].set_ylabel(r"$\Delta$MF")
    # add line at x = 0
    axes[0].axhline(0, color='black', linestyle='--')
    # remove x label
    axes[0].set_xlabel('')
    # barplot of proportion of hyper/hypo sites that get mutated with black line around bars
    hyper_hyper_bar_height = len(delta_mf_by_methy_status[delta_mf_by_methy_status['Methylation status'] == 'Hyper-hyper'])
    hyper_hypo_bar_height = len(delta_mf_by_methy_status[delta_mf_by_methy_status['Methylation status'] == 'Hyper-hypo'])
    hypo_hyper_bar_height = len(delta_mf_by_methy_status[delta_mf_by_methy_status['Methylation status'] == 'Hypo-hyper'])
    hypo_hypo_bar_height = len(delta_mf_by_methy_status[delta_mf_by_methy_status['Methylation status'] == 'Hypo-hypo'])
    axes[1].bar([0, 1, 2, 3], [hyper_hyper_bar_height, hyper_hypo_bar_height, hypo_hypo_bar_height, hypo_hyper_bar_height ], color=['#1434A4', 'steelblue',  'white', '#2060A7'], edgecolor='black')
    # set xticks labels
    axes[1].set_xticks([0, 1, 2, 3])
    axes[1].set_xticklabels(['Hyper-hyper', 'hyper-hypo', 'hypo-hypo', 'hypo-hyper'])
    # label axes
    axes[1].set_ylabel('Count of significant\nmutated site-linked site combinations')
    # super x label
    fig.text(0.5, 0.01, 'Mutated site-linked site methylation status', ha='center', va='center')
    


def plot_hyper_vs_hypo_mutated_site_eff(linked_sites_diffs_dfs, mut_in_measured_cpg_w_methyl_age_df, linked_distances_df, log=True):
    """
    Plot the delta_mf for hypermethylated and hypomethylated sites seperately
    @ linked_sites_diffs_dfs: list of dataframes, each containing the delta_mf for a different percentile for every mutated site
    @ mut_in_measured_cpg_w_methyl_age_df: methylation dataframe with age information for mutated sites
    @ linked_distances_df: df with columns percentile, mutated_cpg_site, linked_cpg_site, distance (from running plot_linked_site_distances)
    """
    # only do for top percentile
    diffs_df = linked_sites_diffs_dfs[0]
    # subset mut_in_measured_cpg_w_methyl_age_df to only tested sites
    mut_in_measured_cpg_w_methyl_age_df = mut_in_measured_cpg_w_methyl_age_df[mut_in_measured_cpg_w_methyl_age_df['#id'].isin(diffs_df.index)]
    # get the hyper and hypo sites
    hyper_sites = mut_in_measured_cpg_w_methyl_age_df[mut_in_measured_cpg_w_methyl_age_df['avg_methyl_frac'] > .5]['#id']
    hypo_sites = mut_in_measured_cpg_w_methyl_age_df[mut_in_measured_cpg_w_methyl_age_df['avg_methyl_frac'] <= .5]['#id']
    # get the hyper and hypo delta_mfs
    hyper_df = diffs_df.loc[hyper_sites]
    hypo_df = diffs_df.loc[hypo_sites]
    # unroll the dfs and concat, adding a column for hyper or hypo
    hyper_df_stacked = hyper_df.unstack().reset_index()
    hyper_df_stacked.columns = ['linked_site_num', 'mutated_site', 'delta_mf']
    hyper_df_stacked['linked_site_num'] = hyper_df_stacked['linked_site_num'].astype(int)
    hyper_df_stacked['Status'] = 'Hypermethylated'
    hypo_df_stacked = hypo_df.unstack().reset_index()
    hypo_df_stacked.columns = ['linked_site_num', 'mutated_site', 'delta_mf']
    hypo_df_stacked['linked_site_num'] = hypo_df_stacked['linked_site_num'].astype(int)
    hypo_df_stacked['Status'] = 'Hypomethylated'
    merged_df = pd.concat([hyper_df_stacked, hypo_df_stacked])
    # plot seaborn violin plot
    fig, axes = plt.subplots(1,2, figsize=(11, 4), dpi=100)
    merged_df['all'] = ""
    sns.violinplot(x='all', y='delta_mf', hue='Status', data=merged_df, ax=axes[0], palette = ['#2060A7', 'white'], split=True, cut=0)
    axes[0].set_ylabel(r"$\Delta$MF")
    axes[0].set_xlabel("Average methylation status of mutated site\nin non-mutated same age samples")
    # add line at x = 0
    axes[0].axhline(0, color='black', linestyle='--')
    # barplot of proportion of hyper/hypo sites that get mutated with black line around bars
    hyper_bar_height = len(hyper_df_stacked)
    hypo_bar_height = len(hypo_df_stacked)
    axes[1].bar([0, 1], [hyper_bar_height, hypo_bar_height], color=['#2060A7', 'white'], edgecolor='black')
    # set xticks labels
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Hypermethylated', 'Hypomethylated'])
    axes[1].set_ylabel("Number of sites")

    # for hypo and hyper seperately, plot the delta_mf vs the distance of the mutated site
    # subset linked_distances_df to only hyper and hypo
    hyper_distances_df = linked_distances_df[linked_distances_df['mutated_site'].isin(hyper_sites)]
    hypo_distances_df = linked_distances_df[linked_distances_df['mutated_site'].isin(hypo_sites)]
    print("hyper:")
    plot_distance_vs_eff(hyper_distances_df, hyper_df, log)
    print("hypo:")
    plot_distance_vs_eff(hypo_distances_df, hypo_df, log)
    return

def count_nearby_muts_one_cpg(cpg_name, all_mut_w_methyl_df, illumina_cpg_locs_df, max_dist = 100000):
    """
    Count the number of nearby mutations to a given CpG in each sample
    @ cpg_name: cpg for which to look for nearby mutants
    @ all_mut_w_methyl_df: df of all mutations that happened in a sample with methylation
    @ illumina_cpg_locs_df: df of cpg locations
    @ returns: the number of nearby C>T mutations 
    """
    # get location of CpG from illumina
    cpg_chr = illumina_cpg_locs_df[illumina_cpg_locs_df['#id'] == cpg_name]['chr'].values[0]
    cpg_start = illumina_cpg_locs_df[illumina_cpg_locs_df['#id'] == cpg_name]['start'].values[0]
    # get mutations that are on the same chr, are within max_dist of the CpG, and are C>T
    nearby_df = all_mut_w_methyl_df[(all_mut_w_methyl_df['chr'] == cpg_chr) & (np.abs(all_mut_w_methyl_df['start'] - cpg_start) < max_dist) & (all_mut_w_methyl_df['mutation'] == 'C>T')]
    return nearby_df

def count_nearby_mutations(cpgs_to_count_df, all_mut_w_methyl_df, illumina_cpg_locs_df, all_methyl_df, max_dist = 100000):
    """
    @ cpgs_to_count_df: df of cpgs to count nearby mutations for
    @ all_mut_w_methyl_df: df of all mutations that happened in a sample with methylation
    @ illumina_cpg_locs_df: df of cpg locations
    @ all_methyl_df: df of all methylation
    @ returns: df of samples x cpgs with number of nearby mutations
    """
    cpg_sample_mut_count_df = pd.DataFrame(0, index=all_methyl_df.columns, columns=cpgs_to_count_df.index)
    # for each of the cpgs to count
    for cpg_name, _ in cpgs_to_count_df.iterrows():
        # find nearby muts across all samples
        nearby_df = count_nearby_muts_one_cpg(cpg_name, all_mut_w_methyl_df, illumina_cpg_locs_df, max_dist)
        # increment count for each sample in result
        nearby_sample_counts = nearby_df['sample'].value_counts()
        cpg_sample_mut_count_df.loc[nearby_sample_counts.index, cpg_name] += nearby_sample_counts.values
    return cpg_sample_mut_count_df

def count_linked_mutations(cpgs_to_count_df, 
                            all_mut_w_methyl_df, 
                            illumina_cpg_locs_df,
                            all_methyl_df, 
                            corr_df, 
                            num_sites=100, 
                            max_dist=100, 
                            percentile_cutoff=.99):
    """
    Count the number of mutations that are in linked sites for each sample
    @ cpgs_to_count_df: df of cpgs to count nearby mutations for
    @ all_mut_w_methyl_df: df of all mutations with methylation
    @ illumina_cpg_locs_df: df of cpg locations
    @ all_methyl_df: df of all methylation
    @ corr_df: df of linkage status
    @ num_sites: number of sites to consider
    @ percentile_cutoff: percentile cutoff for linkage status
    @ returns: df of samples x cpgs with number of mutations in linked sites for that CpG
    """
    cpg_sample_mut_count_df = pd.DataFrame(0, index=all_methyl_df.columns, columns=cpgs_to_count_df.index)
    # for each of the cpgs to count
    for cpg_name, _ in cpgs_to_count_df.iterrows():
        print(cpg_name, flush=True)
        # get this CpG's linked sites
        linked_sites = select_corr_sites(cpg_name, corr_df, num_sites, percentile_cutoff)
        # count the number of mutations in each of these linked sites
        for linked_site in linked_sites:
            # get mutations that are on the same chr, are within max_dist of the CpG, and are C>T
            nearby_df = count_nearby_muts_one_cpg(linked_site, all_mut_w_methyl_df, illumina_cpg_locs_df, max_dist)
            # increment count for each sample in result
            nearby_sample_counts = nearby_df['sample'].value_counts()
            cpg_sample_mut_count_df.loc[nearby_sample_counts.index, cpg_name] += nearby_sample_counts.values
    return cpg_sample_mut_count_df



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
        # a df of the distance from this cpg region to all other cpg regions TODO: split to own fxn
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

class MethylChangeByAge:
    """
    Class that assigns samples to age bins and then calculates observed and predicted mutational methylome difference between age bins
    """
    def __init__(self, methyl_age_df_t):
        # TODO: make drop X, Y, MT chroms
        self.methyl_age_df_t = methyl_age_df_t

    def split_into_age_bins(self, num_age_bins = 10):
        """
        Assign each sample in self.methyl_age_df_t to an age bin
        """
        self.methyl_age_df_t["age_bin"] = pd.qcut(self.methyl_age_df_t['age_at_index'], q=num_age_bins, labels=False)

    def get_observed_methylome_diffs(self):
        """
        Get sum of positive differences and negative differences between the mean of each consecutive age bin
        """
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
        mi = MutationImpact(all_mut_df, linked_sites_diffs_dfs, linked_sites_pvals_dfs)
        # get the expected methylome change for each sample based on mutations
        expected_mut_eff = mi.expected_mut_induced_mf_change(pval_cutoff = pval_cutoff)
        
        # add ages and age bins to expected_mut_eff by joining to self.methyl_age_df_t
        expected_mut_eff_w_age = expected_mut_eff.join(self.methyl_age_df_t[['age_at_index', 'age_bin']], on='sample')

        # group by age bin and calc mean of pos_expected_change, pos_standard_err, neg_expected_change, neg_standard_err
        pred_methylome_diffs = expected_mut_eff_w_age.groupby(['age_bin']).mean()
        # drop first row/age bin to match observed methylome diffs
        pred_methylome_diffs = pred_methylome_diffs.iloc[1:]
        #pred_methylome_diffs = pred_methylome_diffs.iloc[1:]

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


class methylChange:
    """
    Class that compares the observed and predicted methylome change between samples
    """
    def __init__(self, methyl_age_df_t, all_mut_df,linked_sites_diffs_dfs, linked_sites_pvals_dfs, pval_cutoff=0.05, n_pairs=1000, age_window=5, cancer_type='all'):
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

class MutationImpact: 
    """
    Class to calculate information the expected impact of mutations on methyome of the set of samples in sample_mut_df.
    - This the expected impact relative to the same individual with no mutations. 
    """
    def __init__(self, sample_mut_df, linked_sites_diffs_dfs, linked_sites_pvals_dfs, methyl_age_df_t):
        """
        @ sample_mut_df: df of all mutations
        @ linked_sites_diffs_dfs: list of dfs of differences in methylation between linked sites
        @ linked_sites_mwu_pvals_dfs: list of dfs of mwu pvals between linked sites
        @ pval_cutoff: pval cutoff for significance
        """
        # TODO: make drop X, Y, MT chroms
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

def main(corr_fns, illumina_cpg_locs_df, ct_mut_in_measured_cpg_w_methyl_df, all_meta_df, all_methyl_df_t, out_dir):
    # read correlations
    corr_df = read_correlations(corr_fns, illumina_cpg_locs_df)

    # join ages with mutations
    mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t = add_ages_to_methylation(ct_mut_in_measured_cpg_w_methyl_df, all_meta_df, all_methyl_df_t)

    # get num_mut_sites sites with largeset MF differences
    num_mut_sites = 100
    most_negative_diffs = mut_in_measured_cpg_w_methyl_age_df[mut_in_measured_cpg_w_methyl_age_df['#id'].isin(corr_df.columns)].sort_values(by='difference').iloc[:num_mut_sites]
    max_diff_corr_df = corr_df[most_negative_diffs['#id']]
    # calculate mutation impact varying percentile of linked CpG sites
    result_dfs = mutation_eff_varying_linkage(mut_in_measured_cpg_w_methyl_age_df, max_diff_corr_df, all_methyl_age_df_t,illumina_cpg_locs_df, num_linked_sites = 100, age_bin_size = 10)

    # plot sig bars
    plot_sig_bars(result_dfs)

    # write out results
    # output results
    for i in range(len(PERCENTILES)):
        result_dfs[i].to_parquet(os.path.join(out_dir,"result_100_sites_varying_percentile_chr1_linked_sites_df_{}.parquet".format(PERCENTILES[i])))
    
    return result_dfs
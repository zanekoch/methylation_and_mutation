import utils
import pandas as pd
import numpy as np
from rich.progress import track
from statsmodels.stats.weightstats import ztest as ztest
from scipy import stats


PERCENTILES = [1]#np.flip(np.linspace(0, 1, 6))

def detect_effect_in_other_samples(sites_to_test, mut_in_measured_cpg_w_methyl_age_df):
        """
        Return the samples that had a mutation in sites_to_test
        @ sites_to_test: the sites to see if were mutated or in close_measured in other samples (cg#####)
        @ mut_nearby_measured_df: df of mutations that have at least one measured CpG within max_dist of the mutation. 'close_measured' column is a list of the measured cpgs within max_dist of the mutation.
        @ df_w_illum_df: 
        @ returns: a list of samples
        """
        samples_to_exclude = np.array([])
        for site_to_test in sites_to_test:
            # exclude samples that have a mutation in site_to_test
            to_exlude = mut_in_measured_cpg_w_methyl_age_df[mut_in_measured_cpg_w_methyl_age_df['#id'] == site_to_test]['case_submitter_id'].to_numpy()
            samples_to_exclude = np.append(samples_to_exclude, to_exlude)
        return samples_to_exclude

def select_corr_sites(corr_df,
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
    q = corr_df.quantile(percentile, interpolation='lower')
    # select num sites closest to q
    comparison_sites = corr_df.iloc[(corr_df - q).abs().argsort().iloc[:num]].index
    return comparison_sites

def compare_sites(same_age_tissue_methyl_df, mut_sample_methyl_df):
    """
    For a given mutation, compare the methylation of input sites (linked or nonlinked) between mutated and non-mutated sample
    @ same_age_samples_mf_df: dataframe of methylation values for non-mutated samples of same age as mutated sample at comparison sites (either linked or nonlinked)
    @ mut_sample_comparison_mfs_df: dataframe of methylation values for mutated sample at comparison sites (either linked or nonlinked)
    @ returns: Dataframe with rows being comparison sites, columns delta_mf (average difference of mutated sample and non mutated sample at the comparison site that is the row) and ztest_pval (which says if the mutated sample was significantly different from the other samples at that site)
    """
    # subtract mut_sample_comparison_mfs_df from every row (sample) in same_age_samples_mf_df
    difference_at_comparison_sites = same_age_tissue_methyl_df.subtract(mut_sample_methyl_df.iloc[0])
    # switch the sign to make delta_mf = mut_sample_mf - same_age_sample_mf
    difference_at_comparison_sites = difference_at_comparison_sites.mul(-1)
    # get mean average difference (delta_mf) at each site
    mean_diff_each_comparison_site = pd.DataFrame(difference_at_comparison_sites.mean(axis = 0), columns=['delta_mf'])
    # add mut_sample_comparison_mfs_df as last row of same_age_samples_mf_df
    all_samples_at_comparison_sites = same_age_tissue_methyl_df.append(mut_sample_methyl_df)
    # calculate a z score pvalue for the mut_sample site being different from the non-mutated samples
    mean_diff_each_comparison_site['zscore'] = all_samples_at_comparison_sites.apply(lambda x: stats.zscore(x, nan_policy='omit')[-1], axis=0)
    # get 2-sided pvalues
    mean_diff_each_comparison_site['ztest_pval'] = stats.norm.sf(abs(mean_diff_each_comparison_site['zscore']))*2
    return mean_diff_each_comparison_site[['delta_mf', 'ztest_pval']]

def effect_one_mutation(all_methyl_age_df_t,
                        illumina_cpg_locs_df,
                        mut_in_measured_cpg_w_methyl_age_df,
                        percentile,
                        num_linked_sites,
                        age_bin_size,
                        mut_sample,
                        mut_cpg):
    # get chrom of this site
    mut_cpg_chr = illumina_cpg_locs_df[illumina_cpg_locs_df['#id'] == mut_cpg]['chr'].iloc[0]
    # limit comparison sites to cpgs on same chrom 
    same_chr_cpgs = illumina_cpg_locs_df[(illumina_cpg_locs_df['chr'] == mut_cpg_chr) & (illumina_cpg_locs_df['#id'].isin(all_methyl_age_df_t.columns))]['#id'].to_list()
    this_chr_methyl_age_df_t = all_methyl_age_df_t[same_chr_cpgs + ['age_at_index', 'dataset']]
    # get the MF's of the same age samples and tissue samples, not including the mutated sample
    same_age_tissue_samples_mf_df = utils.get_same_age_and_tissue_samples(this_chr_methyl_age_df_t,mut_sample, age_bin_size)
    same_age_tissue_samples_mf_df.drop(columns=['age_at_index', 'dataset'], inplace=True)
    # if there are not enough samples of the same age and tissue, warn and skip this site
    if len(same_age_tissue_samples_mf_df) <= 10:
        print("WARNING: Not enough samples of the same age and tissue to calculate effect of mutation at site: ".format(mut_cpg), flush=True)
        return None
    # in same_age_tissue_samples_mf_df (which does not include the mutated sample), get the correlation of the mutated site to all other sites
    same_age_tissue_samples_mf_no_mut = same_age_tissue_samples_mf_df.drop(columns=[mut_cpg])
    corr_df = same_age_tissue_samples_mf_no_mut.corrwith(same_age_tissue_samples_mf_df[mut_cpg], axis=0)
    # select the linked sites TODO: test if positively or negatively correlated sites are better
    linked_sites = select_corr_sites(corr_df, num_linked_sites, percentile)

    # exclude samples that have a mutation in any of the linked sites
    samples_to_exclude = detect_effect_in_other_samples(linked_sites.to_list(), mut_in_measured_cpg_w_methyl_age_df)
    same_age_tissue_non_mut_samples = np.setdiff1d(same_age_tissue_samples_mf_df.index.values, samples_to_exclude)
    if (len(same_age_tissue_samples_mf_df.index.values) - len(same_age_tissue_non_mut_samples)) > 0:
                print("{} samples excluded".format(len(same_age_tissue_samples_mf_df.index.values) - len(same_age_tissue_non_mut_samples)))
    same_age_tissue_samples_mf_df = same_age_tissue_samples_mf_df.loc[same_age_tissue_non_mut_samples]

    # measure the change in methylation between linked sites in the mutated sample and in other non-mutated samples of the same age and tissue
    linked_diff = compare_sites(same_age_tissue_methyl_df = same_age_tissue_samples_mf_df[linked_sites], mut_sample_methyl_df = this_chr_methyl_age_df_t.loc[mut_sample, linked_sites] )
    # return lists to add to dictionaries
    return mut_cpg, linked_sites, linked_diff['delta_mf'].to_list(), linked_diff['ztest_pval'].to_list()

def measure_mut_eff(muts_to_test,
                    all_methyl_age_df_t,
                    illumina_cpg_locs_df,
                    mut_in_measured_cpg_w_methyl_age_df,
                    percentile,
                    num_linked_sites,
                    age_bin_size):
    """
    Given a correlation matrix, calculate the effect of a mutation on the methylation of linked sites (num_linked_sites starting from specified percentile) 
    @ muts_to_test: list of mutations to test
    @ all_methyl_age_df_t: methylation dataframe with age information
    @ mut_in_measured_cpg_w_methyl_age_df: methylation dataframe with age information for mutated sites
    @ illumina_cpg_locs_df:
    @ percentile: percentile to draw comparison sites from
    @ num_linked_sites: number of comparison sites to draw
    @ age_bin_size: age bin size to use for selected samples to compare at comparison sites
    @ returns: a dataframe of pvals and effect sizes comparing linked sites in mutated sample to linked sites in non-mutated (within age_bin_size/2 year age) samples, for: MabsErr, MavgErr, pearson r, and WilcoxonP 
    """
    # for each mutated CpG that we have correlation for
    linked_sites_names_dict, linked_site_diffs_dict, linked_site_z_pvals_dict = {}, {}, {}
    
    each_perc_result_lists = []
    # iterate across CpGs we are testing
    for mut_tup in track(muts_to_test, description="Analyzing each mutation"):
    #for mut_tup in muts_to_test:
        each_perc_result_lists.append(
            effect_one_mutation(all_methyl_age_df_t, illumina_cpg_locs_df, mut_in_measured_cpg_w_methyl_age_df, percentile, num_linked_sites, age_bin_size, mut_sample = mut_tup[0], mut_cpg = mut_tup[1])
            )
    # go through each result list and if it is == [None], remove it
    each_perc_result_lists = [x for x in each_perc_result_lists if x != None]
    # put the result lists into dictionaries with key being mut_cpg
    for this_perc_result_list in each_perc_result_lists:
        mut_cpg = this_perc_result_list[0]
        linked_sites_names_dict[mut_cpg] = this_perc_result_list[1]
        linked_site_diffs_dict[mut_cpg] = this_perc_result_list[2]
        linked_site_z_pvals_dict[mut_cpg] = this_perc_result_list[3]
    
    linked_sites_names_df = pd.DataFrame.from_dict(linked_sites_names_dict, orient='index')
    linked_sites_diffs_df = pd.DataFrame.from_dict(linked_site_diffs_dict, orient='index')
    linked_sites_z_pvals_df = pd.DataFrame.from_dict(linked_site_z_pvals_dict, orient='index')
    
    return linked_sites_names_df, linked_sites_diffs_df, linked_sites_z_pvals_df

def mutation_eff_varying_linkage_perc(muts_to_test,
                                all_methyl_age_df_t,
                                illumina_cpg_locs_df,
                                mut_in_measured_cpg_w_methyl_age_df,
                                num_linked_sites,
                                age_bin_size = 10):
    """
    For each percentile of comparison sites, get the effect size of the mutation in the mutated sample compared to the effect size of the mutation in the non-mutated samples of the same age.
    @ muts_to_test: list of tuples of (mutated sample, mutated CpG)
    @ mut_in_measured_cpg_w_methyl_age_df: methylation dataframe with age information for mutated sites
    @ all_methyl_age_df_t: methylation dataframe with age information for all samples
    @ illumina_cpg_locs_df: dataframe of CpG locations
    @ num_linked_sites: number of comparison sites to draw
    @ age_bin_size: age bin size to use for selected samples to compare at comparison sites
    """ 
    # calculate results varying percentile of linked CpG sites, only chr1 sites
    linked_sites_names_dfs, linked_sites_diffs_dfs, linked_sites_z_pvals_dfs = [], [], []
    for percentile in PERCENTILES:
        print("Starting percentile: {}".format(percentile))
        linked_sites_names_df, linked_sites_diffs_df, linked_sites_z_pvals_df = measure_mut_eff(muts_to_test, all_methyl_age_df_t, illumina_cpg_locs_df, mut_in_measured_cpg_w_methyl_age_df, percentile, num_linked_sites, age_bin_size)
        linked_sites_names_dfs.append(linked_sites_names_df)
        linked_sites_diffs_dfs.append(linked_sites_diffs_df)
        linked_sites_z_pvals_dfs.append(linked_sites_z_pvals_df)
    return linked_sites_names_dfs, linked_sites_diffs_dfs, linked_sites_z_pvals_dfs
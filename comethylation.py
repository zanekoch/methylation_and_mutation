import pandas as pd
from scipy import stats
import numpy as np
import os
import matplotlib.pyplot as plt
import utils
import seaborn as sns

PERCENTILES = utils.get_percentiles()

def select_pos_corr_sites(in_cpg,
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
    # limit to sites posive correlated with in_cpg
    pos_corr_df = corr_df[corr_df[in_cpg] > 0]
    # get the value of the qth percentile cpg site
    q = pos_corr_df[in_cpg].quantile(percentile, interpolation='lower')
    # select num sites closest to q
    comparison_sites = pos_corr_df.iloc[(pos_corr_df[in_cpg] - q).abs().argsort().iloc[:num], 0].index
    return comparison_sites

def select_closest_sites(in_cpg,
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


def comparison_site_comparison(same_age_samples_mf_df, mut_sample_comparison_mfs_df, bootstrap=False):
    """
    For 1 site, compare the methylation of linked sites between mutated and non-mutated sample
    @ returns: results compared s.t. values are mutated - non-mutated sample (so negative MAvgerr means non-mutated had higher MF)
    """
    # calculate mae, mean abs error, and r^2 between every non-mut and the mut sample
    result_df = pd.DataFrame(columns=['mean_abs_err', 'mean_avg_err'])
    result_df['mean_abs_err'] = same_age_samples_mf_df.apply(lambda x: np.average(np.abs(mut_sample_comparison_mfs_df - x)), axis=1)
    result_df['mean_avg_err'] = same_age_samples_mf_df.apply(lambda x: np.average(mut_sample_comparison_mfs_df - x), axis=1)
    return result_df

def measure_mut_eff_on_module_other_backgrnd(mut_linkage_df,
                                            linkage_type,
                                            all_methyl_age_df_t, 
                                            ct_mut_in_measured_cpg_w_methyl_age_df,
                                            illumina_cpg_locs_df,
                                            percentile,
                                            num_linked_sites,
                                            age_bin_size):
    """
    Given a correlation matrix, calculate the effect of a mutation on the methylation of linked sites (num_linked_sites starting from specified percentile) 
    @ mut_linkage_df: dataframe with columns corresponding to mutated sites we are measured effect of and rows the linkage of that site to all other sites on same chrom
    @ linkage_type: type of linkage in mut_linkage_df either 'methylation_corr' or 'distance'
    @ all_methyl_age_df_t: methylation dataframe with age information
    @ ct_mut_in_measured_cpg_w_methyl_age_df: methylation dataframe with age information for mutated sites
    @ illumina_cpg_locs_df:
    @ percentile: percentile to draw comparison sites from
    @ num_linked_sites: number of comparison sites to draw
    @ age_bin_size: age bin size to use for selected samples to compare at comparison sites
    @ returns: a dataframe of pvals and effect sizes comparing linked sites in mutated sample to linked sites in non-mutated (within age_bin_size/2 year age) samples, for: MabsErr, MavgErr, pearson r, and WilcoxonP 
    """
    # for each mutated CpG that we have correlation for
    all_results = []
    for mut_cpg in mut_linkage_df.columns:
        # get chrom of this site and which sample had this cpg mutated
        mut_cpg_chr = illumina_cpg_locs_df[illumina_cpg_locs_df['#id'] == mut_cpg]['chr'].iloc[0]
        mut_sample = ct_mut_in_measured_cpg_w_methyl_age_df[ct_mut_in_measured_cpg_w_methyl_age_df['#id'] == mut_cpg]
        # limit comparison sites to cpgs on same chrom 
        same_chr_cpgs = illumina_cpg_locs_df[illumina_cpg_locs_df['chr'] == mut_cpg_chr]['#id'].to_list()
        same_chr_linkage_df = mut_linkage_df.loc[mut_linkage_df.index.isin(same_chr_cpgs)]
        # drop mut_cpg from same_chr_linkage_df so it is not selected as a comparison site
        same_chr_linkage_df.drop(mut_cpg, axis=0, inplace=True)
        # get comparison sites based on method
        if linkage_type == 'methylation_corr':
            # get the positively correlated sites that are on the same chromosome as the mutated CpG
            comparison_sites = select_pos_corr_sites(mut_cpg, same_chr_linkage_df, num_linked_sites, percentile)
            low_comparison_sites = select_pos_corr_sites(mut_cpg, same_chr_linkage_df, num_linked_sites, 0)#np.abs(same_chr_linkage_df[mut_cpg]).nsmallest(num_linked_sites).index.to_list()
        elif linkage_type == 'distance':
            # get the closest sites that are on the same chromosome as the mutated CpG
            comparison_sites = select_closest_sites(mut_cpg, same_chr_linkage_df, num_linked_sites, percentile)
            # farthest sites are 100th percentile
            low_comparison_sites = select_closest_sites(mut_cpg, same_chr_linkage_df, num_linked_sites, 1)
        else:
            raise ValueError("linkage_type must be either 'methylation_corr' or 'distance'")
        # get this sample's age and name
        this_age = mut_sample['age_at_index'].to_list()[0]
        this_sample_name = mut_sample['case_submitter_id']
        # get this mutated sample's MF at comparison sites
        mut_sample_comparison_mfs_df = all_methyl_age_df_t.loc[this_sample_name, comparison_sites] 
        # get average mfs at comparison sites for all other samples of within age_bin_size/2 years of age on either side
        same_age_samples_mf_df = all_methyl_age_df_t[np.abs(all_methyl_age_df_t['age_at_index'] - this_age) <= age_bin_size/2]
        # measure the change in methylation between these sites in the mutated sample and in other non-mutated samples of the same age
        mut_result_df = comparison_site_comparison(same_age_samples_mf_df.drop(index = this_sample_name)[comparison_sites], mut_sample_comparison_mfs_df)
        
        # do same comparison but seeing if CpGs with low absolute correlation to mut_cpg also changed same amount
        low_mut_sample_comparison_mfs_df = all_methyl_age_df_t.loc[this_sample_name, low_comparison_sites]
        low_result_df = comparison_site_comparison(same_age_samples_mf_df.drop(index = this_sample_name)[low_comparison_sites], low_mut_sample_comparison_mfs_df)

        # compare mut_result_df to low_result_df to see if there are less significant differences in less linked CpGs
        this_mut_results = []
        # for each mutated site and corresponding linked sistes
        for label, col in low_result_df.items(): 
            # test for diff in distr. of linked vs non linked sites (we expect differences to be less negative in non-linked sites)
            this_mut_results.append(stats.ranksums(col, mut_result_df[label], alternative='greater').pvalue)
            """this_mut_results.append(stats.ttest_ind(col, mut_result_df[label], alternative='greater').pvalue"""
            # get the mean effect across all linked sites
            linked_eff = np.average(mut_result_df[label])
            # and non-linked sites
            non_linked_eff = np.average(col)
            # and mean difference
            effect_size = np.average((mut_result_df[label] - col).dropna()) 
            this_mut_results += [effect_size, linked_eff, non_linked_eff]
        all_results.append(this_mut_results)
    
    result_df = pd.DataFrame(all_results, columns = ['p_mean_abs_err', 'eff_mean_abs_err', 'linked_mean_abs_err', 'non_linked_mean_abs_err', 'p_mean_avg_err', 'eff_mean_avg_err', 'linked_mean_avg_err', 'non_linked_mean_avg_err'] )
    return result_df

def mutation_eff_varying_linkage(mut_linkage_df,
                                linkage_type,
                                ct_mut_in_measured_cpg_w_methyl_age_df,
                                all_methyl_age_df_t,
                                illumina_cpg_locs_df,
                                num_linked_sites,
                                age_bin_size):
    """
    For each percentile of comparison sites, get the effect size of the mutation in the mutated sample compared to the effect size of the mutation in the non-mutated samples of the same age.
    @ mut_linkage_df: dataframe with columns corresponding to mutated sites we are measured effect of and rows the linkage of that site to all other sites on same chrom
    @ linkage_type: type of linkage in mut_linkage_df either 'methylation_corr' or 'distance'
    @ ct_mut_in_measured_cpg_w_methyl_age_df: methylation dataframe with age information for mutated sites
    @ all_methyl_age_df_t: methylation dataframe with age information for all samples
    @ illumina_cpg_locs_df: dataframe of CpG locations
    @ num_linked_sites: number of comparison sites to draw
    @ age_bin_size: age bin size to use for selected samples to compare at comparison sites
    """ 
    # calculate results varying percentile of linked CpG sites, only chr1 sites
    result_dfs = []
    for percentile in PERCENTILES:
        print(percentile)
        res_df = measure_mut_eff_on_module_other_backgrnd(mut_linkage_df, linkage_type, all_methyl_age_df_t, ct_mut_in_measured_cpg_w_methyl_age_df, illumina_cpg_locs_df, percentile, num_linked_sites, age_bin_size)
        result_dfs.append(res_df)
    return result_dfs

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
    ct_mut_in_measured_cpg_w_methyl_age_df =  to_join_ct_mut_in_measured_cpg_w_methyl_df.join(all_meta_df, on =['case_submitter_id'], rsuffix='_r',how='inner')
    # join ages with methylation
    all_methyl_age_df_t = all_meta_df.join(all_methyl_df_t, on =['sample'], rsuffix='_r',how='inner')
    return ct_mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t
    
def plot_sig_bars(result_dfs):
    """
    Plot the sig bars for mean avg err
    """
    # reverse order of result_dfs
    result_dfs = result_dfs[::1]
    sig_result_effs_dict = utils.test_sig(result_dfs)
    # plot abs error
    r = [i for i in range(len(PERCENTILES))]
    raw_data = {'sig': sig_result_effs_dict['m_avg_err_p'], 'non_sig': [len(result_dfs[0]) - i for i in sig_result_effs_dict['m_avg_err_p']] }
    plot_df = pd.DataFrame(raw_data)
    totals = [i+j for i,j in zip(plot_df['sig'], plot_df['non_sig'])]
    sig = [i / j * 100 for i,j in zip(plot_df['sig'], totals)]
    non_sig = [i / j * 100 for i,j in zip(plot_df['non_sig'], totals)]
    # plot
    barWidth = 0.85
    names = [round(i, 2) for i in  PERCENTILES]
    # Create blue Bars
    plt.bar(r, sig, color='steelblue', edgecolor='white', width=barWidth, label="Significant")
    # Create red Bars
    plt.bar(r, non_sig, bottom=sig, color='maroon', edgecolor='white', width=barWidth, label="Not significant")
    # Custom x axis
    plt.xticks(r, names)
    plt.xlabel("Distance-based linkage percentile")
    plt.ylabel("Percent of mutations with significant effect")
    plt.legend()



def plot_eff_line(result_dfs, ct_mut_in_measured_cpg_w_methyl_age_df, max_diff_corr_df, sig_only):
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
            linked_heights.append(result_dfs[i]['linked_mean_avg_err'].mean())
            nonlinked_heights.append(result_dfs[i]['non_linked_mean_avg_err'].mean())

    # get the average MF change of the mutated sites
    avg_mut_mf_change = ct_mut_in_measured_cpg_w_methyl_age_df[ct_mut_in_measured_cpg_w_methyl_age_df['#id'].isin(max_diff_corr_df.columns)]['difference'].mean()
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
    axes.set_xlabel("Linkage percentile")

def plot_eff_violin(result_dfs):
    # make a df with columns: linkage percentile, delta MV, linked or not
    to_plot_dfs = []
    for i in range(len(result_dfs)):
        this_to_plot_dict = {'MavgErr': result_dfs[i]['linked_mean_avg_err'].to_list() + result_dfs[i]['non_linked_mean_avg_err'].to_list(), 'linkage_perc': [round(PERCENTILES[i],2) for j in range(len(result_dfs[i])*2)], 'Linkage status': ["Linked" for j in range(len(result_dfs[i]))] + ["Non-linked" for j in range(len(result_dfs[i]))]}
        this_to_plot_df = pd.DataFrame(this_to_plot_dict)
        to_plot_dfs.append(this_to_plot_df)
    to_plot_df = pd.concat(to_plot_dfs)
    # uncomment next line to reverse for distance-based linkage
    #to_plot_df['linkage_perc'] = round(1-to_plot_df['linkage_perc'],2)
    # violin plot
    fig, axes = plt.subplots(figsize=(16, 5))
    my_pal = {"Linked": "steelblue", "Non-linked": "skyblue"}
    p = sns.violinplot(x="linkage_perc", y="MavgErr", hue="Linkage status", data=to_plot_df, scale="count",palette=my_pal, ax=axes, split=True)
    p.set_xlabel("Correlation-based linkage percentile")
    p.set_ylabel(r"$\Delta$MF")
    axes.invert_xaxis()
    # move legend to the bottom right
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles[::-1], labels[::-1], loc='lower right')
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
        linked_sites = select_pos_corr_sites(cpg_name, corr_df, num_sites, percentile_cutoff)
        # count the number of mutations in each of these linked sites
        for linked_site in linked_sites:
            # get mutations that are on the same chr, are within max_dist of the CpG, and are C>T
            nearby_df = count_nearby_muts_one_cpg(linked_site, all_mut_w_methyl_df, illumina_cpg_locs_df, max_dist)
            # increment count for each sample in result
            nearby_sample_counts = nearby_df['sample'].value_counts()
            cpg_sample_mut_count_df.loc[nearby_sample_counts.index, cpg_name] += nearby_sample_counts.values
    return cpg_sample_mut_count_df

def main(corr_fns, illumina_cpg_locs_df, ct_mut_in_measured_cpg_w_methyl_df, all_meta_df, all_methyl_df_t, out_dir):
    # read correlations
    corr_df = read_correlations(corr_fns, illumina_cpg_locs_df)


    # join ages with mutations
    ct_mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t = add_ages_to_methylation(ct_mut_in_measured_cpg_w_methyl_df, all_meta_df, all_methyl_df_t)

    # get num_mut_sites sites with largeset MF differences
    num_mut_sites = 100
    most_negative_diffs = ct_mut_in_measured_cpg_w_methyl_age_df[ct_mut_in_measured_cpg_w_methyl_age_df['#id'].isin(corr_df.columns)].sort_values(by='difference').iloc[:num_mut_sites]
    max_diff_corr_df = corr_df[most_negative_diffs['#id']]
    # calculate mutation impact varying percentile of linked CpG sites
    result_dfs = mutation_eff_varying_linkage(ct_mut_in_measured_cpg_w_methyl_age_df, max_diff_corr_df, all_methyl_age_df_t,illumina_cpg_locs_df, num_linked_sites = 100, age_bin_size = 10)

    # plot sig bars
    plot_sig_bars(result_dfs)

    # write out results
    # output results
    for i in range(len(PERCENTILES)):
        result_dfs[i].to_parquet(os.path.join(out_dir,"result_100_sites_varying_percentile_chr1_linked_sites_df_{}.parquet".format(PERCENTILES[i])))
    
    return result_dfs
import pandas as pd
from scipy import stats
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import utils

PERCENTILES = np.flip(np.linspace(0.01, .99, 10))

def comparison_site_comparison(same_age_samples_mf_df, mut_sample_comparison_mfs_df, bootstrap=False):
    # calculate mae, mean abs error, and r^2 between every non-mut and the mut sample
    result_df = pd.DataFrame(columns=['mean_abs_err', 'mean_avg_err', 'r', 'wilcoxon_p'])
    result_df['mean_abs_err'] = same_age_samples_mf_df.apply(lambda x: np.average(np.abs(x - mut_sample_comparison_mfs_df)), axis=1)
    result_df['mean_avg_err'] = same_age_samples_mf_df.apply(lambda x: np.average(x - mut_sample_comparison_mfs_df), axis=1)
    result_df['r'] = same_age_samples_mf_df.apply(lambda x: np.corrcoef(x, mut_sample_comparison_mfs_df)[1][0], axis=1)
    wil_ps = []
    for i, row in same_age_samples_mf_df.iterrows():
        if not bootstrap:
            wil_ps.append( stats.ranksums(row.to_numpy(), mut_sample_comparison_mfs_df.to_numpy()[0]).pvalue )
        else:
            wil_ps.append( stats.ranksums(row.to_numpy(), mut_sample_comparison_mfs_df.to_numpy()).pvalue )
    #result_df['wilcoxon_p'] = same_age_samples_mf_df.apply(lambda x: stats.ranksums(x.to_numpy(), mut_sample_comparison_mfs_df.to_numpy()[0]).pvalue, axis=1 )
    result_df['wilcoxon_p'] = wil_ps
    return result_df


def measure_mut_eff_on_module_other_backgrnd(corr_df, all_methyl_age_df_t, ct_mut_in_measured_cpg_w_methyl_age_df, illumina_cpg_locs_df, percentile, num_comp_sites, age_bin_size):
    """
    Given a correlation matrix, calculate the effect of a mutation on the methylation of linked sites (num_comp_sites starting from specified percentile) 
    @ returns: a dataframe of pvals and effect sizes comparing linked sites in mutated sample to linked sites in non-mutated (within age_bin_size/2 year age) samples, for: MabsErr, MavgErr, pearson r, and WilcoxonP 
    
    """
    # for each mutated CpG that we have correlation for
    all_results = []
    for mut_cpg in corr_df.columns:
        # limit comparison sites to cpgs on same chrom first
        mut_cpg_chr = illumina_cpg_locs_df[illumina_cpg_locs_df['#id'] == mut_cpg]['chr'].iloc[0]
        same_chr_cpgs = illumina_cpg_locs_df[illumina_cpg_locs_df['chr'] == mut_cpg_chr]['#id'].to_list()
        same_chr_corr_df = corr_df.loc[corr_df.index.isin(same_chr_cpgs)]
        # take absolute value of all correlations, so we pick comparison sites based on magnitude
        #same_chr_corr_abs_df = np.abs(same_chr_corr_df)
        same_chr_corr_pos_df = corr_df[corr_df[mut_cpg] >=0]

        # get top percentile correlated sites
        q = same_chr_corr_pos_df[mut_cpg].quantile(percentile, interpolation='lower')
        comparison_sites = same_chr_corr_pos_df.iloc[(same_chr_corr_pos_df[mut_cpg] - q).abs().argsort().iloc[:num_comp_sites],0].index
        # get which sample had this cpg mutated and its age
        mut_sample = ct_mut_in_measured_cpg_w_methyl_age_df[ct_mut_in_measured_cpg_w_methyl_age_df['#id'] == mut_cpg]
        try:
            this_age = mut_sample['age_at_index'].to_list()[0]
        except: # for some reason this CpG is not in the dataframes of mutants
            print(mut_cpg)
            print(mut_sample)
            continue
        this_sample_name = mut_sample['case_submitter_id']
        # get this mutated sample's MF at comparison sites
        """CHANGED THIS FROM all_methyl_df_t, might have broken"""
        mut_sample_comparison_mfs_df = all_methyl_age_df_t.loc[this_sample_name, comparison_sites] 
        # get average mfs at comparison sites for all other samples of within age_bin_size/2 years of age on either side
        same_age_samples_mf_df = all_methyl_age_df_t[np.abs(all_methyl_age_df_t['age_at_index'] - this_age) <= age_bin_size/2]
        
        # measure the change in methylation between these sites in the mutated sample and in other non-mutated samples of the same age
        mut_result_df = comparison_site_comparison(same_age_samples_mf_df.drop(index = this_sample_name)[comparison_sites], mut_sample_comparison_mfs_df)
        
        # do same comparison but seeing if CpGs with low absolute correlation to mut_cpg also changed same amount
        low_corr_comparison_sites = np.abs(same_chr_corr_df[mut_cpg]).nsmallest(num_comp_sites).index.to_list()
        
        low_mut_sample_comparison_mfs_df = all_methyl_age_df_t.loc[this_sample_name, low_corr_comparison_sites]
        low_result_df = comparison_site_comparison(same_age_samples_mf_df.drop(index = this_sample_name)[low_corr_comparison_sites], low_mut_sample_comparison_mfs_df)
        # compare mut_result_df to low_result_df to see if there are less significant differences in lower corr CpGs
        this_mut_results = []
        # for each mutated site and corresponding linked sistes
        for label, col in low_result_df.items(): #.iloc[:, :-1]
            # test for diff in distr. of linked vs non linked sites
            # take average of differenes
            # this results in a pval & effect size for each linked site
            if label == 'r' or label == 'wilcoxon_p':
                this_mut_results.append(stats.ranksums(mut_result_df[label], col, alternative='less').pvalue)
                linked_eff = np.average(mut_result_df[label])
                non_linked_eff = np.average(col)
                effect_size = np.average((mut_result_df[label] - col).dropna()) 
                this_mut_results+=[effect_size, linked_eff, non_linked_eff]
            else:
                #this_mut_results.append(stats.ranksums(col, mut_result_df[label], alternative='less').pvalue)
                this_mut_results.append(stats.ttest_ind(col, mut_result_df[label], alternative='less').pvalue)
                linked_eff = np.average(mut_result_df[label])
                non_linked_eff = np.average(col)
                effect_size = np.average((mut_result_df[label] - col).dropna()) 
                this_mut_results+=[effect_size, linked_eff, non_linked_eff]
        all_results.append(this_mut_results)
    
    result_df = pd.DataFrame(all_results, columns = ['p_mean_abs_err', 'eff_mean_abs_err', 'linked_mean_abs_err', 'non_linked_mean_abs_err', 'p_mean_avg_err', 'eff_mean_avg_err', 'linked_mean_avg_err', 'non_linked_mean_avg_err','p_r', 'eff_r', 'linked_r', 'non_linked_r', 'p_wilc', 'eff_wilc','linked_wilc', 'non_linked_wilc'] )
    bonf_p_val = 0.05/len(all_results)
    print(len(result_df[result_df['p_mean_avg_err'] < bonf_p_val]), len(result_df[result_df['p_mean_abs_err'] < bonf_p_val]),len(result_df[result_df['p_r'] < bonf_p_val]), len(result_df[result_df['p_wilc'] < bonf_p_val]),  len(result_df))
    return result_df

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
    all_methyl_age_df_t = all_meta_df.join(all_methyl_df_t, on =['case_submitter_id'], rsuffix='_r',how='inner')
    return ct_mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t

def mutation_eff_varying_linkage(ct_mut_in_measured_cpg_w_methyl_age_df, corr_df, all_methyl_age_df_t,illumina_cpg_locs_df, num_mut_sites, num_linked_sites, age_bin_size):
    # get num_mut_sites sites with largeset MF differences
    most_negative_diffs = ct_mut_in_measured_cpg_w_methyl_age_df[ct_mut_in_measured_cpg_w_methyl_age_df['#id'].isin(corr_df.columns)].sort_values(by='difference').iloc[:num_mut_sites]
    negative_corr_df = corr_df[most_negative_diffs['#id']]
    # calculate results varying percentile of linked CpG sites, only chr1 sites
    result_dfs = []
    for percentile in PERCENTILES:
        res_df = measure_mut_eff_on_module_other_backgrnd(corr_df, all_methyl_age_df_t, ct_mut_in_measured_cpg_w_methyl_age_df, illumina_cpg_locs_df, percentile, num_linked_sites, age_bin_size)
        result_dfs.append(res_df)
        print(percentile)
    return result_dfs
    


def plot_sig_bars(result_dfs):
    """
    Plot the sig bars for mean abs err 
    """
    m_abs_err_p, m_abs_errs_eff, m_linked_mean_abs_err, m_non_linked_mean_abs_err, stdev_linked_mean_abs_err, stdev_non_linked_mean_abs_err, m_avg_err_p, m_avg_errs_eff, m_linked_mean_avg_err, m_non_linked_mean_avg_err, r_p, r_eff, wilc_p, wilc_eff = utils.test_sig(result_dfs)
    # plot abs error
    r = [i for i in range(10)]
    raw_data = {'sig': m_abs_err_p, 'non_sig': [len(result_dfs[0]) - i for i in m_abs_err_p] }
    plot_df = pd.DataFrame(raw_data)
    totals = [i+j for i,j in zip(plot_df['sig'], plot_df['non_sig'])]
    sig = [i / j * 100 for i,j in zip(plot_df['sig'], totals)]
    non_sig = [i / j * 100 for i,j in zip(plot_df['non_sig'], totals)]
    # plot
    barWidth = 0.85
    names = [str(i)[:4] for i in  PERCENTILES]
    # Create green Bars
    plt.bar(r, sig, color='steelblue', edgecolor='white', width=barWidth, label="Significant")
    # Create orange Bars
    plt.bar(r, non_sig, bottom=sig, color='maroon', edgecolor='white', width=barWidth, label="Not significant")
    # Custom x axis
    plt.xticks(r, names)
    plt.xlabel("Linkage percentile")
    plt.ylabel("Percent of tests significant")
    plt.legend()

def main(corr_fns, illumina_cpg_locs_df, ct_mut_in_measured_cpg_w_methyl_df, all_meta_df, all_methyl_df_t, out_dir):
    # read correlations
    corr_df = read_correlations(corr_fns, illumina_cpg_locs_df)

    # join ages with mutations
    ct_mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t = add_ages_to_methylation(ct_mut_in_measured_cpg_w_methyl_df, all_meta_df, all_methyl_df_t)

    # calculate mutation impact varying percentile of linked CpG sites
    result_dfs = mutation_eff_varying_linkage(ct_mut_in_measured_cpg_w_methyl_age_df, corr_df, all_methyl_age_df_t,illumina_cpg_locs_df, num_mut_sites = 100, num_linked_sites = 100, age_bin_size = 10)

    # plot sig bars
    plot_sig_bars(result_dfs)

    # write out results
    # output results
    for i in range(len(PERCENTILES)):
        result_dfs[i].to_parquet(os.path.join(out_dir,"result_100_sites_varying_percentile_chr1_linked_sites_df_{}.parquet".format(PERCENTILES[i])))
    
    return result_dfs
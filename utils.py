from random import Random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.style.use("seaborn-deep")
import os 
from scipy import stats
import statsmodels.api as sm
import sys
from collections import defaultdict
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection


# CONSTANTS
VALID_MUTATIONS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G", "G>C","G>A", "A>T", "A>G" , "A>C", "G>T", "C>-"]
JUST_CT = True
DATA_SET = "TCGA"

PERCENTILES = [1]#np.flip(np.linspace(0, 1, 6))


def get_percentiles():
    return PERCENTILES

def drop_cpgs_by_chrom(all_methyl_df_t, chroms_to_drop_l, illumina_cpg_locs_df):
    """
    @ returns: all_methyl_df_t with cpgs in chroms_to_drop dropped
    """
    # get names of all CpGs in chroms_to_drop
    cpgs_to_drop = illumina_cpg_locs_df[illumina_cpg_locs_df['chr'].isin(chroms_to_drop_l)]['#id']
    cols_to_keep = set(all_methyl_df_t.columns.to_list()) - set(cpgs_to_drop.to_list())
    return all_methyl_df_t.loc[:,cols_to_keep]


# returns mut_df joined s.t. only mutations that are in measured CpG sites with methylation data remain
# be aware that drop_duplicates first 
def join_df_with_illum_cpg(mut_df, illumina_cpg_locs_df, all_methyl_df_t):
    """
    @ returns: mut_df joined s.t. only mutations that are in illumina measured CpG sites with methylation data remain    
    """
    # get the illumina CpG that were measureed
    cpgs_measured = all_methyl_df_t.columns
    cpgs_measured_and_illumina = illumina_cpg_locs_df[illumina_cpg_locs_df["#id"].isin(cpgs_measured)]
    # join with illumina_cpgs=_locations_df to get the cpg names column (on both start and chr incase the start val is not unique)
    illumina_cpg_locs_df_to_join = illumina_cpg_locs_df.set_index(['start', 'chr'])
    mutation_in_cpg_df = mut_df.join(illumina_cpg_locs_df_to_join, on=['start', 'chr'], how='inner')
    if not JUST_CT:
        # then also join on start + 1 to get reverse strand mutations that would look like G>A
        illumina_cpg_locs_df['start'] = illumina_cpg_locs_df['start'] + 1
        illumina_cpg_locs_df_to_join = illumina_cpg_locs_df.set_index(['start', 'chr'])
        mutation_in_cpg_second_df = mut_df.join(illumina_cpg_locs_df_to_join, on=['start', 'chr'],how='inner')
        # need to fix the coordinates of mutation_in_cpg_second_df, because right now their starts are not the same
        mutation_in_cpg_second_df['start'] = mutation_in_cpg_second_df['start'] - 1
        # concat to same name so next steps do not need if statement
        mutation_in_cpg_df = pd.concat([mutation_in_cpg_df, mutation_in_cpg_second_df] )
    mutation_in_cpg_df = mutation_in_cpg_df.drop_duplicates(subset = ['chr', 'start', 'mutation'], ignore_index=True)
    # sbset to only measured CpGs 
    # mutation_in_cpg_df = mutation_in_cpg_df[mutation_in_cpg_df['#id'].isin(cpgs_measured_and_illumina['#id'])]
    mutation_in_cpg_df = mutation_in_cpg_df[mutation_in_cpg_df['#id'].isin(all_methyl_df_t.columns)]
    mutation_in_cpg_df = mutation_in_cpg_df.loc[mutation_in_cpg_df['sample'].isin(all_methyl_df_t.index)]
    return mutation_in_cpg_df

def site_characteristics(comparison_sites_df, all_methyl_age_df_t, mut_in_measured_cpg_w_methyl_age_df):
    """
    Return the methylation fraction of each site in sites_df unrolled
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

def plot_mutation_count_by_age(all_mut_df, all_meta_df, dataset_names_list, out_dir):
    """
    All together and for each dataset, plot # of C>T mutations by age
    """
    # df of just ct mutations
    if JUST_CT:
        ct_mutations_df = all_mut_df[(all_mut_df['mutation'] == 'C>T')]
    else:
        ct_mutations_df = all_mut_df[(all_mut_df['mutation'] == 'C>T') | (all_mut_df['mutation'] == 'G>A')]
    # df of counts of ct mutations in each sample with age
    ct_mut_count_by_age = pd.DataFrame(ct_mutations_df['sample'].value_counts()).join(all_meta_df, how='inner')
    ct_mut_count_by_age = ct_mut_count_by_age.rename(columns={"sample":"num_mut"}) # need to fix names after joining 
    # remove outliers 
    ct_mut_count_by_age = ct_mut_count_by_age[(np.abs(stats.zscore(ct_mut_count_by_age['num_mut'])) < 3)]
    # plot all cancer types together
    fig, axes = plt.subplots(facecolor="white")
    axes = ct_mut_count_by_age.plot.scatter(ax=axes, x='age_at_index', y='num_mut', s=1, color='steelblue', alpha=0.7 )
    # add line of best fit
    rlm = sm.RLM(ct_mut_count_by_age['num_mut'], sm.add_constant(ct_mut_count_by_age['age_at_index']),  M=sm.robust.norms.HuberT())
    rlm_results = rlm.fit()
    y_line = rlm_results.predict(sm.add_constant(np.arange(0,100, 1)))
    axes.plot(np.arange(0,100, 1), y_line, color='maroon', alpha=0.7)
    axes.set_xticks([0,30,60,90])
    # get correlation
    rho, pval = stats.spearmanr(ct_mut_count_by_age['age_at_index'], ct_mut_count_by_age['num_mut'])
    correlation = round(rho, 2)
    if JUST_CT:
        axes.set_ylabel("C>T Mutation count in all cancer types")
    else:
        axes.set_ylabel("C>T/G>A Mutation count in all cancer types")
    # do F-test
    A = np.identity(len(rlm_results.params))
    A = A[1:,:]
    f_results = rlm_results.f_test(A).summary()
    # get just p part of string and shorten
    f_results_p = float(f_results.split(',')[1][3:])
    axes.text(0.20, 0.9, 'p value={}\nr={}'.format('{:.2e}'.format(f_results_p), correlation), ha='center', va='center', transform=axes.transAxes,bbox=dict(facecolor='maroon', alpha=0.3))
    fig.savefig(os.path.join(out_dir, 'ct_mut_count_by_age_all_datasets.png'))
    # if there is only one data set, don't do crazy subplots stuff 
    if len(dataset_names_list) <= 1:
        fig, axes = plt.subplots(2, 1, sharey='row', figsize=(4,9), facecolor="white")
        this_name = dataset_names_list[0]
        this_dataset_df = all_mut_df[all_mut_df['dataset'] == this_name]
        if JUST_CT:
                this_ct_mutations_df = this_dataset_df[this_dataset_df['mutation'] == "C>T"]
        else:
            this_ct_mutations_df = this_dataset_df[(this_dataset_df['mutation'] == "C>T") | (this_dataset_df['mutation'] == "G>A")]
        this_ct_mut_count_by_age = pd.DataFrame(this_ct_mutations_df['sample'].value_counts()).join(all_meta_df, how='inner')
        this_ct_mut_count_by_age = this_ct_mut_count_by_age.rename(columns={"sample":"num_mut"})
        # remove outliers 
        this_ct_mut_count_by_age = this_ct_mut_count_by_age[(np.abs(stats.zscore(this_ct_mut_count_by_age['num_mut'])) < 3)]
        ax = axes[0]
        ax.scatter(x=this_ct_mut_count_by_age['age_at_index'], y=this_ct_mut_count_by_age['num_mut'], s=4, color='steelblue', alpha=0.7 )
        # add line of best fit
        rlm = sm.RLM(this_ct_mut_count_by_age['num_mut'], sm.add_constant(this_ct_mut_count_by_age['age_at_index']),  M=sm.robust.norms.HuberT())
        rlm_results = rlm.fit()
        y_line = rlm_results.predict(sm.add_constant(np.arange(0,100, 1)))
        ax.plot(np.arange(0,100, 1), y_line, color='maroon', alpha=0.7)
        # get correlation
        rho, pval = stats.spearmanr(this_ct_mut_count_by_age['age_at_index'], this_ct_mut_count_by_age['num_mut'])
        correlation = round(rho, 2)
        if JUST_CT:
            ax.set_ylabel("C>T mut. count {}".format(this_name))
        else:
            ax.set_ylabel("C>T/G>A Mutation count in {}".format(this_name))
        # do F-test
        A = np.identity(len(rlm_results.params))
        A = A[1:,:]
        f_results = rlm_results.f_test(A).summary()
        # get just p part of string and shorten
        f_results_p = float(f_results.split(',')[1][3:])
        ax.text(0.33, 0.9, 'p value={}\nr={}'.format('{:.2e}'.format(f_results_p), correlation), ha='center', va='center', transform=ax.transAxes,bbox=dict(facecolor='maroon', alpha=0.3))
        ax.set_xticks([0,30,60,90])
        # plot zoomed version
        ax2 = axes[1]
        ax2.scatter(x=this_ct_mut_count_by_age['age_at_index'], y=this_ct_mut_count_by_age['num_mut'], s=4 , color='steelblue', alpha=0.7 )
        ax2.plot(np.arange(0,100, 1), y_line, color='maroon', alpha=0.7)
        if JUST_CT:
            ax2.set_ylabel("C>T mut. count {}".format(this_name))
        else:
            ax2.set_ylabel("C>T/G>A Mutation count in {}".format(this_name))               
        ax2.set_xticks([0, 30,60,90])
        ax2.set_ylim(0,200)
    else:
        # iteratively create all the plots
        fig, axes = plt.subplots(4, 8, sharey='row', figsize=(20,13), facecolor="white")
        for i in range(len(dataset_names_list)):
            this_name = dataset_names_list[i]
            this_dataset_df = all_mut_df[all_mut_df['dataset'] == this_name]
            if JUST_CT:
                 this_ct_mutations_df = this_dataset_df[this_dataset_df['mutation'] == "C>T"]
            else:
                this_ct_mutations_df = this_dataset_df[(this_dataset_df['mutation'] == "C>T") | (this_dataset_df['mutation'] == "G>A")]
            this_ct_mut_count_by_age = pd.DataFrame(this_ct_mutations_df['sample'].value_counts()).join(all_meta_df, how='inner')
            this_ct_mut_count_by_age = this_ct_mut_count_by_age.rename(columns={"sample":"num_mut"})
            # remove outliers 
            this_ct_mut_count_by_age = this_ct_mut_count_by_age[(np.abs(stats.zscore(this_ct_mut_count_by_age['num_mut'])) < 3)]

            if i < 8:
                # plot this dataset
                ax = axes[0][i]
                ax.scatter(x=this_ct_mut_count_by_age['age_at_index'], y=this_ct_mut_count_by_age['num_mut'], s=4, color='steelblue', alpha=0.7 )
                # add line of best fit
                rlm = sm.RLM(this_ct_mut_count_by_age['num_mut'], sm.add_constant(this_ct_mut_count_by_age['age_at_index']),  M=sm.robust.norms.HuberT())
                rlm_results = rlm.fit()
                y_line = rlm_results.predict(sm.add_constant(np.arange(0,100, 1)))
                ax.plot(np.arange(0,100, 1), y_line, color='maroon', alpha=0.7)
                # get correlation
                rho, pval = stats.spearmanr(this_ct_mut_count_by_age['age_at_index'], this_ct_mut_count_by_age['num_mut'])
                correlation = round(rho, 2)
                if JUST_CT:
                    ax.set_ylabel("C>T mut. count {}".format(this_name))
                else:
                    ax.set_ylabel("C>T/G>A Mutation count in {}".format(this_name))
                # do F-test
                A = np.identity(len(rlm_results.params))
                A = A[1:,:]
                f_results = rlm_results.f_test(A).summary()
                # get just p part of string and shorten
                f_results_p = float(f_results.split(',')[1][3:])
                ax.text(0.5, 0.9, 'p value={}\nr={}'.format('{:.2e}'.format(f_results_p), correlation), ha='center', va='center', transform=ax.transAxes,bbox=dict(facecolor='maroon', alpha=0.3))
                ax.set_xticks([0,30,60,90])
                # plot zoomed version
                ax2 = axes[1][i]
                ax2.scatter(x=this_ct_mut_count_by_age['age_at_index'], y=this_ct_mut_count_by_age['num_mut'], s=4 , color='steelblue', alpha=0.7 )
                ax2.plot(np.arange(0,100, 1), y_line, color='maroon', alpha=0.7)
                if JUST_CT:
                    ax2.set_ylabel("C>T mut. count {}".format(this_name))
                else:
                    ax2.set_ylabel("C>T/G>A Mutation count in {}".format(this_name))
                ax2.set_xticks([0, 30,60,90])
                ax2.set_ylim(0,200)
            else:
                # plot this dataset
                ax = axes[2][i-8]
                ax.scatter(x=this_ct_mut_count_by_age['age_at_index'], y=this_ct_mut_count_by_age['num_mut'], s=4, color='steelblue', alpha=0.7 )
                # add line of best fit
                rlm = sm.RLM(this_ct_mut_count_by_age['num_mut'], sm.add_constant(this_ct_mut_count_by_age['age_at_index']),  M=sm.robust.norms.HuberT())
                rlm_results = rlm.fit()
                y_line = rlm_results.predict(sm.add_constant(np.arange(0,100, 1)))
                ax.plot(np.arange(0,100, 1), y_line, color='maroon', alpha=0.7)
                # get correlation
                rho, pval = stats.spearmanr(this_ct_mut_count_by_age['age_at_index'], this_ct_mut_count_by_age['num_mut'])
                correlation = round(rho, 2)
                if JUST_CT:
                    ax.set_ylabel("C>T mut. count {}".format(this_name))
                else:
                    ax.set_ylabel("C>T/G>A Mutation count in {}".format(this_name))
                # do F-test
                A = np.identity(len(rlm_results.params))
                A = A[1:,:]
                f_results = rlm_results.f_test(A).summary()
                # get just p part of string and shorten
                f_results_p = f_results.split(',')[1]
                f_results_p = float(f_results.split(',')[1][3:])
                ax.text(0.5, 0.9, 'p value={}\nr={}'.format('{:.2e}'.format(f_results_p), correlation), ha='center', va='center', transform=ax.transAxes,bbox=dict(facecolor='maroon', alpha=0.3))
                ax.set_xticks([0,30,60,90])
                # plot zoomed version
                ax2 = axes[3][i-8]
                ax2.scatter(x=this_ct_mut_count_by_age['age_at_index'], y=this_ct_mut_count_by_age['num_mut'], s=4, color='steelblue', alpha=0.7 )
                ax2.plot(np.arange(0,100, 1), y_line, color='maroon', alpha=0.7)
                if JUST_CT:
                    ax2.set_ylabel("C>T mut. count {}".format(this_name))
                else:
                    ax2.set_ylabel("C>T/G>A Mutation count in {}".format(this_name))
                ax2.set_ylim(0,200)
                ax2.set_xticks([0,30,60,90])
        fig.savefig(os.path.join(out_dir, 'ct_mut_count_by_age_each_dataset.png'))
    return

def get_methyl_fractions(ct_mutation_in_measured_cpg_df, all_methyl_df_t):
    methyl_fractions = []
    for _, row in ct_mutation_in_measured_cpg_df.iterrows():
        cpg = row['#id']
        samp = row['sample']
        try:
            methyl_fractions.append(all_methyl_df_t.loc[samp,cpg])
        except:
            #print("{} in sample {} not present".format(cpg, samp))
            methyl_fractions.append(-1)
    return methyl_fractions

def get_same_age_means(ct_mutation_in_measured_cpg_df, all_meta_df, all_methyl_df_t):
    means = []
    for i, row in ct_mutation_in_measured_cpg_df.iterrows():
        sample = row['sample']
        cpg = row['#id']
        same_age_samples = all_meta_df[np.abs(all_meta_df['age_at_index'] - all_meta_df.loc[sample]['age_at_index']) < 2.5]    
        this_cpg_mean = same_age_samples.join(all_methyl_df_t, how='inner')[cpg].mean()
        means.append(this_cpg_mean)
    return means

def calc_correlation(ct_mut_in_measured_cpg_w_methyl_df, all_methyl_df_t, num, chr=''):
    # for each mutated site in CpG, calculate correlation matrix
    corr_matrix_dict = {}
    # subset to a single chromsome
    if chr != '':
        ct_mut_in_measured_cpg_w_methyl_df = ct_mut_in_measured_cpg_w_methyl_df[ct_mut_in_measured_cpg_w_methyl_df['chr'] == str(chr)]
    sorted_df = ct_mut_in_measured_cpg_w_methyl_df.sort_values(by=['DNA_VAF'], ascending=False)
    chosen_cpgs = sorted_df.iloc[num-250:]
    for i, row in chosen_cpgs.iterrows():
        this_cpg_corr_matrix = all_methyl_df_t.corrwith(all_methyl_df_t[row['#id']])
        this_cpg_corr_matrix.drop(row['#id'], inplace=True)
        corr_matrix_dict[row['#id']] = this_cpg_corr_matrix
    return corr_matrix_dict

def test_sig(results_dfs, test='p_wilcoxon'):
    """
    @ returns: dict of counts of mutations with significant effects
    """
    # initialize dict of lists
    result_metrics_dict = defaultdict(list)
    # iterate over percentiles, finding mutations with pvalue of impact on linked sites below cutoff
    for i in range(len(PERCENTILES)):
        this_result_df = results_dfs[i]
        bonf_p_val = 0.05/len(this_result_df)
        result_metrics_dict['p_wilcoxon'].append(len(this_result_df[this_result_df[test] < bonf_p_val]))
        result_metrics_dict['p_barlett'].append(len(this_result_df[this_result_df[test] < bonf_p_val]))
        result_metrics_dict['sig_mean_linked_delta_mf'].append(this_result_df[this_result_df[test] < bonf_p_val]['mean_linked_delta_mf'].mean())
    return result_metrics_dict

def calc_correlation(ct_mut_in_measured_cpg_w_methyl_df, all_methyl_df_t, chr=''):
    # for each mutated site on chr calculate correlation matrix
    corr_matrix_dict = {}
    # subset to a single chromsome
    if chr != '':
        ct_mut_in_measured_cpg_w_methyl_df = ct_mut_in_measured_cpg_w_methyl_df[ct_mut_in_measured_cpg_w_methyl_df['chr'] == str(chr)]

    for _, row in ct_mut_in_measured_cpg_w_methyl_df.iterrows():
        this_cpg_corr_matrix = all_methyl_df_t.corrwith(all_methyl_df_t[row['#id']])
        this_cpg_corr_matrix.drop(row['#id'], inplace=True)
        corr_matrix_dict[row['#id']] = this_cpg_corr_matrix
    corr_df = pd.DataFrame(corr_matrix_dict)
    return corr_df

def EWAS(X, y, out_fn):
    """
    Calculates correlataion of CpGs with age and writes to out_fn
    @ X: dataframe of all CpGs as rows and samples as columns
    @ y: series of ages corresponding to samples
    """
    # add olddogs to path
    sys.path.append('/cellar/users/zkoch/olddogs')
    import olddogs as dogs
    # create scanner
    scanner = dogs.scan.scan_covariate()
    # do correlation
    pearson_corrs, _ = scanner.scanCpGs_Correlate(X, y, parallel = False, method = 'pearson')
    # output
    out_dict = {'pearson_corrs': pearson_corrs}
    out_df = pd.DataFrame(out_dict)
    # set index to CpG names
    out_df.index = X.index
    out_df.to_parquet(out_fn)
    return out_df

def get_distances_one_chrom(chrom_name,
                            illumina_cpg_locs_df):
    """
    Calculate absolute distances between all CpGs on a give chromosome
    @ chrom_name: name of chromosome
    @ illumina_cpg_locs_df: dataframe of CpG locations
    @ returns: dataframe of absolute distances between all CpGs on a given chromosome
    """
    # subset to a single chromsome
    illumina_cpg_locs_df_chr = illumina_cpg_locs_df[illumina_cpg_locs_df['chr'] == str(chrom_name)]
    """# subset to CpGs in cpg_subset
    illumina_cpg_locs_df_chr = illumina_cpg_locs_df_chr[illumina_cpg_locs_df_chr['#id'].isin(cpg_subset)]"""
    # for each CpG in illumina_cpg_locs_df_chr
    distances_dict = {}
    for _, row in illumina_cpg_locs_df_chr.iterrows():
        # calculate distance to all other CpGs
        this_cpg_distances = illumina_cpg_locs_df_chr['start'] - row['start']
        distances_dict[row['#id']] = this_cpg_distances
    distances_df = pd.DataFrame(distances_dict)
    distances_df.index = distances_df.columns
    distances_df = np.abs(distances_df)
    return distances_df
    
def read_in_result_dfs(result_base_path, PERCENTILES=PERCENTILES):
    """
    @ result_base_path: path to file of result dfs without PERCENTILE suffix
    @ returns: list of result dataframes
    """
    linked_sites_names_dfs = []
    linked_sites_diffs_dfs = []
    linked_sites_z_pvals_dfs = [] 
    for i in range(len(PERCENTILES)):
        linked_sites_names_dfs.append(pd.read_parquet(result_base_path + '_linked_sites_names_' + str(PERCENTILES[i]) + '.parquet'))
        linked_sites_diffs_dfs.append(pd.read_parquet(result_base_path + '_linked_sites_diffs_' + str(PERCENTILES[i]) + '.parquet'))
        linked_sites_z_pvals_dfs.append(pd.read_parquet(result_base_path + '_linked_sites_pvals_' + str(PERCENTILES[i]) + '.parquet'))
    return linked_sites_names_dfs, linked_sites_diffs_dfs, linked_sites_z_pvals_dfs

def write_out_results_new(out_dir, name, linked_sites_names_dfs, linked_sites_diffs_dfs, linked_sites_z_pvals_dfs):
    """
    @ out_dir: path to directory to write out result dfs, linked_sites_names_dfs, and linked_sites_diffs_dfs
    """
    for i in range(len(PERCENTILES)):
        linked_sites_names_dfs[i].columns = linked_sites_names_dfs[i].columns.astype(str)
        linked_sites_names_dfs[i].to_parquet(out_dir + '/' + name + '_linked_sites_names_' + str(PERCENTILES[i]) + '.parquet')
        linked_sites_diffs_dfs[i].columns = linked_sites_diffs_dfs[i].columns.astype(str)
        linked_sites_diffs_dfs[i].to_parquet(out_dir + '/' + name + '_linked_sites_diffs_' + str(PERCENTILES[i]) + '.parquet')
        linked_sites_z_pvals_dfs[i].columns = linked_sites_z_pvals_dfs[i].columns.astype(str)
        linked_sites_z_pvals_dfs[i].to_parquet(out_dir + '/' + name + '_linked_sites_pvals_' + str(PERCENTILES[i]) + '.parquet')

def write_out_results(out_dir, name, result_dfs, linked_sites_names_dfs, linked_sites_diffs_dfs, linked_sites_z_pvals_dfs, nonlinked_sites_names_dfs, nonlinked_sites_diffs_dfs, nonlinked_sites_z_pvals_dfs):
    """
    @ out_dir: path to directory to write out result dfs, linked_sites_names_dfs, and linked_sites_diffs_dfs
    """
    for i in range(len(PERCENTILES)):
        result_dfs[i].to_parquet(out_dir + '/' + name + '_' + str(PERCENTILES[i]) + '.parquet')
        linked_sites_names_dfs[i].columns = linked_sites_names_dfs[i].columns.astype(str)
        linked_sites_names_dfs[i].to_parquet(out_dir + '/' + name + '_linked_sites_names_' + str(PERCENTILES[i]) + '.parquet')
        linked_sites_diffs_dfs[i].columns = linked_sites_diffs_dfs[i].columns.astype(str)
        linked_sites_diffs_dfs[i].to_parquet(out_dir + '/' + name + '_linked_sites_diffs_' + str(PERCENTILES[i]) + '.parquet')
        linked_sites_z_pvals_dfs[i].columns = linked_sites_z_pvals_dfs[i].columns.astype(str)
        linked_sites_z_pvals_dfs[i].to_parquet(out_dir + '/' + name + '_linked_sites_pvals_' + str(PERCENTILES[i]) + '.parquet')
        nonlinked_sites_names_dfs[i].columns = nonlinked_sites_names_dfs[i].columns.astype(str)
        nonlinked_sites_names_dfs[i].to_parquet(out_dir + '/' + name + '_nonlinked_sites_names_' + str(PERCENTILES[i]) + '.parquet')
        nonlinked_sites_diffs_dfs[i].columns = nonlinked_sites_diffs_dfs[i].columns.astype(str)
        nonlinked_sites_diffs_dfs[i].to_parquet(out_dir + '/' + name + '_nonlinked_sites_diffs_' + str(PERCENTILES[i]) + '.parquet')
        nonlinked_sites_z_pvals_dfs[i].columns = nonlinked_sites_z_pvals_dfs[i].columns.astype(str)
        nonlinked_sites_z_pvals_dfs[i].to_parquet(out_dir + '/' + name + '_nonlinked_sites_pvals_' + str(PERCENTILES[i]) + '.parquet')

def get_diff_from_mean(methyl_df_t):
    """
    @ methyl_df_t: dataframe of methylation data
    @ returns: df of samples x sites where each entry is (sample mf - avg methyl frac across samples)
    """
    diff_from_mean_df = methyl_df_t.sub(methyl_df_t.mean(axis=0), axis=1)
    return diff_from_mean_df

def convert_csv_to_parquet(in_fn):
    from pyarrow import csv, parquet
    out_fn = in_fn.split('.')[0] + '.parquet'
    table = csv.read_csv(in_fn, parse_options=csv.ParseOptions(delimiter="\t"))
    parquet.write_table(table, out_fn)

def plot_corr_dist_boxplots(corr_dist_df):
    """
    Plots distance vs correlation boxplots
    @ corr_dist_df: dataframe with 2 columns: dists and corrs
    """
    fig, axes = plt.subplots(figsize=(7,5), dpi=175)
    bin_edges = [0, 10, 10**3, 10**5, 10**7, 10**9]

    boxes = [corr_dist_df[(corr_dist_df['dists'] < bin_edges[i+1] ) & (corr_dist_df['dists'] >= bin_edges[i])]['corrs'] for i in range(len(bin_edges)-1)]
    bp = axes.boxplot(boxes, flierprops=dict(markersize=.1), showfliers=False, labels=[r"$0-10$", r"$10-10^3$", r"$10^3-10^5$", r"$10^5-10^7$", r"$10^7-10^9$"], patch_artist=True, boxprops=dict(facecolor="maroon", alpha=0.7, ))
    # change color of median
    for median in bp['medians']: 
        median.set(color ='black', 
                linewidth = 1)

    axes.set_xlabel("Distance between CpG sites (bp)")
    axes.set_ylabel("Pearson correlation of methylation fraction")


    fig2, axes2 = plt.subplots(figsize=(7,5), dpi=175)
    bin_edges = [0, 10, 10**3, 10**5, 10**7, 10**9]

    # add a new colmn to corr_dist_df that is the distances binned into bin_edges
    corr_dist_df['dists_binned'] = pd.cut(corr_dist_df['dists'], bin_edges, labels=[r"$0-10$", r"$10-10^3$", r"$10^3-10^5$", r"$10^5-10^7$", r"$10^7-10^9$"])

    sns.violinplot(data=corr_dist_df, x='dists_binned', y='corrs', ax=axes2, palette='Reds')


def methylome_pca(all_methyl_df_t, illumina_cpg_locs_df, all_mut_df, num_pcs=5):
    """
    @ all_methyl_df: dataframe of methylation data
    @ returns: pca object
    """
    # import PCA and standard scaler
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # subset all_methyl_df_t to only include cpgs on chromosome 1
    methyl_chr1 = all_methyl_df_t.loc[:,set(illumina_cpg_locs_df[illumina_cpg_locs_df['chr'] == '1']['#id'].values) & set(all_methyl_df_t.columns.values)]
    # scale
    methyl_chr1_scaled = StandardScaler().fit_transform(methyl_chr1)
    # pca
    pca = PCA(n_components=num_pcs)
    methyl_chr1_tranf = pca.fit_transform(methyl_chr1_scaled)
    
    # count c>T mutations for each sample on chr 1, and fill in missing samples with 0
    mut_counts_by_sample = all_mut_df[(all_mut_df.chr == '1') & (all_mut_df.mutation == 'C>T')]['case_submitter_id'].value_counts().reindex(all_methyl_df_t.index.values).fillna(0)
    # put in same order as methyl_chr1
    mut_counts_by_sample = mut_counts_by_sample.loc[set(methyl_chr1.index.values) & set(mut_counts_by_sample.index.values)]
    # measure correlation of each sample projected onto each pc with mut_counts_by_sample
    pc_corrs_w_mut_counts = [np.corrcoef(mut_counts_by_sample, methyl_chr1_tranf[:,i])[0,1] for i in range(num_pcs)]

    fig, axes = plt.subplots(1,2 , figsize=(12,5), dpi=100)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    axes[0].bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    axes[0].set_ylabel('percentange of explained variance')
    axes[0].set_xlabel('principal component')
    # plot correlation of each pc with mut_counts_by_sample
    axes[1].bar(x=range(1, len(pc_corrs_w_mut_counts)+1), height=pc_corrs_w_mut_counts, tick_label=labels)


    # same but with age
    # reindex 
    ages = all_methyl_df_t.loc[all_methyl_df_t.index.isin(mut_counts_by_sample.index), 'age_at_index']
    # put in same order as methyl_chr1
    ages = ages.loc[set(methyl_chr1.index.values) & set(ages.index.values)]

    # measure correlation of each sample projected onto each pc with mut_counts_by_sample
    pc_corrs_w_ages = [np.corrcoef(ages, methyl_chr1_tranf[:,i])[0,1] for i in range(num_pcs)]
    fig, axes = plt.subplots(1,2 , figsize=(12,5), dpi=100)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    axes[0].bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    axes[0].set_ylabel('percentange of explained variance')
    axes[0].set_xlabel('principal component')
    # plot correlation of each pc with mut_counts_by_sample
    axes[1].bar(x=range(1, len(pc_corrs_w_ages)+1), height=pc_corrs_w_ages, tick_label=labels)



    return pca, methyl_chr1_tranf, pc_corrs_w_mut_counts

def add_ages_to_mut_and_methyl(mut_df, all_meta_df, all_methyl_df_t):
    to_join_mut_df = mut_df.rename(columns={'sample':'case_submitter_id'})
    mut_w_methyl_age_df =  to_join_mut_df.join(all_meta_df, on =['case_submitter_id'], rsuffix='_r',how='inner')
    # join ages with methylation
    all_methyl_age_df_t = all_meta_df.join(all_methyl_df_t, on =['sample'], rsuffix='_r',how='inner')
    return mut_w_methyl_age_df, all_methyl_age_df_t

def get_same_age_and_tissue_samples(methyl_age_df_t, mut_sample_name, age_bin_size = 10):
    """
    Get the sample that has the mutation in the mutated CpG and the samples of the same age as that sample
    @ methyl_age_df_t: dataframe with columns=CpGs and rows=samples and entries=methylation fraction
    @ age_bin_size: size of age bins to use (will be age_bin_size/2 on either side of the mutated sample's age)
    @ returns: the methylation fraction for samples of the same age and dset as the mutated sample
    """
    # get this sample's age
    this_age = methyl_age_df_t.loc[mut_sample_name, 'age_at_index']
    this_dset = methyl_age_df_t.loc[mut_sample_name, 'dataset']
    # get the mf all other samples of within age_bin_size/2 years of age on either side
    same_age_dset_samples_mf_df = methyl_age_df_t[(np.abs(methyl_age_df_t['age_at_index'] - this_age) <= age_bin_size/2) & (methyl_age_df_t['dataset'] == this_dset)]
    # drop the mutated sample
    same_age_dset_samples_mf_df = same_age_dset_samples_mf_df.drop(index = mut_sample_name)
    return same_age_dset_samples_mf_df

def stack_and_merge(diffs_df, pvals_df, names_df = None):
    """
    Take one diffs and one pvals df, stack them and merge
    """
    # stack the dfs
    diffs_df = diffs_df.stack().reset_index()
    pvals_df = pvals_df.stack().reset_index()
    if type(names_df) != type(None):
        names_df = names_df.stack().reset_index()
    # rename the columns
    diffs_df.columns = ['mut_site', 'comparison_site', 'delta_mf']
    pvals_df.columns = ['mut_site', 'comparison_site', 'pval']
    if type(names_df) != type(None):
        names_df.columns = ['mut_site', 'comparison_site', 'linked_site']
    # set comparison site columns to be dytpe = int
    diffs_df['comparison_site'] = diffs_df['comparison_site'].astype(int)
    pvals_df['comparison_site'] = pvals_df['comparison_site'].astype(int)
    if type(names_df) != type(None):
        names_df['comparison_site'] = names_df['comparison_site'].astype(int)
    # and mut site columns to be dytpe = str
    diffs_df['mut_site'] = diffs_df['mut_site'].astype(str)
    pvals_df['mut_site'] = pvals_df['mut_site'].astype(str)
    if type(names_df) != type(None):
        names_df['mut_site'] = names_df['mut_site'].astype(str)
    # merge
    if type(names_df) == type(None):
        merged_df = pd.merge(diffs_df, pvals_df, on=['comparison_site', 'mut_site'])
    else:
        merged_df = pd.merge(diffs_df, pvals_df, on=['comparison_site', 'mut_site'])
        merged_df = pd.merge(merged_df, names_df, on=['comparison_site', 'mut_site'])
    return merged_df

def half(l, which_half):
    if which_half == 'first':
        return l[:int(len(l)/2)]
    else:
        return l[int(len(l)/2):]

def fdr_correct(df, pval_col_name = 'ztest_pval'):
    df = df.dropna(subset=[pval_col_name])
    df['sig'], df['fdr_pval'] = fdrcorrection(df.loc[:, pval_col_name], alpha=0.05)
    return df

def fdr_correct_split(df, pval_col_name = 'ztest_pval', split_col = 'mutated'):
    df.dropna(subset=[pval_col_name], inplace=True)
    # split on split_col
    df1 = df[df[split_col] == True]
    df2 = df[df[split_col] == False]
    # correct
    df1.loc[:, 'sig'], df1.loc[:, 'fdr_pval'] = fdrcorrection(df1.loc[:, pval_col_name], alpha=0.05)
    df2.loc[:, 'sig'], df2.loc[:, 'fdr_pval'] = fdrcorrection(df2.loc[:, pval_col_name], alpha=0.05)
    # merge on index
    df = pd.concat([df1, df2])
    return df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-deep")
import os 
from scipy import stats
import statsmodels.api as sm
import sys
from collections import defaultdict
import seaborn as sns


# CONSTANTS
VALID_MUTATIONS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G", "G>C","G>A", "A>T", "A>G" , "A>C", "G>T", "C>-"]
JUST_CT = True
DATA_SET = "TCGA"
PERCENTILES = [1]#np.flip(np.linspace(0, 1, 11))


def get_percentiles():
    return PERCENTILES

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

def plot_mutations_distributions(all_mut_df, out_dir, illumina_cpg_locs_df, all_methyl_df_t):
    """
    For each dataset plot distribtion of mutation types, also for just mutations in illumina measured CpG sites
    """
    # plot distribution of mutation type all together
    fig, axes = plt.subplots(figsize=(7,6), facecolor="white")
    axes = all_mut_df['mutation'].value_counts().plot.bar(ax=axes, xlabel="Mutation in any sample", color='steelblue', alpha=0.7, ylabel="Count")
    fig.savefig(os.path.join(out_dir, 'mut_type_count_all_datasets.png'))
    # plot distribution of just mutations in measured CpG sites
    fig2, axes2 = plt.subplots(figsize=(7,6), facecolor="white")
    mut_in_measured_cpg_df = join_df_with_illum_cpg(all_mut_df, illumina_cpg_locs_df, all_methyl_df_t)
    axes2 = mut_in_measured_cpg_df['mutation'].value_counts().plot.bar(ax=axes2, xlabel="Mutation in any sample in measured CpG", color='steelblue', alpha=0.7,ylabel="Count")
    fig2.savefig(os.path.join(out_dir, 'mut_type_count_in_measured_cpg_datasets.png'))
    return mut_in_measured_cpg_df

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

def test_sig(results_dfs):
    """
    @ returns: dict of effect sizes for each signfiicant result and metrics
    """
    # initialize dict of lists
    result_metrics_dict = defaultdict(list)

    for i in range(len(PERCENTILES)):
        this_result_df = results_dfs[i]
        bonf_p_val = 0.05/len(this_result_df)
        result_metrics_dict['m_avg_err_p'].append(len(this_result_df[this_result_df['p_mean_avg_err'] < bonf_p_val]))
        result_metrics_dict['m_avg_errs_eff'].append(this_result_df[this_result_df['p_mean_avg_err'] < bonf_p_val]['linked_minus_non_delta_mf'].mean())
        result_metrics_dict['m_linked_mean_avg_err'].append(this_result_df[this_result_df['p_mean_avg_err'] < bonf_p_val]['linked_delta_mf'].mean())
        result_metrics_dict['m_non_linked_mean_avg_err'].append(this_result_df[this_result_df['p_mean_avg_err'] < bonf_p_val]['non_linked_delta_mf'].mean())
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
    result_dfs = []
    linked_sites_names_dfs = []
    linked_sites_diffs_dfs = []
    for i in range(len(PERCENTILES)):
        result_dfs.append(pd.read_parquet(result_base_path + '_' + str(PERCENTILES[i]) + '.parquet'))
        linked_sites_names_dfs.append(pd.read_parquet(result_base_path + '_linked_sites_names_' + str(PERCENTILES[i]) + '.parquet'))
        linked_sites_diffs_dfs.append(pd.read_parquet(result_base_path + '_linked_sites_diffs_' + str(PERCENTILES[i]) + '.parquet'))
    return result_dfs, linked_sites_names_dfs, linked_sites_diffs_dfs

def write_out_results(out_dir, name, result_dfs, linked_sites_names_dfs, linked_sites_diffs_dfs):
    """
    @ out_dir: path to directory to write out result dfs, linked_sites_names_dfs, and linked_sites_diffs_dfs
    """
    for i in range(len(PERCENTILES)):
        result_dfs[i].to_parquet(out_dir + '/' + name + '_' + str(PERCENTILES[i]) + '.parquet')
        linked_sites_names_dfs[i].columns = linked_sites_names_dfs[i].columns.astype(str)
        linked_sites_names_dfs[i].to_parquet(out_dir + '/' + name + '_linked_sites_names_' + str(PERCENTILES[i]) + '.parquet')
        linked_sites_diffs_dfs[i].columns = linked_sites_diffs_dfs[i].columns.astype(str)
        linked_sites_diffs_dfs[i].to_parquet(out_dir + '/' + name + '_linked_sites_diffs_' + str(PERCENTILES[i]) + '.parquet')

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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import ticker as mticker
from matplotlib.patches import Rectangle
import seaborn as sns
from statsmodels.stats.weightstats import ztest as ztest

import utils

PERCENTILES = [1]#np.flip(np.linspace(0, 1, 6))


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
    fig, axes = plt.subplots(figsize=(10, 4), dpi=100)
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


def plot_eff_violin(tested_mut_in_measured_cpg_w_methyl_age_df, linked_sites_pvals_dfs, linked_sites_diffs_dfs, sig_thresh=1, mean = False, absolut = False):
    """
    Plot the violin plots for mean avg err
    @ result_dfs: list of dataframes, each containing the results for a different percentile
    @ mut_in_measured_cpg_w_methyl_age_df: dataframe containing the methylation and mutation data for the sites that were tested
    @ mut_linkage_df: dataframe containing the linkage data for the tested sites
    @ linkage_type: string, the type of linkage used
    """
    # make a df with columns: linkage percentile, delta_mf, pval, and linkage_status
    to_plot_dfs = []
    if mean:
        for i in range(len(PERCENTILES)):
            # get mean delta_mf for each mutation in linked and nonlinked sites
            if absolut:
                mean_linked_diff = np.abs(linked_sites_diffs_dfs[i]).mean(axis=1)
            else:
                mean_linked_diff = linked_sites_diffs_dfs[i].mean(axis=1)
            to_plot_dict = {'delta_mf': mean_linked_diff.to_list(), 
                                'Linkage percentile': [int(round(PERCENTILES[i], 1)*100) for j in range(len(linked_sites_diffs_dfs[i]))],
                                 'Linkage status': ["Linked CpGs" for j in range(len(linked_sites_diffs_dfs[i]))]}
            to_plot_df = pd.DataFrame(to_plot_dict)
            to_plot_dfs.append(to_plot_df)
        # concat all the dfs
        to_plot_df = pd.concat(to_plot_dfs, axis=0)
    else:
        for i in range(len(PERCENTILES)):
            linked_diffs_pvals = utils.stack_and_merge(linked_sites_diffs_dfs[i], linked_sites_pvals_dfs[i])
            # fdr correct
            linked_diffs_pvals = utils.fdr_correct(linked_diffs_pvals, 'pval')
            # add a column for the linkage percentile
            linked_diffs_pvals['Linkage percentile'] = int(round(PERCENTILES[i], 1)*100)
            # add a column for the linkage status
            linked_diffs_pvals['Linkage status'] = ["Linked CpGs" for _ in range(len(linked_diffs_pvals))]
            to_plot_dfs.append(linked_diffs_pvals)
        # concat all the dfs
        to_plot_df = pd.concat(to_plot_dfs, axis=0)
        # subset to only the significant sites
        to_plot_df = to_plot_df[to_plot_df['fdr_pval'] <= sig_thresh]

    # violin plot of the mean avg err for each linkage percentile
    _, axes = plt.subplots(1, 2, figsize=(20, 5) if len(PERCENTILES) > 1 else (10, 5), gridspec_kw={'width_ratios': [1, 6]}, sharey=True, constrained_layout=True)
    my_pal = {"Linked CpGs": "steelblue", "Non-linked CpGs": "skyblue"}
    p = sns.violinplot(x="Linkage percentile", y="delta_mf", data=to_plot_df, scale="area", ax=axes[1], cut=0, linewidth=2)
    p.set_ylabel(r"$\Delta$MF")
    axes[1].invert_xaxis()
    # add a % sign after each x tick label
    axes[1].set_xticklabels(reversed([str(int(round(i, 1)*100))+'%' for i in PERCENTILES]))
    # move legend to the bottom right
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[::-1], labels[::-1], loc='lower right')

    # violin plot of the mean avg err for all mutated sites
    mut_mf_change = tested_mut_in_measured_cpg_w_methyl_age_df['difference'].reset_index(drop=True)
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

def plot_heatmap_corr(mut_site, linked_sites_names_df, nonlinked_sites_names_df, mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t, age_bin_size=10):
    """
    Given a set of linked sites, nonlinked sites, mutated sample, and mutated site, plots a heatmap of the methylation fraction of same age samples at the linked, nonlinked, and mutated sites
    @ mut_site: name of mutated site
    @ linked_sites_names_df: dataframe of linked sites names
    @ nonlinked_sites_names_df: dataframe of nonlinked sites names
    @ mut_in_measured_cpg_w_methyl_age_df: dataframe of mutations in samples
    @ all_methyl_age_df_t: dataframe of methylation data with ages attached
    """

    # get the MFs of the same age samples, find which sample had the mutation, and the dataset of this sample
    mut_sample, same_age_dset_samples_mf_df = utils.get_same_age_and_tissue_samples(all_methyl_age_df_t, 
                                                                                mut_in_measured_cpg_w_methyl_age_df,
                                                                                age_bin_size,
                                                                                mut_site)
    mut_sample = mut_sample.case_submitter_id.to_numpy()[0]
    # get list of same age AND same dataset samples
    same_age_tissue_samples = same_age_dset_samples_mf_df.index

    # get the names of the linked sites and nonlinked sites
    linked_sites = linked_sites_names_df.loc[mut_site].to_numpy()
    nonlinked_sites = nonlinked_sites_names_df.loc[mut_site].to_numpy()

    # list of samples to plot
    samples_to_plot = np.concatenate((utils.half(same_age_tissue_samples, 'first'), [mut_sample], utils.half(same_age_tissue_samples, 'second')))
    # list of sites to plot
    sites_to_plot = np.concatenate((utils.half(nonlinked_sites, 'first'), utils.half(linked_sites, 'first'), [mut_site], utils.half(linked_sites, 'second'), utils.half(nonlinked_sites, 'second')))
    # select cpgs and samples to plot
    to_plot_df = all_methyl_age_df_t.loc[samples_to_plot, sites_to_plot]

    _, axes = plt.subplots(figsize=(15,10))
    ax = sns.heatmap(to_plot_df, annot=False, center=0.5, xticklabels=False, yticklabels=False, cmap="Blues", cbar_kws={'label': 'Methylation fraction'}, ax=axes)
    # highlight the mutated cpg
    ax.add_patch(Rectangle((int(len(linked_sites)/2) + int(len(nonlinked_sites)/2), int(len(same_age_tissue_samples)/2)), 1, 1, fill=False, edgecolor='red', lw=1.5))
    axes.set_xlabel("Linked CpG sites")
    axes.set_ylabel("Samples (same age as mutated sample)")
    # seperate linked and nonlinked sites with vertical lines
    axes.axvline(x=int(len(nonlinked_sites)/2), color='red', linestyle='-')
    axes.axvline(x=int(len(nonlinked_sites)/2) + len(linked_sites), color='red', linestyle='-')

    return
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-deep")
#from sklearnex import patch_sklearn
#patch_sklearn()
from scipy import stats
import sys
import os 
import seaborn as sns
from matplotlib.ticker import PercentFormatter


# CONSTANTS
VALID_MUTATIONS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G", "G>C","G>A", "A>T", "A>G" , "A>C", "G>T", "C>-"]
JUST_CT = True
DATA_SET = "TCGA"

import utils

def plot_mutations_distributions(all_mut_df, out_dir, illumina_cpg_locs_df, all_methyl_df_t):
    """
    For each dataset plot distribtion of mutation types, also for just mutations in illumina measured CpG sites
    """
    # plot distribution of mutation type all together
    fig, axes = plt.subplots(figsize=(7,6), facecolor="white")
    axes = (all_mut_df['mutation'].value_counts() / len(all_mut_df['mutation'])).plot.bar(ax=axes, xlabel="Mutation type", color='maroon', alpha=0.7, ylabel="Pan-cancer frequency of mutation")
    fig.savefig(os.path.join(out_dir, 'mut_type_count_all_datasets.png'))
    # plot distribution of just mutations in measured CpG sites
    fig2, axes2 = plt.subplots(figsize=(7,6), facecolor="white")
    mut_in_measured_cpg_df = utils.join_df_with_illum_cpg(all_mut_df, illumina_cpg_locs_df, all_methyl_df_t)
    axes2 = (mut_in_measured_cpg_df['mutation'].value_counts() / len(mut_in_measured_cpg_df['mutation'])).plot.bar(ax=axes2, xlabel="Mutations in measured CpG sites", color='maroon', alpha=0.7,ylabel="Pan-cancer frequency of mutation")
    fig2.savefig(os.path.join(out_dir, 'mut_type_count_in_measured_cpg_datasets.png'))
    return mut_in_measured_cpg_df

def plot_mutations(all_mut_df, all_meta_df, dataset_names_list, out_dir, illumina_cpg_locs_df, all_methyl_df_t):
    """
    @ all_mut_df: pandas dataframe of all mutations
    @ all_meta_df: pandas dataframe of all metadata
    @ dataset_names_list: list of dataset names
    @ out_dir: output directory
    @ illumina_cpg_locs_df: pandas dataframe of illumina cpg locations
    @ all_methyl_df_t: pandas dataframe of all methylation, processed 
    @ returns: pandas dataframe of all mutations in illumina measured CpG sites
    """
    # for each dataset plot distribtion of all mutations, mutations in CpG sites, and C>T mutations in CpG sites
    mut_in_measured_cpg_df = plot_mutations_distributions(all_mut_df, out_dir, illumina_cpg_locs_df, all_methyl_df_t)
    # for each dataset plot # of C>T mutations by age
    # TODO: fix plot_mutation_count_by_age to work with PANCAN
    """utils.plot_mutation_count_by_age(all_mut_df, all_meta_df, dataset_names_list, out_dir)"""
    return mut_in_measured_cpg_df

def compare_mf_mutated_sample_vs_avg(mutation_in_measured_cpg_df, out_dir, all_methyl_df_t, dataset="TCGA"):
    """
    Plot MF at sites of mutation event vs same site with no mutation. Write pvals of testing difference of distribution between mutated and not 
    """
    # output plots of avg vs not as seaborn kde's
    fig, axes = plt.subplots(dpi=200)
    non_mutated_methyl_df_t = all_methyl_df_t[all_methyl_df_t.columns[~all_methyl_df_t.columns.isin(mutation_in_measured_cpg_df['#id'])]]
    
    # limit to only big DNA_VAF mutations
    mutation_in_measured_cpg_df = mutation_in_measured_cpg_df[mutation_in_measured_cpg_df['DNA_VAF'] >.4]
    to_plot_df = pd.DataFrame(pd.concat([mutation_in_measured_cpg_df['avg_methyl_frac'], mutation_in_measured_cpg_df['methyl_fraction'], non_mutated_methyl_df_t.mean(axis=0)], axis=0)).reset_index(drop=True)
    to_plot_df.columns = ['Methylation Fraction']
    to_plot_df['Type'] = ['Non mutated CpGs'] * len(mutation_in_measured_cpg_df['avg_methyl_frac']) +  ['Mutated CpGs'] * len(mutation_in_measured_cpg_df['methyl_fraction']) + ['Site of no CpG mutation'] * len(non_mutated_methyl_df_t.mean(axis=0))
    # seaborn kde plot
    p = sns.kdeplot(
        data=to_plot_df, x='Methylation Fraction', hue='Type', fill=True,
        common_norm=False, clip=[0,1], palette = ['steelblue', 'maroon', 'grey'], 
        ax=axes, legend=False
        )

    fig.savefig(os.path.join(out_dir, '{}_methylation_fraction_comparison.png'.format(dataset)))
    # seperately save kde of just mutated, just non mutated, and just site of no CpG mutation
    # set transparent background
    fig2, axes2 = plt.subplots(dpi=200)
    p2 = sns.kdeplot(
        data=to_plot_df[to_plot_df['Type'] == 'Non mutated CpGs'], 
        x='Methylation Fraction', fill=True, common_norm=False, 
        clip=[0,1], palette = ['grey'], ax=axes2, legend=False
        )
    
    # hexbin plot of mutation_in_measured_cpg_df['methyl_fraction'] vs mutation_in_measured_cpg_df['avg_methyl_frac']
    fig, axes = plt.subplots(dpi=200)
    p3 = axes.hexbin(mutation_in_measured_cpg_df['avg_methyl_frac'], mutation_in_measured_cpg_df['methyl_fraction'], gridsize=50, bins='log', cmap='inferno')
    axes.set_xlabel('Average MF of non-mutated individuals')
    axes.set_ylabel('Average MF of mutated individuals')
    # add a y=x line to the plot
    axes.plot([0,1], [0,1], transform=axes.transAxes, color='black')
    # add colorbar
    cbar = fig.colorbar(p3)
    # label colorbar
    cbar.set_label('Count of CpG sites', rotation=270, labelpad=15)

    # write pvals and effect sizes to file
    with open(os.path.join(out_dir, "{}_methylation_fraction_results.txt".format(dataset)), "w+") as f:
        if JUST_CT:
            f.write("Difference in methylation fraction between C>T mutated samples and not, at same CpG\n")
        else:
            f.write("Difference in methylation fraction between C>T/G>A mutated samples and not, at same CpG\n")
        f.write("Number of CpGs tested: {}\n".format(len(mutation_in_measured_cpg_df)))
        f.write("Effect size: {}\n".format( mutation_in_measured_cpg_df.mean()))
        statistic, wilc_p_val = stats.ranksums(mutation_in_measured_cpg_df['methyl_fraction'].to_numpy(), mutation_in_measured_cpg_df['avg_methyl_frac'].to_numpy(), alternative='less')
        f.write("Wilcoxon rank sum p-value {}\n".format(wilc_p_val))
        statistic, p_val = stats.mannwhitneyu(mutation_in_measured_cpg_df['methyl_fraction'].to_numpy(), mutation_in_measured_cpg_df['avg_methyl_frac'].to_numpy(), alternative='less', method='auto')
        f.write("MannWhitney U p-value {}\n".format(p_val))
        result = stats.binomtest(len(mutation_in_measured_cpg_df[mutation_in_measured_cpg_df['difference']<0]), len(mutation_in_measured_cpg_df), p=0.5, alternative='greater')
        f.write("Binomial test of greater than p=0.5 p-value {}\n".format(result.pvalue))
    # barplot 
    fig, axes = plt.subplots(facecolor="white", dpi=200)
    num_less_zero = len(mutation_in_measured_cpg_df[mutation_in_measured_cpg_df['difference']<0])
    num_greater_zero = len(mutation_in_measured_cpg_df[mutation_in_measured_cpg_df['difference']>0])
    axes.bar( x= ['Decrease', 'Increase'], color = ['darkgrey', 'lightgrey'], edgecolor='black', linewidth=2, height = [num_less_zero, num_greater_zero] )
    axes.set_xlabel("Change in MF at site of C>T mutation")
    axes.set_ylabel("Number of sites")
    fig.savefig(os.path.join(out_dir, '{}_methylation_fraction_difference_hist.png'.format(dataset)))
    return

def compare_mf_site_of_mutation_vs_not(mutation_in_measured_cpg_df, all_methyl_df_t, out_dir):
    # test for difference between average methylation fraction at non-mutated CpG sites and at mutated CpG sites in non-mutated samples
    non_mutated_methyl_df_t = all_methyl_df_t[all_methyl_df_t.columns[~all_methyl_df_t.columns.isin(mutation_in_measured_cpg_df['#id'])]]
    with open(os.path.join(out_dir, "methylation_fraction_results.txt"), "a+") as f:
        statistic, p_val = stats.ranksums(non_mutated_methyl_df_t.mean().to_numpy(), mutation_in_measured_cpg_df['avg_methyl_frac'].to_numpy(), alternative='less')
        f.write("Wilcoxon rank sum p-value testing if the dsitr. of average methylation fraction at non-mutated CpG sites is lesser than at mutated CpG sites in non-mutated samples {} and statistic {}\n".format(p_val, statistic))
        f.write("mean average methylation fraction at non-mutated {} mutated CpG sites in non-mutated samples {}".format(non_mutated_methyl_df_t.mean().mean(),mutation_in_measured_cpg_df['avg_methyl_frac'].mean() ))
        # fisher
        non_mut_less = len(non_mutated_methyl_df_t.mean()[non_mutated_methyl_df_t.mean()<=.5])
        mut_loc_less = len(mutation_in_measured_cpg_df['avg_methyl_frac'] <= .5)
        non_mut_greater = len(non_mutated_methyl_df_t.mean()[non_mutated_methyl_df_t.mean()>.5])
        mut_loc_greater = len(mutation_in_measured_cpg_df['avg_methyl_frac']>.5)
        contingency_table = [[non_mut_less, mut_loc_less],[non_mut_greater, mut_loc_greater]]
        oddsr, p = stats.fisher_exact(table=contingency_table, alternative='less')
        f.write("Fisher p-value for dsitr. of average methylation fraction at non-mutated CpG sites has greater proportion <.5 than at mutated CpG sites in non-mutated samples: {}".format(p))

    # histogram of average methylation fraction at sites with C>T mutation vs without
    fig, axes = plt.subplots(facecolor="white", dpi=200)
    weights = np.ones_like(mutation_in_measured_cpg_df['avg_methyl_frac']) / len(mutation_in_measured_cpg_df['avg_methyl_frac'])
    mutation_in_measured_cpg_df['avg_methyl_frac'].plot.hist(weights=weights,bins=12, ax = axes,alpha=.7, color = 'goldenrod')
    weights = np.ones_like(non_mutated_methyl_df_t.loc['mean']) / len(non_mutated_methyl_df_t.loc['mean'])
    non_mutated_methyl_df_t.loc['mean'].plot.hist(weights = weights,bins=12, ax = axes, alpha=.7, color='dimgray')
    axes.legend(["Sites of C>T mutation events\n(including mutated samples)", "Sites of no C>T mutation\n event"])
    axes.set_ylabel("Probability")
    axes.set_xlabel("Mean methylation fraction")
    fig.savefig(os.path.join(out_dir, 'non_mut_vs_mut_site_mf.png'))

def get_same_age_tissue_means(mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t, age_bin_size = 10):
    """
    Get the average MF at the mutated sites in the same age bin range and tissue type as the sample where the mutation occured
    """
    # for each mutation in mut_in_measured_cpg_w_methyl_age_df, get the average MF at the same CpG site in the same age bin range and tissue type as the sample where the mutation occured
    # get the average MF at the same CpG site in the same age bin range and tissue type as the sample where the mutation occured

    mut_in_measured_cpg_w_methyl_age_df['avg_methyl_frac'] = mut_in_measured_cpg_w_methyl_age_df.apply(lambda mut_row: all_methyl_age_df_t[(np.abs(all_methyl_age_df_t['age_at_index'] - mut_row['age_at_index']) <= age_bin_size/2) & (all_methyl_age_df_t['dataset'] == mut_row['dataset'])][mut_row['#id']].mean(), axis=1)
    return mut_in_measured_cpg_w_methyl_age_df

def methylation_fraction_comparison(
    all_mut_df, illumina_cpg_locs_df, all_methyl_df_t,
    out_dir, all_meta_df, age_bin_size = 10
    ):
    """
    Measure the effect a mutation has on MF at that site
    @ returns: pandas dataframe of all mutations in illumina measured CpG sites, their methylation fraction in mutated sample, and average methylation at that site across other samples (within 5 years of age)
    """
    # get just C>T mutations
    if JUST_CT:
        mutations_df = all_mut_df[all_mut_df['mutation'] == 'C>T']
    else:
        mutations_df = all_mut_df
    #for each CpG with methylation data get genomic location of its C
    mut_in_measured_cpg_df = utils.join_df_with_illum_cpg(mutations_df, illumina_cpg_locs_df, all_methyl_df_t)
    methyl_fractions = utils.get_methyl_fractions(mut_in_measured_cpg_df, all_methyl_df_t)
    mut_in_measured_cpg_df['methyl_fraction'] = methyl_fractions

    # add ages and datasets to both dfs
    mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t = utils.add_ages_to_mut_and_methyl(mut_in_measured_cpg_df, all_meta_df, all_methyl_df_t)

    # get rid of samples that do not have methylayion data
    mut_in_measured_cpg_w_methyl_age_df = mut_in_measured_cpg_w_methyl_age_df[mut_in_measured_cpg_w_methyl_age_df['methyl_fraction'] != -1]
    # drop samples with nan age value
    mut_in_measured_cpg_w_methyl_age_df = mut_in_measured_cpg_w_methyl_age_df.dropna(subset=['age_at_index'])
    
    # get means 
    mut_in_measured_cpg_w_methyl_age_df = get_same_age_tissue_means(
        mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t, age_bin_size = age_bin_size
        )
    """# old way
    mutation_in_measured_cpg_df['avg_methyl_frac'] = all_methyl_df_t[mutation_in_measured_cpg_df['#id']].mean().values"""

    # get difference between mean and mutated sample
    mut_in_measured_cpg_w_methyl_age_df['difference'] = mut_in_measured_cpg_w_methyl_age_df['methyl_fraction'] - mut_in_measured_cpg_w_methyl_age_df['avg_methyl_frac']
    # test for a difference
    compare_mf_mutated_sample_vs_avg(mut_in_measured_cpg_w_methyl_age_df, out_dir, all_methyl_df_t)
    """compare_mf_site_of_mutation_vs_not(mut_in_measured_cpg_w_methyl_age_df, all_methyl_df_t, out_dir)"""
    return mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t

def main(illumina_cpg_locs_df, out_dir, all_mut_df, all_methyl_df_t, all_meta_df, dataset_names_list, age_bin_size):
    # make output directories
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "bootstrap"), exist_ok=True)

    # do mutation analysis 
    #mut_in_measured_cpg_df = plot_mutations(all_mut_df, all_meta_df, dataset_names_list, out_dir, illumina_cpg_locs_df, all_methyl_df_t)
    # subset to only C>T mutations
    # TODO: remove this return if possible
    
    mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t = methylation_fraction_comparison(all_mut_df, illumina_cpg_locs_df, all_methyl_df_t, out_dir, all_meta_df, age_bin_size = age_bin_size)

    #return mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t
    
    

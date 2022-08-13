import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-deep")
#from sklearnex import patch_sklearn
#patch_sklearn()
from scipy import stats
import sys
import os 

# CONSTANTS
VALID_MUTATIONS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G", "G>C","G>A", "A>T", "A>G" , "A>C", "G>T", "C>-"]
JUST_CT = True
DATA_SET = "TCGA"

import get_data
import utils

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
    mut_in_measured_cpg_df = utils.plot_mutations_distributions(all_mut_df, out_dir, illumina_cpg_locs_df, all_methyl_df_t)
    # for each dataset plot # of C>T mutations by age
    utils.plot_mutation_count_by_age(all_mut_df, all_meta_df, dataset_names_list, out_dir)
    return mut_in_measured_cpg_df

def compare_mf_mutated_sample_vs_avg(ct_mutation_in_measured_cpg_df, out_dir, dataset="TCGA"):
    """
    Plot MF at sites of mutation event vs same site with no mutation. Write pvals of testing difference of distribution between mutated and not 
    """
    # output plots of avg vs not
    # histograms
    fig, axes = plt.subplots()
    ct_mutation_in_measured_cpg_df[['avg_methyl_frac', 'methyl_fraction']].plot.hist(bins=12, alpha=.7, color=['maroon', 'steelblue'], ax = axes)
    axes.legend(["Mean of non-mutated samples at\nCpGs with >=1 mutation event", "C>T mutation events" ])
    axes.set_ylabel('Count')
    axes.set_xlabel('Methylation fraction')
    fig.savefig(os.path.join(out_dir, '{}_methylation_fraction_comparison.png'.format(dataset)))

    # plot average by itself as well
    fig, axes = plt.subplots(facecolor="white")
    ct_mutation_in_measured_cpg_df['avg_methyl_frac'].plot.hist(ax = axes, bins=12,  color='steelblue', alpha=.7)
    axes.set_xlabel("Mean of non-mutated samples at sites of C>T mutation event")
    axes.set_ylabel("Count")
    fig.savefig(os.path.join(out_dir, '{}_avg_methylation_fraction_non_mutated_at_mut_site.png'.format(dataset)))

    fig, axes = plt.subplots(facecolor="white")
    num_less_zero = len(ct_mutation_in_measured_cpg_df[ct_mutation_in_measured_cpg_df['difference']<0])
    num_greater_zero = len(ct_mutation_in_measured_cpg_df[ct_mutation_in_measured_cpg_df['difference']>0])
    axes.bar( x= ["Difference < 0", "Difference > 0"], color= ['maroon', 'steelblue'], alpha=.7, height = [num_less_zero, num_greater_zero] )
    axes.set_ylabel("Count")
    if JUST_CT:
        axes.set_xlabel('C>T mutation event MF - mean MF of  non-mutated samples\nat same CpG site')
    else:
        axes.set_xlabel('Methylation fraction difference at C>T/G>A sites in mutated - non-mutated samples')
    fig.savefig(os.path.join(out_dir, '{}_methylation_fraction_difference_hist.png'.format(dataset)))
    # write pvals and effect sizes to file
    with open(os.path.join(out_dir, "{}_methylation_fraction_results.txt".format(dataset)), "w+") as f:
        if JUST_CT:
            f.write("Difference in methylation fraction between C>T mutated samples and not, at same CpG\n")
        else:
            f.write("Difference in methylation fraction between C>T/G>A mutated samples and not, at same CpG\n")
        f.write("Number of CpGs tested: {}\n".format(len(ct_mutation_in_measured_cpg_df)))
        f.write("Effect size: {}\n".format( ct_mutation_in_measured_cpg_df.mean()))
        statistic, p_val = stats.ranksums(ct_mutation_in_measured_cpg_df['methyl_fraction'].to_numpy(), ct_mutation_in_measured_cpg_df['avg_methyl_frac'].to_numpy(), alternative='less')
        f.write("Wilcoxon rank sum p-value {}\n".format(p_val))
        statistic, p_val = stats.mannwhitneyu(ct_mutation_in_measured_cpg_df['methyl_fraction'].to_numpy(), ct_mutation_in_measured_cpg_df['avg_methyl_frac'].to_numpy(), alternative='less', method='auto')
        f.write("MannWhitney U p-value {}\n".format(p_val))
        result = stats.binomtest(len(ct_mutation_in_measured_cpg_df[ct_mutation_in_measured_cpg_df['difference']<0]), len(ct_mutation_in_measured_cpg_df), p=0.5, alternative='greater')
        f.write("Binomial test of greater than p=0.5 p-value {}\n".format(result.pvalue))
    return

def compare_mf_site_of_mutation_vs_not(ct_mutation_in_measured_cpg_df, all_methyl_df_t, out_dir):
    # test for difference between average methylation fraction at non-mutated CpG sites and at mutated CpG sites in non-mutated samples
    non_mutated_methyl_df_t = all_methyl_df_t[all_methyl_df_t.columns[~all_methyl_df_t.columns.isin(ct_mutation_in_measured_cpg_df['#id'])]]
    with open(os.path.join(out_dir, "methylation_fraction_results.txt"), "a+") as f:
        statistic, p_val = stats.ranksums(non_mutated_methyl_df_t.mean().to_numpy(), ct_mutation_in_measured_cpg_df['avg_methyl_frac'].to_numpy(), alternative='less')
        f.write("Wilcoxon rank sum p-value testing if the dsitr. of average methylation fraction at non-mutated CpG sites is lesser than at mutated CpG sites in non-mutated samples {} and statistic {}\n".format(p_val, statistic))
        f.write("mean average methylation fraction at non-mutated {} mutated CpG sites in non-mutated samples {}".format(non_mutated_methyl_df_t.mean().mean(),ct_mutation_in_measured_cpg_df['avg_methyl_frac'].mean() ))
        # fisher
        non_mut_less = len(non_mutated_methyl_df_t.mean()[non_mutated_methyl_df_t.mean()<=.5])
        mut_loc_less = len(ct_mutation_in_measured_cpg_df['avg_methyl_frac'] <= .5)
        non_mut_greater = len(non_mutated_methyl_df_t.mean()[non_mutated_methyl_df_t.mean()>.5])
        mut_loc_greater = len(ct_mutation_in_measured_cpg_df['avg_methyl_frac']>.5)
        contingency_table = [[non_mut_less, mut_loc_less],[non_mut_greater, mut_loc_greater]]
        oddsr, p = stats.fisher_exact(table=contingency_table, alternative='less')
        f.write("Fisher p-value for dsitr. of average methylation fraction at non-mutated CpG sites has greater proportion <.5 than at mutated CpG sites in non-mutated samples: {}".format(p))
    # plot
    fig, axes = plt.subplots(facecolor="white")
    non_mutated_methyl_df_t.loc['mean'] = non_mutated_methyl_df_t.mean()
    non_mutated_methyl_df_t.loc['mean'].plot.hist(ax = axes, color= 'steelblue',alpha=0.7, bins=12)
    axes.set_ylabel("Count")
    axes.set_xlabel("Average methylation fraction at CpG sites with no C>T mutation")
    fig.savefig(os.path.join(out_dir, 'avg_methylation_fraction_non_ct_mut_sites.png'))

    fig, axes = plt.subplots(facecolor="white")
    weights = np.ones_like(ct_mutation_in_measured_cpg_df['avg_methyl_frac']) / len(ct_mutation_in_measured_cpg_df['avg_methyl_frac'])
    ct_mutation_in_measured_cpg_df['avg_methyl_frac'].plot.hist(weights=weights,bins=12, ax = axes,alpha=.7, color = 'goldenrod')
    weights = np.ones_like(non_mutated_methyl_df_t.loc['mean']) / len(non_mutated_methyl_df_t.loc['mean'])
    non_mutated_methyl_df_t.loc['mean'].plot.hist(weights = weights,bins=12, ax = axes, alpha=.7, color='dimgray')
    axes.legend(["Sites of C>T\nmutation event", "Sites of no C>T mutation\n event"])
    axes.set_ylabel("Probability")
    axes.set_xlabel("Mean methylation fraction")
    fig.savefig(os.path.join(out_dir, 'non_mut_vs_mut_site_mf.png'))

def methylation_fraction_comparison(all_mut_df, illumina_cpg_locs_df, all_methyl_df_t, out_dir, all_meta_df):
    """
    Measure the effect a mutation has on MF at that site
    @ returns: pandas dataframe of all mutations in illumina measured CpG sites, their methylation fraction in mutated sample, and average methylation at that site across other samples (within 5 years of age)
    """
    # get just C>T mutations
    if JUST_CT:
        ct_mutations_df = all_mut_df[all_mut_df['mutation'] == 'C>T']
    else:
        ct_mutations_df = all_mut_df[(all_mut_df['mutation'] == 'C>T') | (all_mut_df['mutation'] == 'G>A') ]
    #for each CpG with methylation data get genomic location of its C
    ct_mutation_in_measured_cpg_df = utils.join_df_with_illum_cpg(ct_mutations_df, illumina_cpg_locs_df, all_methyl_df_t)
    methyl_fractions = utils.get_methyl_fractions(ct_mutation_in_measured_cpg_df, all_methyl_df_t)
    ct_mutation_in_measured_cpg_df['methyl_fraction'] = methyl_fractions
    # get rid of samples that do not have methylayion data
    ct_mutation_in_measured_cpg_df = ct_mutation_in_measured_cpg_df[ct_mutation_in_measured_cpg_df['methyl_fraction'] != -1]
    # get rid of samples that do not have age
    ct_mutation_in_measured_cpg_df = ct_mutation_in_measured_cpg_df.loc[ct_mutation_in_measured_cpg_df['sample'].isin(all_meta_df.index)]
    # get means 
    """ct_mutation_in_measured_cpg_df['avg_methyl_frac'] = utils.get_same_age_means(ct_mutation_in_measured_cpg_df, all_meta_df, all_methyl_df_t)"""
    ct_mutation_in_measured_cpg_df['avg_methyl_frac'] = all_methyl_df_t[ct_mutation_in_measured_cpg_df['#id']].mean().values
    # get difference between mean and mutated sample
    ct_mutation_in_measured_cpg_df['difference'] = ct_mutation_in_measured_cpg_df['methyl_fraction'] - ct_mutation_in_measured_cpg_df['avg_methyl_frac']
    # test for a difference
    compare_mf_mutated_sample_vs_avg(ct_mutation_in_measured_cpg_df, out_dir)
    compare_mf_site_of_mutation_vs_not(ct_mutation_in_measured_cpg_df, all_methyl_df_t, out_dir)
    
    return ct_mutation_in_measured_cpg_df

def main(illumina_cpg_locs_df, out_dir, all_mut_df, all_methyl_df_t, all_meta_df, dataset_names_list):
    # make output directories
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "bootstrap"), exist_ok=True)

    if DATA_SET == "TCGA":

        # do mutation analysis 
        mut_in_measured_cpg_df = plot_mutations(all_mut_df, all_meta_df, dataset_names_list, out_dir, illumina_cpg_locs_df, all_methyl_df_t)
        # subset to only C>T mutations
        # TODO: remove this return if possible
        ct_mut_in_measured_cpg_df = mut_in_measured_cpg_df[mut_in_measured_cpg_df.mutation == "C>T"]
        
        ct_mut_in_measured_cpg_w_methyl_df = methylation_fraction_comparison(all_mut_df, illumina_cpg_locs_df, all_methyl_df_t, out_dir, all_meta_df)

        return mut_in_measured_cpg_df, ct_mut_in_measured_cpg_df, ct_mut_in_measured_cpg_w_methyl_df
    elif DATA_SET == "ICGC":
        sys.exit(1)
    

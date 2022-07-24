import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-deep")
import sys
import os 
import glob

# CONSTANTS
VALID_MUTATIONS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G", "G>C","G>A", "A>T", "A>G" , "A>C", "G>T", "C>-"]
JUST_CT = True
DATA_SET = "TCGA"


def infer_fns_from_data_dirs(data_dirs):
    """
    @ data_dirs: list of data directories
    @ returns: dict of dicts of filenames
    """
    data_files_by_name = {}
    dataset_names_list = []
    for data_dir in data_dirs:
        this_files_dict = {}
        data_set_name = data_dir.split('/')[-1].split('_')[1]
        dataset_names_list.append(data_set_name)
        # if there is a parquet version of methylation use that one
        if len(glob.glob( os.path.join(data_dir, "TCGA.{}.sampleMap2FHumanMethylation450.parquet".format(data_set_name.upper())) , recursive=False)) >= 1:
            methyl_fn = os.path.join(data_dir, "TCGA.{}.sampleMap2FHumanMethylation450.parquet".format(data_set_name.upper()))
        else:
            methyl_fn = os.path.join(data_dir, "TCGA.{}.sampleMap2FHumanMethylation450.gz".format(data_set_name.upper()))
        this_files_dict['methyl_fn'] = methyl_fn
        mut_fn = os.path.join(data_dir, "mc32F{}_mc3.txt.gz".format(data_set_name.upper()))
        this_files_dict['mut_fn'] = mut_fn
        # get ALL clinical files because there may be multiple as for coadread
        clinical_meta_fns = []
        for clinical_meta_fn in glob.glob( os.path.join(data_dir, "clinical*.tsv") , recursive=False):
            clinical_meta_fns.append(clinical_meta_fn)
        this_files_dict['clinical_meta_fns'] = clinical_meta_fns
        # add this_set_files dict to data_files_by_name under the name data_set_name
        data_files_by_name[data_set_name] = this_files_dict
    return data_files_by_name, dataset_names_list

def get_mutations(data_files_by_name):
    """
    @ data_files_by_name: dict of dicts of filenames
    @ returns: pandas dataframe of mutations
    """
    mut_dfs = []
    for dataset_name in data_files_by_name:
        mut_fn = data_files_by_name[dataset_name]['mut_fn']
        mut_df = pd.read_csv(mut_fn, sep='\t')
        # change sample names to not have '-01' at end
        mut_df['sample'] = mut_df['sample'].str[:-3]
        # subset cols
        mut_df = mut_df[['sample', 'chr', 'start', 'end', 'reference', 'alt', 'DNA_VAF']]
        mut_df["mutation"] = mut_df["reference"] + '>' + mut_df["alt"]
        # only keep rows with valid mutations
        mut_df = mut_df[mut_df["mutation"].isin(VALID_MUTATIONS)]
        mut_df['dataset'] = dataset_name
        mut_dfs.append(mut_df)
    # combine all mut_dfs
    all_mut_df = pd.concat(mut_dfs)
    return all_mut_df

def get_methylation(data_files_by_name, illumina_cpg_locs_df, let_na_pass = False):
    """
    @ data_files_by_name: dict of dicts of filenames
    @ returns: pandas dataframe of methylation fractions for all samples, subset to only sites without any NAs and on illumina array
    """
    methyl_dfs = []
    for dataset_name in data_files_by_name:
        print(dataset_name)
        methyl_fn = data_files_by_name[dataset_name]['methyl_fn']
        if methyl_fn.split('.')[-1] == "parquet":
            methyl_df = pd.read_parquet(methyl_fn, engine="fastparquet")
        else: # if not parquet the ending is .gz
            methyl_df = pd.read_csv(methyl_fn, sep='\t')
            # write to parquet so it is fast next time
            new_name = methyl_fn[:-3] + '.parquet'
            methyl_df.to_parquet(new_name, engine="fastparquet")
        # drop any column with a missing methylation value
        if not let_na_pass:
            methyl_df = methyl_df.dropna(how='any')
        # change sample names to not have '-01' at end
        new_column_names = [col[:-3] for col in methyl_df.columns]
        new_column_names[0] = "sample"
        methyl_df.columns = new_column_names
        # drop duplicate columns
        methyl_df = methyl_df.loc[:,~methyl_df.columns.duplicated()]
        # rename sample to cpg and then make it the index so we can join on it
        methyl_df = methyl_df.rename(columns={"sample":"cpg_name"}) # need to fix names after joining 
        methyl_df.set_index(['cpg_name'], inplace=True)
        methyl_dfs.append(methyl_df)
    all_methyl_df = methyl_dfs[0].join(methyl_dfs[1:], how='inner')
    # subset methylation to only be positions on the illumina array (only removes 77 positions, most named rsNNNNNN as if they were snps)
    all_methyl_df = all_methyl_df[all_methyl_df.index.isin(illumina_cpg_locs_df['#id'])]
    return all_methyl_df

def get_metadata(data_files_by_name):
    """
    @ data_files_by_name: dict of dicts of filenames
    @ returns: pandas dataframe of metadata for all samples with duplicates removed and ages as ints
    """
    meta_dfs = []
    for dataset_name in data_files_by_name:
        meta_list = data_files_by_name[dataset_name]['clinical_meta_fns']
        # iterate across list
        for meta_fn in meta_list:
            meta_df = pd.read_csv(meta_fn, sep='\t')
            meta_df = meta_df[['case_submitter_id', 'age_at_index']].drop_duplicates().set_index(['case_submitter_id'])
            meta_df['age_at_index'] = meta_df['age_at_index'].astype(str)
             # drop non-int 
            meta_df['age_at_index'] = meta_df.loc[meta_df['age_at_index'].str.contains(r'\d+')]
            meta_df['dataset'] = dataset_name
            meta_dfs.append(meta_df)
    all_meta_df = pd.concat(meta_dfs)
    # remove any possible duplicate entries
    all_meta_df = all_meta_df.loc[all_meta_df.index.drop_duplicates()]
    all_meta_df.dropna(inplace=True)
    # convert back to int
    all_meta_df['age_at_index'] = all_meta_df['age_at_index'].astype(int)
    return all_meta_df

def transpose_methylation(all_methyl_df):
    """
    @ all_methyl_df: pandas dataframe of methylation fractions for all samples
    @ returns: pandas dataframe of methylation fractions for all samples, transposed
    """
    # turn methylation to numpy for fast transpose
    # save row and col names
    cpg_names = all_methyl_df.index
    sample_names = all_methyl_df.columns
    all_methyl_arr = all_methyl_df.to_numpy()
    all_methyl_arr_t = np.transpose(all_methyl_arr)
    # convert back
    all_methyl_df_t = pd.DataFrame(all_methyl_arr_t, index = sample_names, columns=cpg_names)
    return all_methyl_df_t


def main(illum_cpg_locs_fn, out_dir, data_dirs):
    # make output directories
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "bootstrap"), exist_ok=True)

    if DATA_SET == "TCGA":
        # infer files from data_dirs
        data_files_by_name, dataset_names_list = infer_fns_from_data_dirs(data_dirs)
        run_name = '_'.join(dataset_names_list)

        # read in illumina cpg locations
        illumina_cpg_locs_df = pd.read_csv(illum_cpg_locs_fn, sep=',', dtype={'CHR': str}, low_memory=False)
        illumina_cpg_locs_df = illumina_cpg_locs_df.rename({"CHR": "chr", "MAPINFO":"start", "IlmnID": "#id"}, axis=1)
        illumina_cpg_locs_df = illumina_cpg_locs_df[['#id','chr', 'start', 'Strand']]

        # read in and combine mutation files
        all_mut_df = get_mutations(data_files_by_name)

        # read in and combine methylation files
        all_methyl_df = get_methylation(data_files_by_name, illumina_cpg_locs_df)
        # transpose methylation
        all_methyl_df_t = transpose_methylation(all_methyl_df)

        # read in and combine metadata files (ages)
        all_meta_df = get_metadata(data_files_by_name)

        return illumina_cpg_locs_df, all_mut_df, all_methyl_df, all_methyl_df_t, all_meta_df, run_name, dataset_names_list
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-deep")
import os 
import glob
import dask.dataframe as dd

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

def get_mutations(mut_fn):
    """
    @ data_files_by_name: dict of dicts of filenames
    @ returns: pandas dataframe of mutations
    """
    mut_df = pd.read_csv(mut_fn, sep='\t')
    # change sample names to not have '-01' at end
    mut_df['sample'] = mut_df['sample'].str[:-3]
    # subset cols
    mut_df = mut_df[['sample', 'chr', 'start', 'end', 'reference', 'alt', 'DNA_VAF']]
    mut_df["mutation"] = mut_df["reference"] + '>' + mut_df["alt"]
    # only keep rows with valid mutations
    mut_df = mut_df[mut_df["mutation"].isin(VALID_MUTATIONS)]
    
    return mut_df

def get_methylation(methylation_dir):
    """
    Read in the already preprocessed methylation data
    @ methylation_dir: directory of methylation data
    @ returns: pandas dataframe of methylation data
    """
    methyl_dd = dd.read_parquet(methylation_dir)
    print("Converting Dask df to pandas df, takes ~10min", flush=True)
    methyl_df = methyl_dd.compute()
    return methyl_df

def get_metadata(meta_fn):
    """
    @ metadata_fn: filename of metadata
    @ returns: 
        @ meta_df: pandas dataframe of metadata for all samples with duplicates removed and ages as ints
        @ dataset_names_list: list of dataset names
    """
    # get metadata
    meta_df = pd.read_csv(meta_fn, sep='\t')
    meta_df = meta_df[['sample', 'age_at_initial_pathologic_diagnosis', 'cancer type abbreviation', 'gender']].drop_duplicates()
    meta_df['sample'] = meta_df['sample'].str[:-3]
    meta_df.set_index('sample', inplace=True)
    # drop nans
    meta_df.dropna(inplace=True)
    # rename to TCGA names
    meta_df = meta_df.rename(columns={"age_at_initial_pathologic_diagnosis":"age_at_index", "cancer type abbreviation":"dataset"})
    # drop ages that can't be formated as ints
    meta_df['age_at_index'] = meta_df['age_at_index'].astype(str)
    meta_df['age_at_index'] = meta_df[meta_df['age_at_index'].str.contains(r'\d+')]['age_at_index']
    dataset_names_list = list(meta_df['dataset'].unique())
    # make sure to duplicates still
    meta_df = meta_df.loc[meta_df.index.drop_duplicates()]
    # convert back to int, through float so e.g. '58.0' -> 58.0 -> 58
    meta_df['age_at_index'] = meta_df['age_at_index'].astype(float).astype(int)
    # drop rows with duplicate index
    meta_df = meta_df[~meta_df.index.duplicated(keep='first')]
    return meta_df, dataset_names_list

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

def get_illum_locs(illum_cpg_locs_fn):
    illumina_cpg_locs_df = pd.read_csv(illum_cpg_locs_fn, sep=',', dtype={'CHR': str}, low_memory=False)
    illumina_cpg_locs_df = illumina_cpg_locs_df.rename({"CHR": "chr", "MAPINFO":"start", "IlmnID": "#id"}, axis=1)
    illumina_cpg_locs_df = illumina_cpg_locs_df[['#id','chr', 'start', 'Strand']]
    return illumina_cpg_locs_df

def main(illum_cpg_locs_fn, out_dir, methyl_dir, mut_fn, meta_fn):
    # make output directories
    os.makedirs(out_dir, exist_ok=True)
    # read in illumina cpg locations
    illumina_cpg_locs_df = get_illum_locs(illum_cpg_locs_fn)
    # read in mutations, methylation, and metadata
    all_mut_df = get_mutations(mut_fn)
    all_meta_df, dataset_names_list = get_metadata(meta_fn)
    # add dataset column to all_mut_df
    all_mut_df = all_mut_df.join(all_meta_df, on='sample', how='inner')
    all_mut_df = all_mut_df.drop(columns = ['age_at_index'])
    print("Got mutations and metadata, reading methylation", flush=True)
    all_methyl_df = get_methylation(methyl_dir)
    print("Got methylation, transposing", flush=True)
    # also create transposed methylation
    all_methyl_df_t = transpose_methylation(all_methyl_df)
    print("Done", flush=True)
    return illumina_cpg_locs_df, all_mut_df, all_methyl_df, all_methyl_df_t, all_meta_df, dataset_names_list

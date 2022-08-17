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

def preprocess_methylation(methyl_fn, all_meta_df, illumina_cpg_locs_df, out_dir):
    """
    Takes in a .parquet methylation file to pre-process and outputs a directory of .parquet processed methylation files with only samples with ages in all_meta_df and CpG sites in illumina_cpg_locs_df
    @ methyl_fn: filename of methylation file
    @ all_meta_df: pandas dataframe of metadata for all samples 
    @ illumina_cpg_locs_df: pandas dataframe of CpG sites in illumina
    @ out_dir: directory to output processed methylation files to
    """
    # read in with dask
    methyl_dd = dd.read_parquet(methyl_fn)
    # change sample names to not have '-01' at end
    new_column_names = [col[:-3] for col in methyl_dd.columns]
    new_column_names[0] = "sample"
    methyl_dd.columns = new_column_names
    # drop duplicate columns
    methyl_dd = methyl_dd.loc[:,~methyl_dd.columns.duplicated()]
    # convert to pandas
    methyl_df = methyl_dd.compute()
    # rename sample to cpg and then make it the index
    methyl_df = methyl_df.rename(columns={"sample":"cpg_name"})
    methyl_df = methyl_df.set_index(['cpg_name'])
    # dropna
    methyl_df = methyl_df.dropna(how='any')
    # subset to only samples with ages in all_meta_df
    methyl_df = methyl_df[methyl_df.columns[methyl_df.columns.isin(all_meta_df.index)]]
    # subset to only CpG sites in illumina_cpg_locs_df
    methyl_df = methyl_df[methyl_df.index.isin(illumina_cpg_locs_df['#id'])]
    # convert back to dask to output as 200 parquets
    proc_methyl_dd = dd.from_pandas(methyl_df, npartitions=200)
    # output as parquet
    proc_methyl_dd.to_parquet(out_dir)
    return

def get_methylation(methylation_dir):
    """
    Read in the already preprocessed methylation data
    @ methylation_dir: directory of methylation data
    @ returns: pandas dataframe of methylation data
    """
    methyl_dd = dd.read_parquet(methylation_dir)
    print("Converting Dask df to pandas df", flush=True)
    methyl_df = methyl_dd.compute()
    return methyl_df

"""def get_methylation(data_files_by_name, illumina_cpg_locs_df, let_na_pass = False):

    methyl_dfs = []
    for dataset_name in data_files_by_name:
        print("Getting methylation for {}".format(dataset_name), flush=True)
        methyl_fn = data_files_by_name[dataset_name]['methyl_fn']
        
        # changed from parquet because breaks with some datasets on vscode
        #methyl_df = pd.read_csv(methyl_fn, sep='\t')

        if methyl_fn.split('.')[-1] == "parquet":
            methyl_df = pd.read_parquet(methyl_fn, engine="pyarrow")
        else: # if not parquet the ending is .gz
            methyl_df = pd.read_csv(methyl_fn, sep='\t')
            # write to parquet so it is fast next time
            new_name = methyl_fn[:-3] + '.parquet'
            methyl_df.to_parquet(new_name, engine="pyarrow")
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
    return all_methyl_df"""

def get_metadata(meta_fn):
    """
    @ metadata_fn: filename of metadata
    @ returns: 
        @ meta_df: pandas dataframe of metadata for all samples with duplicates removed and ages as ints
        @ dataset_names_list: list of dataset names
    """
    # get metadata
    meta_df = pd.read_csv(meta_fn, sep='\t')
    meta_df = meta_df[['sample', 'age_at_initial_pathologic_diagnosis', 'cancer type abbreviation']].drop_duplicates()
    meta_df['sample'] = meta_df['sample'].str[:-3]
    meta_df.set_index('sample', inplace=True)
    # drop nans
    meta_df.dropna(inplace=True)
    # rename to TCGA names
    meta_df = meta_df.rename(columns={"age_at_initial_pathologic_diagnosis":"age_at_index"})
    meta_df = meta_df.rename(columns={"cancer type abbreviation":"dataset"})
    # drop ages that can't be formated as ints
    meta_df['age_at_index'] = meta_df['age_at_index'].astype(str)
    meta_df['age_at_index'] = meta_df[meta_df['age_at_index'].str.contains(r'\d+')]['age_at_index']
    dataset_names_list = list(meta_df['dataset'].unique())
    # make sure to duplicates still
    meta_df = meta_df.loc[meta_df.index.drop_duplicates()]
    # convert back to int, through float so e.g. '58.0' -> 58.0 -> 58
    meta_df['age_at_index'] = meta_df['age_at_index'].astype(float).astype(int)

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

def main(illum_cpg_locs_fn, out_dir, methyl_dir, mut_fn, meta_fn):
    # make output directories
    os.makedirs(out_dir, exist_ok=True)
    # read in illumina cpg locations
    illumina_cpg_locs_df = pd.read_csv(illum_cpg_locs_fn, sep=',', dtype={'CHR': str}, low_memory=False)
    illumina_cpg_locs_df = illumina_cpg_locs_df.rename({"CHR": "chr", "MAPINFO":"start", "IlmnID": "#id"}, axis=1)
    illumina_cpg_locs_df = illumina_cpg_locs_df[['#id','chr', 'start', 'Strand']]
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


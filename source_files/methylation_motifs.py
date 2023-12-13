
from pymemesuite.common import MotifFile
import Bio.SeqIO
from pymemesuite.common import Sequence
from pymemesuite.fimo import FIMO
from pyfaidx import Fasta
import pandas as pd
import os
import argparse
import glob
import dask.dataframe as dd
import get_data


def extract_motifs(meme_file: str, out_dir: str):
    """
    Given the wei wang, or other, motif meme file 
    extract each motif into a separate meme file
    """
    with open(meme_file, 'r') as file:
        content = file.read()

    header_end = content.find('MOTIF')
    header = content[:header_end].strip()
    motifs = content[header_end:].split('MOTIF ')[1:]
    for motif in motifs:
        motif_lines = motif.strip().split('\n')
        motif_name = motif_lines[0].split()[0]
        motif_info = '\n'.join(motif_lines[1:])

        motif_file = f'{motif_name}.meme'
        with open(os.path.join(out_dir, motif_file), 'w') as file:
            file.write(f'{header}\n\nMOTIF {motif_name} {motif_name}\n\n{motif_info}\n')
        
        print(f'Saved motif {motif_name} to {motif_file}')
        
        
def extract_surrounding_seq_each_cpg(
    bp_num: int = 15000,
    illumina_cpg_locs_df: pd.DataFrame = None,
    out_fname: str = None,
    ):
    """
    Get the +- bp_num bp surrounding each cpg in illumina_cpg_locs_df
    Write to a fasta file
    """
    reference_genome = Fasta(
    '/cellar/users/zkoch/methylation_and_mutation/data/genome_annotations/hg19.fa'
    )
    # get bp upstream and downstream of each cpg in illumina_cpg_locs_df
    illumina_cpg_locs_df['surrounding_sequences'] = illumina_cpg_locs_df.apply(
        lambda x: reference_genome['chr'+x['chr']][x['start']-bp_num:x['start']+bp_num].seq.upper(), axis = 1
    )
    # write sequences to fasta file with cpg names as sequence names
    # concatenate all sequences into one string
    sequences = illumina_cpg_locs_df.apply(
        lambda row: f'>{row["#id"]}\n{row["surrounding_sequences"]}\n', axis=1
        ).str.cat()
    # write to fasta file
    with open(out_fname, 'w') as file:
        file.write(sequences)
        
def search_motifs_in_surrounding_seqs(
    surrounding_seq_fasta_fn: str = "",
    motif_dir: str = "",
    out_dir: str = "",
    partition_num: int = 0,
    num_partitions: int = 20
    ):
    """
    Search for motifs in the surrounding sequences of each cpg
    Return motif locations in a dataframe with chr, start, end, motif name, cpg name, distance to cpg
    """
    print("reading sequences", flush=True)
    sequences = [
        Sequence(str(record.seq), name=record.id.encode())
        for record in Bio.SeqIO.parse(surrounding_seq_fasta_fn, "fasta")
    ]
    
    print("read sequences", flush=True)
    # for each motif file in motif_dir
    motif_fns = [fn for fn in os.listdir(motif_dir) if fn.endswith('.meme')]
    print(f"read {len(motif_fns)} motif files", flush=True)
    
    number_motifs = len(motif_fns)
    motifs_per_partition = (number_motifs // num_partitions) + 1
    start_index = partition_num * motifs_per_partition
    print(f"starting motif search at motif number {start_index}", flush=True)
    for motif_fn in motif_fns[start_index:start_index+motifs_per_partition]:
        # read in 
        with MotifFile(os.path.join(motif_dir, motif_fn)) as motif_file:
            motif = motif_file.read()
        # search for motif in surrounding sequences
        fimo = FIMO(both_strands=True, max_stored_scores = 100000000)
        # search for the motif in the sequences
        pattern = fimo.score_motif(motif, sequences, motif_file.background)
        motif_occurence_dict = {}
        i = 0
        for m in pattern.matched_elements:
            motif_occurence_dict[i] = {
                'cpg_name': m.source.accession.decode(), 'motif_name': motif.name.decode(),
                'motif_start': m.start, 'motif_stop': m.stop, 'strand': m.strand,
                'score': m.score, 'pvalue': m.pvalue, 'qvalue': m.qvalue
                }
            i += 1
        motif_occurence_df = pd.DataFrame.from_dict(motif_occurence_dict, orient = 'index')        
        motif_occurence_df.to_parquet(os.path.join(out_dir, f"{motif.name.decode()}_motif_occurence_df.parquet"))
        print(f"done with motif {motif.name.decode()}", flush=True)
    
    return 

def combine_and_proc_motif_occurence_files(
    glob_path: str,
    out_dir: str,
    illumina_cpg_locs_df: pd.DataFrame,
    bp_num: int = 15000
    ):
    """
    Read in all motif occurence files matching glob_path, combine into one df, join with illumina_cpg_locs_df
    """
    dfs = []
    for fn in glob.glob(glob_path):
        df = pd.read_parquet(fn, engine="pyarrow", use_threads=True)
        dfs.append(df)
        print(f"read in {fn}", flush=True)
    # read in all motif occurence files
    all_motif_occurence_df = pd.concat(dfs)
    
    print(f"read in all motif occurence files for a total of {all_motif_occurence_df.shape[0]} motifs", flush=True)
    # filter out motifs with large p values
    all_motif_occurence_df = all_motif_occurence_df.loc[
        all_motif_occurence_df['qvalue'] < 0.05
        ]
    print(
        f"filtered out motifs with large p values, leaving {all_motif_occurence_df.shape[0]} motifs",
        flush=True
        )
    all_motif_occurence_df.rename(columns = {'cpg_name': '#id'}, inplace = True)
    # join with illumina_cpg_locs_df
    all_motif_occurence_w_illum = all_motif_occurence_df.merge(
        illumina_cpg_locs_df, on = '#id', how = 'left'
        )
    print("joined with illumina_cpg_locs_df", flush=True)
    # get distance of motif start and stop to cpg
    all_motif_occurence_w_illum['start_distance_to_cpg'] = all_motif_occurence_w_illum['motif_start'] - bp_num
    all_motif_occurence_w_illum['end_distance_to_cpg'] = all_motif_occurence_w_illum['motif_stop'] - bp_num
    # get genomic start and end of motif
    all_motif_occurence_w_illum['motif_genomic_start'] = all_motif_occurence_w_illum['start'] \
        + all_motif_occurence_w_illum['start_distance_to_cpg']
    all_motif_occurence_w_illum['motif_genomic_end'] = all_motif_occurence_w_illum['start'] \
        + all_motif_occurence_w_illum['end_distance_to_cpg']
    print("got distance of motif start and stop to cpg", flush=True)
    # create a column of if MM or UM
    all_motif_occurence_w_illum['MM_or_UM'] = all_motif_occurence_w_illum.apply(
        lambda x: 'MM' if x['motif_name'].startswith('M') else 'UM', axis = 1
        )
    print("created MM_or_UM column", flush=True)
    # convert to dask df and write to parquet
    if out_dir != "":
        all_motif_occurence_w_illum_dd = dd.from_pandas(all_motif_occurence_w_illum, npartitions = 100)
        all_motif_occurence_w_illum_dd.to_parquet(out_dir)
        print("wrote to parquet directory", flush=True)
    return all_motif_occurence_w_illum
    
    
def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob_path", type=str, default="",
    )
    parser.add_argument(
        "--out_fn", type=str, default="",
    )
    parser.add_argument(
        "--out_dir", type=str, default="",
    )
    # prase
    args = parser.parse_args()
    glob_path = args.glob_path
    out_fn = args.out_fn
    out_dir = args.out_dir
    # get illum
    # qnorm 
    _, illumina_cpg_locs_df, _, _, _ = get_data.read_icgc_data()
    all_motif_occurence_w_illum = combine_and_proc_motif_occurence_files(
        glob_path=glob_path,
        out_dir=out_dir,
        illumina_cpg_locs_df=illumina_cpg_locs_df
        )
    all_motif_occurence_w_illum.to_parquet(out_fn)
    print(f"wrote to {out_fn}", flush=True)
    
    """parser.add_argument(
        "--meme_dir", type=str, default="",
        help="path to wei wang motif dir"
    )
    parser.add_argument(
        "--surrounding_seq_fasta_fn", type=str, default="",
        help="path to surrounding_seq_fasta_fn"
    )
    parser.add_argument(
        "--out_dir", type=str, default="",
        help="path to output directory"
    )
    parser.add_argument(
        "--partition_num", type=int,
        help="partition number"
    )
    parser.add_argument(
        "--num_partitions", type=int,
        help=" number"
    )
    args = parser.parse_args()
    meme_dir = args.meme_dir
    surrounding_seq_fasta_fn = args.surrounding_seq_fasta_fn
    out_dir = args.out_dir
    partition_num = args.partition_num
    num_partitions = args.num_partitions
    
    search_motifs_in_surrounding_seqs(
        surrounding_seq_fasta_fn=surrounding_seq_fasta_fn,
        motif_dir=meme_dir,
        out_dir=out_dir,
        partition_num=partition_num,
        num_partitions=num_partitions
        )"""
    
if __name__ == "__main__":
    main()
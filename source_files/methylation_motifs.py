
from pymemesuite.common import MotifFile
import Bio.SeqIO
from pymemesuite.common import Sequence
from pymemesuite.fimo import FIMO
from pyfaidx import Fasta
import pandas as pd
import os
import argparse

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
    out_fn: str = "",
    partition_num: int = 0,
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
    num_paritions = 20
    num_seq = len(sequences)
    
    print("read sequences", flush=True)
    # for each motif file in motif_dir
    motif_fns = [fn for fn in os.listdir(motif_dir) if fn.endswith('.meme')]
    print(f"read {len(motif_fns)} motif files", flush=True)
    
    all_motif_occurence_dfs = []
    for motif_fn in motif_fns:
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
        all_motif_occurence_dfs.append(motif_occurence_df)
        print(f"done with motif {motif.name.decode()}", flush=True)
    all_motif_occurence_df = pd.concat(all_motif_occurence_dfs)
    all_motif_occurence_df['start_distance_to_cpg'] = motif_occurence_df.apply(
            lambda x: x['motif_start'] - 15000, axis = 1
            )
    all_motif_occurence_df.to_parquet(out_fn)
    print(f"wrote to {out_fn}", flush=True)
    return all_motif_occurence_df

    
    
def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meme_dir", type=str, default="",
        help="path to wei wang motif dir"
    )
    parser.add_argument(
        "--surrounding_seq_fasta_fn", type=str, default="",
        help="path to surrounding_seq_fasta_fn"
    )
    parser.add_argument(
        "--out_fn", type=str, default="",
        help="path to output directory"
    )
    args = parser.parse_args()
    meme_dir = args.meme_dir
    surrounding_seq_fasta_fn = args.surrounding_seq_fasta_fn
    out_fn = args.out_fn
    
    search_motifs_in_surrounding_seqs(
        surrounding_seq_fasta_fn=surrounding_seq_fasta_fn,
        motif_dir=meme_dir,
        out_fn=out_fn,
        )
    
if __name__ == "__main__":
    main()
from glob import glob
import pandas as pd

##################
# specify the input files and directories
##################
# MUST HAVE SAME NUMBER OF SAMPLES IN EACH FILE AND IN SAME ORDER
# the file that contains the methylation data
## samples X CpGs matrix of methylation values [0, 1]
methyl_fn = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/methyl.csv.gz" #"/cellar/users/zkoch/methylation_and_mutation/data/icgc/for_matrixQTL/icgc_methyl_df_cpgXsamples.csv.gz"
# the file that contains the mutation data
## must have, at least, columns 'mut_loc' (chr:pos), 'sample', and 'MAF'
mut_fn = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/tcga_muts.parquet" #"/cellar/users/zkoch/methylation_and_mutation/data/icgc/for_matrixQTL/icgc_mut_df.parquet"
# output directory where partitioned mutation files, matrixQTL outputs, and predictors will be stored
out_dir = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/clumped_muts" #"/cellar/users/zkoch/methylation_and_mutation/output_dirs/icgc_muts_011423"

##################
# define constants and constantishs
##################
CHROMS = [str(i) for i in range(1,23)]
MUT_PER_FILE = 10000
CLUMP_WINDOW_SIZE = 1000
mut_df = pd.read_parquet(mut_fn)
# set NUM_UNIQUE_MUTS to the number of unique mutations in the mutation file or number of clumps
if CLUMP_WINDOW_SIZE > 1:
  def round_down(num, divisor):
    return num - (num%divisor)
  mut_df['binary_MAF'] = 1
  # clump start is 'start' rounded down to nearest 1000
  mut_df['clump_start'] = mut_df['start'].apply(lambda x: round_down(x, CLUMP_WINDOW_SIZE))
  mut_df['clump_loc'] = mut_df['chr'].astype(str) + ':' + mut_df['clump_start'].astype(str)
  # combine rows with the same sample, chr, and clump_start values by adding the MAF values
  grouped_mut_df = mut_df.groupby(
      ['sample', 'clump_loc']
      ).agg({'MAF': 'sum', 'binary_MAF': 'sum'}).reset_index()
  NUM_UNIQUE_MUTS = len(grouped_mut_df['clump_loc'].unique()) # constantish
else:
  NUM_UNIQUE_MUTS = len(mut_df['mut_loc'].unique()) # constantish

rule all:
  input:
    # testing 
    expand("{mut_fn}", mut_fn = [os.path.join(out_dir, f"muts_{i}.csv.gz") for i in range(0, NUM_UNIQUE_MUTS, MUT_PER_FILE)])
    # when want to run matrixQTL
    # expand(os.path.join(out_dir, "chr{chrom}_meqtl.parquet"), chrom=CHROMS)

rule clump_and_partition_mutations:
  """
  Partition the mutation file into smaller files
  """
  input:
    mut_fn = mut_fn
  conda:
    "big_data"
  output:
    expand("{mut_fn}", mut_fn = [os.path.join(out_dir, f"muts_{i}.csv.gz") for i in range(0, NUM_UNIQUE_MUTS, MUT_PER_FILE)])
  shell:
    "python /cellar/users/zkoch/methylation_and_mutation/snake_source_files/clump_and_partition_mutations.py --mut_fn {input.mut_fn} --out_dir {out_dir} --mut_per_file {MUT_PER_FILE} --clump_window_size {CLUMP_WINDOW_SIZE}"

rule matrixQTL:
  """
  Given a partitioned mutation file and the methylation file, run matrixQTL
  """
  input:
    expand("{mut_fn}", mut_fn = [os.path.join(out_dir, f"muts_{i}.csv.gz") for i in range(0, NUM_UNIQUE_MUTS, MUT_PER_FILE)]),
    mut_fn = "{muts_fn}",
    methyl_fn = methyl_fn
  conda: 
    "renv"
  output:
    "{muts_fn}.meqtl"
  shell:
    "Rscript /cellar/users/zkoch/methylation_and_mutation/snake_source_files/run_matrixQTL.R {input.mut_fn} {input.methyl_fn}"

rule group_meqtls:
  """
  Take the output of matrixQTL and group the results by CpG chromosome
  """
  input:
    expand("{mut_fn}.meqtl", mut_fn = [os.path.join(out_dir, f"muts_{i}.csv.gz") for i in range(0, NUM_UNIQUE_MUTS, MUT_PER_FILE)])
  conda:
    "big_data"
  output:
    os.path.join(out_dir, "chr{chrom}_meqtl.parquet")
  shell:
    "python /cellar/users/zkoch/methylation_and_mutation/snake_source_files/group_meqtls_by_cpg.py --chrom {wildcards.chrom} --out_fn {output}"

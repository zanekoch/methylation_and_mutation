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
mut_fn = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/muts.parquet" #"/cellar/users/zkoch/methylation_and_mutation/data/icgc/for_matrixQTL/icgc_mut_df.parquet"
# output directory where partitioned mutation files, matrixQTL outputs, and predictors will be stored
out_dir = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/muts" #"/cellar/users/zkoch/methylation_and_mutation/output_dirs/icgc_muts_011423"
##################
# define constants and constantishs
##################
CHROMS = [str(i) for i in range(1,23)]
MUT_PER_FILE = 10000
mut_df = pd.read_parquet(mut_fn)
NUM_UNIQUE_MUTS = 2420000 # len(mut_df['mut_loc'].unique()) # constantish
##################
# the number of cpgs to partition the methylation data based on
# TODO: make this dynamically set based on the number of cpgs in methyl_fn
total_cpgs = 267152
cpg_starts = [i for i in range(0, total_cpgs, 1000)]
cpg_ends = [i for i in range(999, total_cpgs, 1000)]
cpg_ends.append(total_cpgs)
cpgs = list(zip(cpg_starts, cpg_ends))
##################

rule all:
  input:
    # when want to train predictors
    # expand(os.path.join(predictors_dir, "{cpg[0]}_{cpg[1]}.txt"), cpg = cpgs)
    # when want to run matrixQTL
    expand(os.path.join(out_dir, "chr{chrom}_meqtl.parquet"), chrom=CHROMS)

rule partition_mutations:
  input:
    mut_fn = mut_fn
  conda:
    "big_data"
  output:
    expand("{mut_fn}", mut_fn = [os.path.join(out_dir, f"muts_{i}.csv.gz") for i in range(0, NUM_UNIQUE_MUTS, MUT_PER_FILE)])
  shell:
    "python /cellar/users/zkoch/methylation_and_mutation/snake_source_files/partition_mutations.py --mut_fn {input.mut_fn} --out_dir {out_dir} --mut_per_file {MUT_PER_FILE}"

rule matrixQTL:
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
  input:
    expand("{mut_fn}.meqtl", mut_fn = [os.path.join(out_dir, f"muts_{i}.csv.gz") for i in range(0, NUM_UNIQUE_MUTS, MUT_PER_FILE)])
  conda:
    "big_data"
  output:
    os.path.join(out_dir, "chr{chrom}_meqtl.parquet")
  shell:
    "python /cellar/users/zkoch/methylation_and_mutation/snake_source_files/group_meqtls_by_cpg.py --chrom {wildcards.chrom} --out_fn {output}"

rule balance_train_test_split:
  output:
    os.path.join(out_dir, "train_samples.txt"),
    os.path.join(out_dir, "test_samples.txt")
  conda:
    "big_data"
  shell:
    "python /cellar/users/zkoch/methylation_and_mutation/snake_source_files/balance_train_test_split.py --out_dir {out_dir}"

rule train_methyl_predictors:
  input:
    expand(os.path.join(out_dir, "chr{chrom}_meqtl.parquet"), chrom=CHROMS),
    train_samples_fn = os.path.join(out_dir, "train_samples.txt"),
    test_samples_fn = os.path.join(out_dir, "test_samples.txt")
  output:
    os.path.join(out_dir, "{cpg_start}_{cpg_end}.txt")
  conda:
    "big_data"
  shell:
    "python /cellar/users/zkoch/methylation_and_mutation/snake_source_files/train_methyl_predictors.py --cpg_start {wildcards.cpg_start} --cpg_end {wildcards.cpg_end} --out_dir {out_dir} --train_samples_fn {input.train_samples_fn}"
    

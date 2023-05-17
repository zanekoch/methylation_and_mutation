from glob import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold


##################
# specify the input files and directories
# MUST HAVE SAME NUMBER OF SAMPLES IN EACH FILE AND IN SAME ORDER
##################
snake_source_dir="/cellar/users/zkoch/methylation_and_mutation/snake_source_files"
# the file that contains the methylation data: samples X CpGs matrix of methylation values [0, 1]
# methyl_fn = "/cellar/users/zkoch/methylation_and_mutation/data/icgc_matrixQTL_data/icgc_methyl_df_cpgXsamples.csv.gz"
methyl_fn = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/tcga_methyl.csv.gz" 

# the file that contains the mutation data
## must have, at least, columns 'mut_loc' (chr:pos), 'sample', and 'MAF'
#mut_fn = "/cellar/users/zkoch/methylation_and_mutation/data/icgc_matrixQTL_data/icgc_mut_df.parquet"
mut_fn = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/tcga_muts.parquet" 

# file containing gender, age, and dataset covariates
cov_fn = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/tcga_covariates.csv.gz"
# cov_fn = "/cellar/users/zkoch/methylation_and_mutation/data/icgc_matrixQTL_data/icgc_cov_for_matrixQTL.csv.gz"

# output directory where partitioned mutation files, matrixQTL outputs, and predictors will be stored
#"/cellar/users/zkoch/methylation_and_mutation/output_dirs/icgc_muts_011423"
out_dir = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/tcga_clumped_muts_CV"
# ut_dir = "/cellar/users/zkoch/methylation_and_mutation/data/icgc_matrixQTL_data/icgc_clumped_muts_CV"



##################
# define constants and constantishs
##################
NUM_FOLDS = 5 # set number of cross validations
NUM_MATRIXQTL_PARTITIONS = 10
CHROMS = [str(i) for i in range(1,23)]
MUT_PER_FILE = 10000
CLUMP_WINDOW_SIZE = 1000

# TODO: because CV changes the number of samples the samples in mutation CV files do not match the overall methylation file
# - solution 1: make create folds split the methylation file.
  # This would take forever to re-write the methylation file tho

# - solution 2: make R script subset/only read in methylation for samples in mutation file


rule all:
  input:
    # when want to run matrixQTL
    expand(
      os.path.join(out_dir, "chr{chrom}_meqtl_fold_{fold}.parquet"),
      chrom=CHROMS, 
      fold = [y for y in range(NUM_FOLDS)]
      )
  resources:
    mem_mb = 5*1024,
    runtime = 5

rule create_folds:
  """
  Split samples into train and test sets
  """
  input:
    cov_fn = cov_fn
  conda:
    "big_data"
  resources:
    mem_mb = 36*1024,
    runtime = 5
  output:
    expand(os.path.join(out_dir, "train_samples_fold_{i}.pkl"), i = range(NUM_FOLDS)),
    expand(os.path.join(out_dir, "test_samples_fold_{i}.pkl"), i = range(NUM_FOLDS))
  shell:
    "python {snake_source_dir}/train_test_split.py --num_folds {NUM_FOLDS} --covariate_fn {input.cov_fn} --out_dir {out_dir}"
  
rule piv_and_clump_mutations:
  """
  Pivot and optionally clump mutations and write to files
  """
  input:
    mut_fn = mut_fn,
    train_samples = os.path.join(out_dir, "train_samples_fold_{fold}.pkl"),
    test_samples = os.path.join(out_dir, "test_samples_fold_{fold}.pkl")
  conda:
    "big_data"
  resources:
    mem_mb = 96*1024,
    runtime = 80
  output:
    os.path.join(out_dir, "muts_fold_{fold}.csv.gz")
  shell:
    "python {snake_source_dir}/clump_and_partition_mutations.py --mut_fn {input.mut_fn} --out_dir {out_dir}  --clump_window_size {CLUMP_WINDOW_SIZE} --training_samples_fn {input.train_samples}"

rule matrixQTL:
  """
  Given a a mutation file and the methylation file, run matrixQTL
  """
  input:
    muts_fn = os.path.join(out_dir, "muts_fold_{fold}.csv.gz"),
    methyl_fn = methyl_fn,
    cov_fn = cov_fn,
  conda: 
    "renv"
  resources:
    mem_mb = 60*1024,
    runtime = 45
  output:
    os.path.join(out_dir, "muts_fold_{fold}_partition_{partition_num}.meqtl")
  shell:
    "Rscript {snake_source_dir}/run_matrixQTL.R {input.muts_fn} {input.methyl_fn} {input.cov_fn} {wildcards.partition_num} {NUM_MATRIXQTL_PARTITIONS}"

rule group_meqtls:
  """
  Take the output of matrixQTL and group the results by CpG chromosome
  """
  input:
    expand(
      os.path.join(out_dir, "muts_fold_{{fold}}_partition_{partition_num}.meqtl"),
      partition_num = [y for y in range(NUM_MATRIXQTL_PARTITIONS)]
      )
  conda:
    "big_data"
  resources:
    mem_mb = 128*1024,
    runtime = 45
  output:
    os.path.join(out_dir, "chr{chrom}_meqtl_fold_{fold}.parquet")
  shell:
    "python {snake_source_dir}/group_meqtls_by_cpg.py --chrom {wildcards.chrom} --out_fn {output} --matrix_qtl_dir {out_dir} --fold {wildcards.fold}"

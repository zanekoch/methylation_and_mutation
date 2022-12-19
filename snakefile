from glob import glob

# specify the input files
mut_fns = glob("/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/muts/*csv.gz")
methyl_fn = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/methyl.csv.gz"

# define the rule to run the R script for each file in the muts directory
rule all:
  input:
    expand("{mut_fn}.meqtl", mut_fn=mut_fns)

rule run_matrixQTL:
  input:
    mut_fn = "{muts_fn}",
    methyl_fn = methyl_fn
  conda: 
    "renv"
  output:
    "{muts_fn}.meqtl"
  shell:
    "Rscript /cellar/users/zkoch/methylation_and_mutation/submission_scripts/run_matrixQTL.R {input.mut_fn} {input.methyl_fn}"

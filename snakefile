from glob import glob

# specify the input files
mut_dir = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/binary_muts"
mut_fns = glob(mut_dir + "/*csv.gz")
methyl_fn = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/methyl.csv.gz"
chroms = [str(i) for i in range(1,23)]

rule all:
  input:
    expand(os.path.join(mut_dir, "chr{chrom}_meqtl.parquet"), chrom=chroms)

rule matrixQTL:
  input:
    mut_fn = "{muts_fn}",
    methyl_fn = methyl_fn
  conda: 
    "renv"
  output:
    "{muts_fn}.meqtl"
  shell:
    "Rscript /cellar/users/zkoch/methylation_and_mutation/snake_source_files/run_matrixQTL.R {input.mut_fn} {input.methyl_fn}"


# define the rule to run the R script for each file in the muts directory
rule group_meqtls:
  input:
    expand("{mut_fn}.meqtl", mut_fn=mut_fns)
  conda:
    "big_data"
  output:
    os.path.join(mut_dir, "chr{chrom}_meqtl.parquet")
  shell:
    "python /cellar/users/zkoch/methylation_and_mutation/snake_source_files/group_meqtls_by_cpg.py --chrom {wildcards.chrom} --out_fn {output}"


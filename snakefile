from glob import glob

# specify the input files
mut_dir = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/muts"
mut_fns = glob(mut_dir + "/*csv.gz")
methyl_fn = "/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/methyl.csv.gz"
chroms = [str(i) for i in range(1,23)]
predictors_dir = "/cellar/users/zkoch/methylation_and_mutation/output_dirs/output_010423"

total_cpgs = 10000
cpg_starts = [i for i in range(0, total_cpgs, 500)]
cpg_ends = [i for i in range(499, total_cpgs, 500)]
cpg_ends.append(total_cpgs)
cpgs = list(zip(cpg_starts, cpg_ends))

rule all:
  input:
    expand(os.path.join(predictors_dir, "{cpg[0]}_{cpg[1]}.txt"), cpg = cpgs)

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

rule group_meqtls:
  input:
    expand("{mut_fn}.meqtl", mut_fn=mut_fns)
  conda:
    "big_data"
  output:
    os.path.join(mut_dir, "chr{chrom}_meqtl.parquet")
  shell:
    "python /cellar/users/zkoch/methylation_and_mutation/snake_source_files/group_meqtls_by_cpg.py --chrom {wildcards.chrom} --out_fn {output}"

rule train_methyl_predictors:
  input:
    expand(os.path.join(mut_dir, "chr{chrom}_meqtl.parquet"), chrom=chroms)
  output:
    os.path.join(predictors_dir, "{cpg_start}_{cpg_end}.txt")
  conda:
    "big_data"
  shell:
    "python /cellar/users/zkoch/methylation_and_mutation/snake_source_files/train_methyl_predictors.py --cpg_start {wildcards.cpg_start} --cpg_end {wildcards.cpg_end} --out_dir {predictors_dir}"
    

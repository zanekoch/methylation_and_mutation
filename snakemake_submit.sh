snakemake --use-conda --cluster "sbatch -A zkoch -p nrnb-compute --mem=96GB -t 1:13:00" --jobs 25

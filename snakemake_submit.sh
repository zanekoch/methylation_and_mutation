snakemake --use-conda --cluster "sbatch -A zkoch -p nrnb-compute --mem=128GB -t 6:00:00" --jobs 5 --retries 3
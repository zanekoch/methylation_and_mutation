snakemake --keep-incomplete --use-conda --cluster "sbatch -A zkoch -p nrnb-compute --mem=300GB -t 6:00:00" --jobs 20 --retries 0

snakemake --keep-incomplete --use-conda --cluster "sbatch -A zkoch -p nrnb-compute --mem=512GB -t 6:00:00" --jobs 2 --retries 0

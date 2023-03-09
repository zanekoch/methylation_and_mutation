snakemake --dry-run --keep-incomplete --use-conda --cluster "sbatch -A zkoch -p nrnb-compute --mem=128GB -t 2:00:00" --jobs 23 --retries 2 --rerun-triggers mtime 
# takes <1 hour and 48GB 
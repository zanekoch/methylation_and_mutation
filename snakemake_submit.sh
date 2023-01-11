snakemake --use-conda --cluster "sbatch -A zkoch -p nrnb-compute --mem=75GB --cpus-per-task=10 -t 6:00:00" --jobs 20 --retries 1

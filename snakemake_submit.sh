#snakemake --keep-incomplete --use-conda --retries 2 --rerun-triggers mtime --cluster-config cluster.json --cluster "sbatch -A zkoch -p nrnb-compute --job-name={cluster.name} --output={cluster.output} --error={cluster.error} --cpus-per-task={cluster.cpus} --mem={cluster.mem} --time={cluster.time}" --jobs 23

snakemake --rerun-incomplete --use-conda --cluster "sbatch -A zkoch -p nrnb-compute --mem=96GB -t 01:20:00" --jobs 23 --retries 2 --rerun-triggers mtime
# takes <1 hour and 48GB 
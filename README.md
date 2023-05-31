# methylation_and_mutation
# Data wrangling (yeehaw!)

## Environments ##
### Conda ###
- `big_data` for snakemake (matrixQTL)
- `tf_env` for everything else

## **Stage 0**: Data formatting
1. ### Convert methylation into a samples x CpGs matrix
    - To go from .csv to parquet `/cellar/users/zkoch/methylation_and_mutation/submission_scripts/convert_pancanMethylcsv_to_methylParquet.py`
    - Drop all CpGs with missing values
2. ### Convert mutations into a tall df where each row is a mutation event
    - Drop all non-SBS mutations
3. ### Create a meta df with dataset, age, and gender info
4. ### Example of doing this: `/methylation_and_mutation/icgc_comethylation_051523.ipynb`
5. ### Outputs:
    - TCGA: `/cellar/users/zkoch/methylation_and_mutation/data/final_tcga_data`
    - ICGC: `/cellar/users/zkoch/methylation_and_mutation/data/final_icgc_data`
## **Stage 1**: Data preprocessing
1. ### Quantile normalize methylation data, writing to a directory of `.parquet` files ###
    - Script: `utils.each_tissue_drop_divergent_and_qnorm()` and/or `utils.preprocess_methylation()`
        - Drop any sample with >3SD mean methylation from other samples of same tissue
        - Qnorm within each tissue (ICGC)
        - Or across all tissue (TCGA)
    - Outputs: 
        - ICGC: `/cellar/users/zkoch/methylation_and_mutation/data/final_icgc_data` 
        - TCGA: `/cellar/users/zkoch/methylation_and_mutation/data/final_tcga_data`

2. ### Run matrixQTL aggregating nearby mutations to find pututative somatic-meQTLs ###
    - Script: `/cellar/users/zkoch/methylation_and_mutation/snakefile`
    - Inputs:
        - `methyl_fn` (.csv): **non-qnormed** methylation in .csv format, CpGs x Samples
        - `mut_fn` (.parquet): mutation file with columns: sample, chr, start, MAF, and mut_loc (chr:start)
        - `cov_fn` (.csv): covariates x samples df with columns: samples, rows: age_at_index, encoded-dataset, one-hot-encoded gender
        - `out_dir`
        - TCGA: `/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/`
        - ICGC: `/cellar/users/zkoch/methylation_and_mutation/data/icgc_matrixQTL_data/`
    - Outputs:
        - TCGA: `/cellar/users/zkoch/methylation_and_mutation/data/matrixQtl_data/tcga_clumped_muts_CV`
            - has both the processed files and the outputs of matrixQTL
        - ICGC: `/cellar/users/zkoch/methylation_and_mutation/data/icgc_matrixQTL_data/icgc_clumped_muts_CV`
3. ### Calculate correlation matrices within each chrom and tissue type ###
    - Script: `/cellar/users/zkoch/methylation_and_mutation/submission_scripts/calc_all_chrDset_corrs.py`
    - Inputs:
        - `consortium`: which consortium to use **qnormed** methylation from
        - `out_dir`
    - Outputs:
        - TCGA: `/cellar/users/zkoch/methylation_and_mutation/data/final_tcga_data/chr_dset_corrs_qnorm`
        - ICGC `/cellar/users/zkoch/methylation_and_mutation/data/final_icgc_data/chr_dset_corrs_qnorm`

# Methylation disturbance analysis
## **Methylation disturbance**: 
1. ### Run the comethylation disturbance analysis ###
    - Script: `/cellar/users/zkoch/methylation_and_mutation/submission_scripts/run_compute_comethylation.py` 
    - Inputs:
        - `consortium`: which consortium to use **qnormed** methylation from
        - `out_dir`
        - preprocessed methylation and mutation data
        - parameters specifying the comethylation analysis
    - Outputs:
        - comparison_sites_df: a dir of parquet files specifying the comparison sites for each mutation event
        - all_metrics_df: a dir of parquet files specifying the resulting methylation disturbance metrics (e.g. delta MF) for each mutation event   
2. ### Get the mean effect across each mutated locus ###
    - Script: `/cellar/users/zkoch/methylation_and_mutation/submission_scripts/run_get_mean_metrics_comethylation.py`
    - Inputs:
        - `all_metrics_glob_path`: glob to find all_metrics_df parquet files
        - `out_dir`
    - Outputs:
        - One mean metrics file in outdir for each `all_metrics_glob_path`
3. ### Annotate mutated loci ###
    

# soMage
## **soMage Step 1**: Feature creating and methylation prediction
1. ### Create features an train methylation predictors ###
    - Script: `/cellar/users/zkoch/methylation_and_mutation/submission_scripts/run_somatic_mut_clock.py`
    - Outputs:
        - trained model files in `out_dir`
        - predicted methylation in `out_dir`
## **soMage Step 2**: Clock training and validation
1. ### Train predicted methylation clocks ###
    - Script: `/cellar/users/zkoch/methylation_and_mutation/submission_scripts/run_combined_predicted_methyl_clock.py`
    - Inputs:
        - `somage_path`: path to directory of somage directories containing feat mats, predicted methyls, trained models
        - `directory_glob`: glob to find specific directories in somage_path (e.g. for one CV)
        - `file_suffix`: suffix to find specific files in somage_path/directory_glob (e.g. for scrambled or non-scrambled)
        - `cross_val`: cross val fold number
        - `do`: train_clocks, output_methyl_and_dset_perf
        - `out_dir`: output directory
    - Outputs:
        - A combined predicted methylation file
        - Trained clocks from a grid-search
2. ### Train actual methylation clocks ###
    - Trains a elasticNet and xgboost clock for each tissue type within a dataset
    - Also trains an elasticNet and xgboost clock for all tissues together within a dataset
    - Script: `/cellar/users/zkoch/methylation_and_mutation/submission_scripts/train_actual_methyl_epi_clocks.py`
    - Inputs:
        - `consortium`, `cv_num`, and `out_dir`
        - Depends on data from Stage 1.1 already being in `final_tcga_data` or `final_icgc_data`
    - Outputs:
        - trained model files in `out_dir`
        - ICGC: `/cellar/users/zkoch/methylation_and_mutation/data/final_icgc_data/actual_methyl_epi_clocks`

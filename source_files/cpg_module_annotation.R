library(arrow)
library(missMethyl)
library(dplyr)
library(tibble)
library(jsonlite)


get_close_cpgs <- function(
    dist_df, 
    cpg_name,
    N
    ) {
    # A function to get the top N CpGs that are closest to the specified CpG site
    # @ param dist_df: a data frame containing the distance values for all CpGs
    # @ param cpg_name: the name of the CpG site to get the top correlated CpGs for
    # @ param N: the number of top correlated CpGs to return
    cpg_dist_list <- dist_df[,cpg_name]
    # choose the closest N CpGs, exluding the CpG site itself
    close_cpg_indices <- order(cpg_dist_list, decreasing = FALSE)[2:N]
    # and return the CpG names at those indices
    return(colnames(dist_df)[close_cpg_indices])
}

get_random_close_cpgs <- function(
    dist_df,
    cpg_name,
    N,
    min_dist) {
    # A function to get N random CpGs that are between min_dist and min_dist*10 away from the specified CpG site
    # @ param dist_df: a data frame containing the distance values for all CpGs
    # @ param cpg_name: the name of the CpG site to get the top correlated CpGs for
    # @ param N: the number of top correlated CpGs to return
    # @ param min_dist: the minimum distance between the CpG site and the top correlated CpGs
    # @ return: a vector of CpG names

    # get the index of rows where cpg_name column is greater than min_dist
    pass_dist_thresh <- dist_df[,cpg_name] > min_dist
    index_pass_dist_thresh <- rownames(dist_df)[pass_dist_thresh]
    
    # choose N random indices from close_cpg_indices, weighting by the inverse of the distance
    pass_treshold_dist_df <- dist_df[index_pass_dist_thresh, cpg_name]
    inverted <- max(pass_treshold_dist_df) - pass_treshold_dist_df
    probs <- inverted / sum(inverted)

    chosen_cpg_names <- sample(index_pass_dist_thresh, N, replace = FALSE, prob = as.list(t(probs)))
    # return the CpG names at those indices
    return(chosen_cpg_names)
}



get_top_correlated_distal_cpgs <- function(
    corr_df, 
    dist_df,
    cpg_name, 
    N,
    min_dist) {
    # A function to get the top N CpGs correlated with the specified CpG site
    # @ param corr_df: a data frame containing the correlation values for all CpGs
    # @ param dist_df: a data frame containing the distance values for all CpGs
    # @ param cpg_name: the name of the CpG site to get the top correlated CpGs for
    # @ param N: the number of top correlated CpGs to return
    # @ param min_dist: the minimum distance between the CpG site and the top correlated CpGs
    # @ return: a vector of CpG names
    
    # Get the distance vector for the specified CpG site
    cpg_dist_list <- dist_df[,cpg_name]
    pass_dist_thresh <- cpg_dist_list > min_dist
    # using dyplr, select the rows of corr_df that pass the distance threshold
    corr_passing_dist_df <- corr_df %>% 
        rownames_to_column("cpg_name") %>% 
        filter(pass_dist_thresh) %>% 
        column_to_rownames("cpg_name")
    # return the row names of corr_passing_dist_df ordered by the correlation value of the specified CpG site
    row_order <- order(corr_passing_dist_df[,cpg_name], decreasing = TRUE)
    sorted_cpgs <- rownames(corr_passing_dist_df)[row_order]
    # return the top N CpGs
    return(sorted_cpgs[1:N])
}

#' A function to run the annotation pipeline
#' 
#' @param corr_df: a data frame containing the correlation values for all CpGs
#' @param dist_df: a data frame containing the distance values for all CpGs
#' @param num_cpgs: the number of CpGs to run the annotation pipeline on
#' @param site_type: the type of CpG sites to use for annotation
#'   - 'corr': the top N CpGs correlated with the specified CpG site
#'   - 'close': the top N CpGs closest to the specified CpG site
#'   - 'random': N random CpGs between min_dist and farther, weighted by the inverse of the distance
#' @param N: the number of top correlated or random CpGs to run on
#' @param min_dist: the minimum distance between the CpG site and the top correlated CpGs
#' @return: 
#'   - a data frame containing the annotation results
#'   - a vector containing the number of significant annotations for each CpG
run_annotation <- function(
    corr_df, 
    dist_df,
    num_cpgs, 
    site_type,
    N,
    min_dist) {
    # choose the first num_cpgs CpGs from corr_df
    cpg_names <- colnames(corr_df)[1:num_cpgs]
    
    # initialize empty result lists
    all_annot_result_dfs <- vector("list", length = length(cpg_names))
    cpg_modules <- vector("list", length = length(cpg_names))
    cpg_names_to_remove <- vector("list", length = length(cpg_names))
    # loop through each CpG
    for (i in 1:length(cpg_names)) {
        # chooose the top N correlated or random CpGs
        if (site_type == 'corr'){
            cpg_module_names <- get_top_correlated_distal_cpgs(
                corr_df = corr_df,
                dist_df = dist_df,
                cpg_name = cpg_names[i],
                N = N,
                min_dist = min_dist
                )
        } else if (site_type == 'close') {
            cpg_module_names <- get_close_cpgs(
                dist_df = dist_df,
                cpg_name = cpg_names[i],
                N = N
                )
        } else { # site_type == 'random'
            # choose from all sites between min_dist and min_dist*10 away
            cpg_module_names <- get_random_close_cpgs(
                dist_df = dist_df,
                cpg_name = cpg_names[i],
                N = N,
                min_dist = min_dist
                )
            #cpg_module_names <- sample(colnames(corr_df), N)
        }

        # catch errors 
        tryCatch ({
            # use miss methyl to annototate with GO terms
            annot_result_df <- gometh(
                sig.cpg=cpg_module_names, # cpgs to test
                all.cpg=colnames(corr_df), # background 
                collection="GO", # ontology
                plot.bias=FALSE # do not plot bias
                )
        }, error = function(e) {
            # if error, return data frame with columns: ONTOLOGY,TERM,N,DE,P.DE,FDR	
            annot_result_df <- data.frame(
                ONTOLOGY = NA,
                TERM = NA,
                N = NA,
                DE = NA,
                P.DE = NA,
                FDR = 1,
                cpg_name = NA
                )
            # add to names to remove 
            cpg_names_to_remove[[i]] <- cpg_names[i]
            print(paste0("error: ", e))
            print(paste0("cpg_name: ", cpg_names[i]))

            # continue to next iteration
            return(NULL)
        })
        
        # add cpg_module_names to cpg_modules result list
        cpg_modules[[i]] <- cpg_module_names
        # cpg name add to result
        annot_result_df$cpg_name <- cpg_names[i]
        # subset to significant results
        annot_result_df <- annot_result_df[annot_result_df$FDR < .05,]
        # # add to result lists
        all_annot_result_dfs[[i]] <- annot_result_df
        # print progress if multiple of 10
        if (i %% 10 == 0) {print(i)}
    }
    # combine all_annot_result_dfs into a single data frame
    all_annot_result_df <- do.call(rbind, all_annot_result_dfs)
    # remove cpgs that caused errors from cpg_names
    cpg_names <- cpg_names[!cpg_names %in% unlist(cpg_names_to_remove)]
    # name the cpg_modules result list with the cpg_names
    names(cpg_modules) <- cpg_names
    # convert cpg_modules to json
    cpg_modules_json <- toJSON(cpg_modules, auto_unbox = TRUE)

    # return with names
    return_list <- list("df" = all_annot_result_df, "cpg_modules" = cpg_modules_json)
    return(return_list)
}


# read in correlation data
corr_fn <- "/cellar/users/zkoch/methylation_and_mutation/dependency_files/all_corrs_chrom1_PANCAN.parquet"
# last column, called cpg_name, should be the row names
corr_df <-  read_parquet(corr_fn, row.names = "cpg_name")
# Set the row names of 'df' to be the values from cpg_name
to_be_rownames <- corr_df$cpg_name
corr_df <- subset(corr_df, select = -cpg_name)
rownames(corr_df) <- to_be_rownames

# and distance data
dist_fn <- "/cellar/users/zkoch/methylation_and_mutation/dependency_files/all_distances_chrom1_PANCAN.parquet"
dist_df <-  read_parquet(dist_fn)
# rename #id to id
colnames(dist_df)[colnames(dist_df) == "#id"] <- "id"
to_be_rownames <- dist_df$id
# remove the #id column
dist_df <- subset(dist_df, select = -id)
# set the row names
rownames(dist_df) <- to_be_rownames

# get intersection of CpGs in both data frames
common_cpgs <- intersect(colnames(corr_df), colnames(dist_df))
# subset both data frames to only include common CpGs
corr_df <- corr_df[common_cpgs,common_cpgs]
dist_df <- dist_df[common_cpgs,common_cpgs]
# set rownames to be the CpG names
rownames(corr_df) <- common_cpgs
rownames(dist_df) <- common_cpgs

# read in command line arguments
args <- commandArgs(trailingOnly = TRUE)
# get the number of CpGs to run the annotation pipeline on
num_cpgs <- as.numeric(args[1])
# get the number of top correlated CpGs to run the annotation pipeline on
N <- as.numeric(args[2])
# get the minimum distance between the CpG site and the top correlated CpGs
min_dist <- as.numeric(args[3])
# get the output directory
output_dir <- args[4]

# print arguments
print(paste0("num_cpgs: ", num_cpgs))
print(paste0("N: ", N))
print(paste0("min_dist: ", min_dist))
print(paste0("output_dir: ", output_dir))

# run close annotation
close_annot_results <- run_annotation(
    corr_df = corr_df,
    dist_df = dist_df,
    num_cpgs = num_cpgs, 
    site_type = 'close', 
    N = N,
    min_dist = min_dist
    )
print("finished close annotation")
close_annot_results_fn <- paste0(output_dir, "/close_annot_results_df.parquet")
write_parquet(close_annot_results$df, close_annot_results_fn)
close_cpg_modules_output_fn <- paste0(output_dir, "/cpg_modules_close.json")
writeLines(close_annot_results$cpg_modules, close_cpg_modules_output_fn)


# run random annotation
random_annot_results <- run_annotation(
    corr_df = corr_df,
    dist_df = dist_df,
    num_cpgs = num_cpgs, 
    site_type = 'random', 
    N = N,
    min_dist = min_dist
    )
print("finished random annotation")
random_annot_results_fn <- paste0(output_dir, "/random_annot_results_df.parquet")
write_parquet(random_annot_results$df, random_annot_results_fn)
random_cpg_modules_output_fn <- paste0(output_dir, "/cpg_modules_random.json")
writeLines(random_annot_results$cpg_modules, random_cpg_modules_output_fn)


# run annotation
annot_results <- run_annotation(
    corr_df = corr_df,
    dist_df = dist_df,
    num_cpgs = num_cpgs, 
    site_type = 'corr',
    N = N,
    min_dist = min_dist
    )
print("finished corr annotation")
annot_results_fn <- paste0(output_dir, "/annot_results_df.parquet")
write_parquet(annot_results$df, annot_results_fn)
cpg_modules_output_fn <- paste0(output_dir, "/cpg_modules_correlation.json")
writeLines(annot_results$cpg_modules, cpg_modules_output_fn)


# save annot_results$df to output_dir
# combine output_dir with annot_results_df.parquet



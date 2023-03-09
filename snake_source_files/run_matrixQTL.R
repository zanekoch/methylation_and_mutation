# Load the MatrixEQTL package
library("MatrixEQTL")
# Parse the filenames from the commandline arguments
SNP_file_name = commandArgs(trailingOnly=TRUE)[1]
methyl_file_name = commandArgs(trailingOnly=TRUE)[2]
covariates_file_name = commandArgs(trailingOnly=TRUE)[3]
# make the output file name from the output directory and named SNP_file_name but with a .meqtl extension
output_file_name = paste(SNP_file_name, ".meqtl", sep = "")

# Load the SNP data
som_muts = SlicedData$new()
som_muts$LoadFile(SNP_file_name, skipRows = 1, skipColumns = 1, sliceSize = 1000, omitCharacters = "NA", delimiter = ",")
# Load the methyl data
methyl = SlicedData$new()
methyl$LoadFile(methyl_file_name, skipRows = 1, skipColumns = 1, sliceSize = 2000, omitCharacters = "NA", delimiter = ",")
# Load the covariates data
covariates = SlicedData$new()
covariates$LoadFile(covariates_file_name, skipRows = 1, skipColumns = 1, omitCharacters = "NA", delimiter = ",")

# get the column names from each 
som_muts_colnames = colnames(som_muts)
covariates_colnames = colnames(covariates)
methyl_colnames = colnames(methyl)

# subset methyl and cov to only include the columns that are in som_muts
covariates$ColumnSubsample( which(covariates_colnames %in% som_muts_colnames) )
methyl$ColumnSubsample( which(methyl_colnames %in% som_muts_colnames) )


# Run the Matrix_eQTL_engine
me = Matrix_eQTL_engine(
    snps = som_muts,
    gene = methyl,
    cvrt = covariates,
    output_file_name = output_file_name,
    pvOutputThreshold = 5e-8,
    useModel = modelLINEAR,
    errorCovariance = numeric(),
    verbose = TRUE,
    pvalue.hist = FALSE,
    min.pv.by.genesnp = FALSE,
    noFDRsaveMemory = FALSE)

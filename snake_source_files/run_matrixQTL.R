# Load the MatrixEQTL package
library("MatrixEQTL")
# Parse the filenames from the commandline arguments
SNP_file_name = commandArgs(trailingOnly=TRUE)[1]
expression_file_name = commandArgs(trailingOnly=TRUE)[2]
# make the output file name from the output directory and named SNP_file_name but with a .meqtl extension
output_file_name = paste(SNP_file_name, ".meqtl", sep = "")

# Load the SNP data
snps = SlicedData$new()
snps$LoadFile(SNP_file_name, skipRows = 1, skipColumns = 1, sliceSize = 1000, omitCharacters = "NA", delimiter = ",")
# Load the methyl data
methyl = SlicedData$new()
methyl$LoadFile(expression_file_name, skipRows = 1, skipColumns = 1, sliceSize = 2000, omitCharacters = "NA", delimiter = ",")
# Load the covariates data
covariates_file_name = character()
covariates = SlicedData$new()

# Run the Matrix_eQTL_engine
me = Matrix_eQTL_engine(
    snps = snps,
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

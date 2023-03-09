# Load the MatrixEQTL package
library("MatrixEQTL")
# Parse the filenames from the commandline arguments
som_mut_file_name = commandArgs(trailingOnly=TRUE)[1]
methyl_file_name = commandArgs(trailingOnly=TRUE)[2]
covariates_file_name = commandArgs(trailingOnly=TRUE)[3]
# read in the partition number as an int (0-based)
partition_num = as.integer(commandArgs(trailingOnly=TRUE)[4])
# number of partitions
number_of_partitions = as.integer(commandArgs(trailingOnly=TRUE)[5])

# print the partition number
print("partion number:")
print(partition_num)

# make the output file name from the output directory and named SNP_file_name but with a .meqtl extension
# drop the .csv.gz extension from the som_mut_file_name
som_mut_file_name_strip = gsub(".csv.gz", "", som_mut_file_name)
# add _partition_num to the end of the som_mut_file_name
output_file_name = paste(som_mut_file_name_strip, "_partition_", partition_num,  ".meqtl", sep = "")
print("writing to file:")
print(output_file_name)

# Load the SNP data
som_muts = SlicedData$new()
som_muts$LoadFile(som_mut_file_name, skipRows = 1, skipColumns = 1, sliceSize = 1000, omitCharacters = "NA", delimiter = ",")
# get number of rows in the mut data
num_rows = length(som_muts$GetAllRowNames())
num_rows_per_partition = num_rows / number_of_partitions
# get this partition's start and end rows 
start_row = ((partition_num) * num_rows_per_partition) + 1
end_row = (partition_num + 1) * num_rows_per_partition
# subset the som_muts data to only include this partition's rows
som_muts$RowReorder( seq(start_row, end_row) )

print("subset to rows:")
print(start_row)
print(end_row)
# print the number of rows in the som_muts data now
print("Leaving som_muts rows:")
print(length(som_muts$GetAllRowNames()))
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

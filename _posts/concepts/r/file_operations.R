# File Operations in R

# Create a directory for our examples
dir_path <- "example_files"
if (!dir.exists(dir_path)) {
    dir.create(dir_path)
    print(paste("Created directory:", dir_path))
}

# ---- Writing Files ----

# 1. Writing a text file
sample_text <- c("Hi My name is Masum", "This is line 2", "This is line 3")
writeLines(sample_text, file.path(dir_path, "sample.txt"))
print("\nCreated sample.txt")

# 2. Writing a CSV file
employee_data <- data.frame(
    Name = c("John", "Alice", "Bob"),
    Age = c(30, 25, 35),
    Salary = c(50000, 60000, 45000)
)
write.csv(employee_data, file.path(dir_path, "employees.csv"), row.names = FALSE)
print("Created employees.csv")

# ---- Reading Files ----

# 1. Reading a text file
print("\nReading text file (sample.txt):")
text_content <- readLines(file.path(dir_path, "sample.txt"))
print(text_content)

# 2. Reading a CSV file
print("\nReading CSV file (employees.csv):")
csv_content <- read.csv(file.path(dir_path, "employees.csv"))
print(csv_content)

# ---- File Information ----

# 1. Check if file exists
file_path <- file.path(dir_path, "sample.txt")
print(paste("\nDoes sample.txt exist?", file.exists(file_path)))

# 2. Get file information
file_info <- file.info(file_path)
print("\nFile information for sample.txt:")
print(file_info)

# 3. List files in directory
print("\nFiles in the directory:")
files <- list.files(dir_path)
print(files)

# ---- File Manipulation ----

# 1. Append to a text file
cat("\nThis is an appended line", file = file_path, append = TRUE)
print("\nAfter appending to sample.txt:")
print(readLines(file_path))

# 2. Copy a file
file_copy <- file.path(dir_path, "sample_copy.txt")
file.copy(file_path, file_copy)
print(paste("\nCopied file exists?", file.exists(file_copy)))

# 3. Rename a file
new_name <- file.path(dir_path, "sample_renamed.txt")
file.rename(file_copy, new_name)
print(paste("Renamed file exists?", file.exists(new_name)))

# ---- Working with Binary Files ----

# 1. Writing binary data
binary_data <- raw(10)  # Create 10 bytes of zero
binary_file <- file.path(dir_path, "binary_sample")
writeBin(binary_data, binary_file)
print("\nCreated binary file")

# 2. Reading binary data
read_binary <- readBin(binary_file, raw(), n = 10)
print("Read binary data:")
print(read_binary)

# ---- File Paths and Properties ----

# 1. Get current working directory
print("\nCurrent working directory:")
print(getwd())

# 2. Get absolute path
print("\nAbsolute path of sample.txt:")
print(normalizePath(file_path))

# 3. Get file extension
print("\nFile extensions:")
print(tools::file_ext(files))

# ---- Clean up ----
# Uncomment the following line to delete the example directory and all its contents
# unlink(dir_path, recursive = TRUE)

print("\nFile operations completed successfully!")

# ---- Error Handling Example ----
print("\nTrying to read a non-existent file (with error handling):")
tryCatch({
    content <- readLines("non_existent_file.txt")
}, error = function(e) {
    print(paste("Error:", e$message))
})

# ---- File Connections ----
print("\nUsing file connections:")
conn <- file(file.path(dir_path, "connection_example.txt"), "w")
writeLines(c("Line 1 using connection", "Line 2 using connection"), conn)
close(conn)

# Read using connection
conn <- file(file.path(dir_path, "connection_example.txt"), "r")
connection_content <- readLines(conn)
close(conn)
print("Content read using connection:")
print(connection_content)

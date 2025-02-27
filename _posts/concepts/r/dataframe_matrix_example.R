# Example of Data Frames and Matrices in R

# ---- MATRICES ----
# Creating a matrix
# Matrix can only contain data of the same type (numeric, character, etc.)
print("Creating a 3x3 matrix:")
mat1 <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9), nrow = 3, ncol = 3)
print(mat1)

# Matrix with row and column names
print("\nMatrix with row and column names:")
rownames(mat1) <- c("Row1", "Row2", "Row3")
colnames(mat1) <- c("Col1", "Col2", "Col3")
print(mat1)

# Matrix operations
print("\nMatrix multiplication:")
mat2 <- matrix(c(2, 0, 1, 1, 3, 2, 0, 1, 4), nrow = 3, ncol = 3)
print("Second matrix:")
print(mat2)
print("Result of matrix multiplication:")
print(mat1 %*% mat2)

# Matrix element-wise operations
print("\nElement-wise addition:")
print(mat1 + mat2)

# Accessing matrix elements
print("\nAccessing matrix elements:")
print("First row of mat1:")
print(mat1[1,])
print("Second column of mat1:")
print(mat1[,2])
print("Element at position (2,3) of mat1:")
print(mat1[2,3])

# ---- DATA FRAMES ----
# Creating a data frame
# Data frames can contain different types of data
print("\nCreating a data frame:")
df <- data.frame(
    Name = c("John", "Alice", "Bob", "Carol"),
    Age = c(25, 30, 35, 28),
    Salary = c(50000, 60000, 75000, 55000),
    Department = c("IT", "HR", "Finance", "IT"),
    stringsAsFactors = FALSE
)
print(df)

# Structure of the data frame
print("\nStructure of the data frame:")
str(df)

# Basic statistics of numeric columns
print("\nSummary statistics:")
summary(df)

# Accessing data frame elements
print("\nAccessing data frame elements:")
print("Names column:")
print(df$Name)
print("\nFirst two rows:")
print(df[1:2,])
print("\nAge and Salary columns:")
print(df[,c("Age", "Salary")])

# Filtering data
print("\nFiltering: Employees in IT department:")
it_employees <- df[df$Department == "IT",]
print(it_employees)

# Adding a new column
print("\nAdding a new column (Bonus):")
df$Bonus <- df$Salary * 0.1
print(df)

# Aggregate functions
print("\nAverage salary by department:")
avg_salary <- aggregate(Salary ~ Department, data = df, FUN = mean)
print(avg_salary)

# Converting between matrix and data frame
print("\nConverting numeric columns to matrix:")
salary_matrix <- as.matrix(df[,c("Salary", "Bonus")])
print("Salary and Bonus as matrix:")
print(salary_matrix)

# Demonstrating the difference between matrix and data frame
print("\nKey differences between matrix and data frame:")
print("Matrix can only store one data type:")
print(class(mat1))
print("Data frame can store multiple data types:")
print(sapply(df, class))

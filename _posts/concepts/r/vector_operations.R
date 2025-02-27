# Vector Operations in R

# ---- Creating Vectors ----
# Numeric vector
numbers <- c(1, 2, 3, 4, 5)
print("Numeric vector:")
print(numbers)

# Character vector
fruits <- c("apple", "banana", "orange", "mango")
print("\nCharacter vector:")
print(fruits)

# Logical vector
logical_values <- c(TRUE, FALSE, TRUE, TRUE)
print("\nLogical vector:")
print(logical_values)

# Sequence vector
sequence <- 1:5  # Creates vector from 1 to 5
print("\nSequence vector:")
print(sequence)

# Vector with pattern
pattern <- seq(from = 0, to = 10, by = 2)
print("\nPattern vector (even numbers):")
print(pattern)

# ---- Vector Arithmetic ----
vec1 <- c(1, 2, 3, 4)
vec2 <- c(5, 6, 7, 8)

print("\nVector arithmetic:")
print("Addition:")
print(vec1 + vec2)

print("Subtraction:")
print(vec2 - vec1)

print("Multiplication:")
print(vec1 * vec2)

print("Division:")
print(vec2 / vec1)

# ---- Vector Functions ----
numbers <- c(15, 8, 3, 22, 7, 10)
print("\nOperations on vector:")
print("Original vector:")
print(numbers)

print("Sum:")
print(sum(numbers))

print("Mean:")
print(mean(numbers))

print("Maximum:")
print(max(numbers))

print("Minimum:")
print(min(numbers))

print("Sorted vector:")
print(sort(numbers))

print("Length of vector:")
print(length(numbers))

# ---- Vector Indexing ----
fruits <- c("apple", "banana", "orange", "mango", "grape")
print("\nVector indexing:")
print("Original vector:")
print(fruits)

print("First element:")
print(fruits[1])

print("Elements 2 to 4:")
print(fruits[2:4])

print("Multiple specific elements:")
print(fruits[c(1, 3, 5)])

# ---- Vector Filtering ----
ages <- c(25, 18, 35, 22, 45, 30, 28)
print("\nVector filtering:")
print("Ages vector:")
print(ages)

print("Ages greater than 30:")
print(ages[ages > 30])

print("Ages between 25 and 35:")
print(ages[ages >= 25 & ages <= 35])

# ---- Vector Modification ----
numbers <- c(1, 2, 3, 4, 5)
print("\nVector modification:")
print("Original vector:")
print(numbers)

# Replace single element
numbers[3] <- 10
print("After replacing third element:")
print(numbers)

# Replace multiple elements
numbers[c(1, 5)] <- c(100, 500)
print("After replacing first and last elements:")
print(numbers)

# ---- Vector Recycling ----
vec1 <- c(1, 2, 3, 4)
vec2 <- c(10, 20)
print("\nVector recycling:")
print("Vector 1:")
print(vec1)
print("Vector 2:")
print(vec2)
print("Addition with recycling:")
print(vec1 + vec2)  # vec2 will be recycled to c(10, 20, 10, 20)

# ---- Named Vectors ----
scores <- c(math = 90, science = 85, history = 88)
print("\nNamed vector:")
print(scores)
print("Accessing by name:")
print(scores["math"])

# ---- Vector Type Conversion ----
print("\nVector type conversion:")
# Numeric to character
num_vec <- c(1, 2, 3)
char_vec <- as.character(num_vec)
print("Numeric to character:")
print(char_vec)

# Character to numeric
char_nums <- c("1", "2", "3")
num_conv <- as.numeric(char_nums)
print("Character to numeric:")
print(num_conv)

# Logical to numeric
log_vec <- c(TRUE, FALSE, TRUE)
num_log <- as.numeric(log_vec)
print("Logical to numeric:")
print(num_log)

# ---- Vector Operations with NA ----
vec_with_na <- c(1, NA, 3, NA, 5)
print("\nHandling NA values:")
print("Vector with NA:")
print(vec_with_na)
print("Sum (with na.rm = TRUE):")
print(sum(vec_with_na, na.rm = TRUE))
print("Mean (with na.rm = TRUE):")
print(mean(vec_with_na, na.rm = TRUE))

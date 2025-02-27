# Lists in R - A Comprehensive Guide

# ---- Creating Lists ----

# 1. Basic list with different types of elements
basic_list <- list(
    numbers = c(1, 2, 3, 4, 5),
    text = "Hello R",
    logical = TRUE,
    vector = c("apple", "banana", "orange")
)
print("Basic list:")
print(basic_list)

# 2. Nested list
nested_list <- list(
    personal = list(
        name = "John Doe",
        age = 30,
        hobbies = c("reading", "gaming")
    ),
    professional = list(
        company = "Tech Corp",
        position = "Developer",
        skills = c("R", "Python", "SQL")
    )
)
print("\nNested list:")
print(nested_list)

# ---- Accessing List Elements ----

# 1. Using $ notation
print("\nAccessing elements using $ notation:")
print("Numbers from basic list:")
print(basic_list$numbers)

# 2. Using index notation
print("\nAccessing elements using index notation:")
print("Text from basic list:")
print(basic_list[[2]])  # Access second element

# 3. Accessing nested elements
print("\nAccessing nested elements:")
print("Name from nested list:")
print(nested_list$personal$name)
print("Skills from nested list:")
print(nested_list$professional$skills)

# ---- List Manipulation ----

# 1. Adding new elements
basic_list$new_element <- "New Value"
print("\nList after adding new element:")
print(basic_list)

# 2. Modifying elements
basic_list$numbers <- c(10, 20, 30)
print("\nList after modifying numbers:")
print(basic_list)

# 3. Removing elements
basic_list$logical <- NULL
print("\nList after removing logical element:")
print(basic_list)

# ---- List Functions ----

# 1. Length of list
print("\nLength of basic list:")
print(length(basic_list))

# 2. Names of list elements
print("\nNames of list elements:")
print(names(basic_list))

# 3. Check if element exists
print("\nChecking if elements exist:")
print("'text' exists:")
print("text" %in% names(basic_list))
print("'logical' exists:")
print("logical" %in% names(basic_list))

# ---- List Operations ----

# 1. Combining lists
list1 <- list(a = 1, b = 2)
list2 <- list(c = 3, d = 4)
combined_list <- c(list1, list2)
print("\nCombined lists:")
print(combined_list)

# 2. Converting list to vector
vector_list <- list(1, 2, 3, 4, 5)
converted_vector <- unlist(vector_list)
print("\nConverted list to vector:")
print(converted_vector)

# ---- List with Different Data Structures ----

# Creating a complex list with different data structures
complex_list <- list(
    # Vector
    numeric_vector = c(1, 2, 3, 4, 5),
    
    # Matrix
    matrix_data = matrix(1:9, nrow = 3),
    
    # Data Frame
    df_data = data.frame(
        name = c("John", "Alice", "Bob"),
        age = c(25, 30, 35)
    )
)
print("\nComplex list with different data structures:")
print(complex_list)

# ---- List Iteration ----

# 1. Using lapply (returns a list)
print("\nUsing lapply to multiply numbers by 2:")
result_lapply <- lapply(complex_list$numeric_vector, function(x) x * 2)
print(result_lapply)

# 2. Using sapply (simplifies result)
print("\nUsing sapply to multiply numbers by 2:")
result_sapply <- sapply(complex_list$numeric_vector, function(x) x * 2)
print(result_sapply)

# ---- List Filtering ----

# Creating a list with numbers
number_list <- list(a = 10, b = 5, c = 15, d = 8)

# Filtering elements greater than 7
filtered_list <- number_list[sapply(number_list, function(x) x > 7)]
print("\nFiltered list (elements > 7):")
print(filtered_list)

# ---- Error Handling with Lists ----

# Safe list access with default values
safe_access <- function(lst, element, default = NA) {
    if (element %in% names(lst)) {
        return(lst[[element]])
    } else {
        return(default)
    }
}

print("\nSafe list access:")
print("Accessing existing element:")
print(safe_access(basic_list, "text"))
print("Accessing non-existing element:")
print(safe_access(basic_list, "nonexistent"))

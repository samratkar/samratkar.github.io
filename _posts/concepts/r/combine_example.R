# Examples of c() function in R

# 1. Creating a numeric vector
numbers <- c(1, 2, 3, 4, 5)
print("Numeric vector:")
print(numbers)

# 2. Creating a character vector
names <- c("John", "Alice", "Bob")
print("\nCharacter vector:")
print(names)

# 3. Mixing different types (note: R will convert all to the same type)
mixed <- c(1, "hello", 3)
print("\nMixed vector (converts to character):")
print(mixed)

# 4. Combining existing vectors
vector1 <- c(1, 2, 3)
vector2 <- c(4, 5, 6)
combined <- c(vector1, vector2)
print("\nCombining two vectors:")
print(combined)

# 5. Creating a sequence
sequence <- c(1:5)  # Same as c(1, 2, 3, 4, 5)
print("\nSequence using c():")
print(sequence)

# 6. Using c() with logical values
logical_vector <- c(TRUE, FALSE, TRUE)
print("\nLogical vector:")
print(logical_vector)

# 7. Naming elements in a vector
named_vector <- c(first = 1, second = 2, third = 3)
print("\nNamed vector:")
print(named_vector)

# Mutable and Immutable Data Types
# Immutable Data Types - int, string, tuple
# Example 1: Integer
sugar = 4
print(f"id(sugar) before: {id(sugar)} --> value: {sugar}")
sugar += 2
print(f"id(sugar) after: {id(sugar)} --> value: {sugar}")
print ("Insight : Integer is immutable, so a new object is created when we modify its value.")

# Example 2: String
greeting = "Hello"
print(f"id(greeting) before: {id(greeting)} --> value: {greeting}")
greeting += ", World!"
print(f"id(greeting) after: {id(greeting)} --> value: {greeting}")
print ("Insight : String is immutable, so a new object is created when we modify its value.")   

# Mutable Data Types - list, dictionary, set
# Example 1: List
fruits = ["apple", "banana"]
print(f"id(fruits) before: {id(fruits)} --> value: {fruits}")
fruits.append("cherry")
print(f"fruits.append('cherry')  # modified value")
print(f"id(fruits) after: {id(fruits)} --> value: {fruits}")
print ("Insight : List is mutable, so the same object is modified when we change its contents.")    

# Example 2: Dictionary
person = {"name": "Alice", "age": 30}   
print(f"id(person) before: {id(person)} --> value: {person}")
person["age"] = 31
print(f"id(person) after: {id(person)} --> value: {person}")    
print ("Insight : Dictionary is mutable, so the same object is modified when we change its contents.")

# Example 3: Set
colors = {"red", "green"}   
print(f"id(colors) before: {id(colors)} --> value: {colors}")
colors.add("blue")
print(f"id(colors) after: {id(colors)} --> value: {colors}")
print ("Insight : Set is mutable, so the same object is modified when we change its contents.")

# Conclusion
print("In summary, immutable data types create new objects when modified, while mutable data types modify the existing object.")

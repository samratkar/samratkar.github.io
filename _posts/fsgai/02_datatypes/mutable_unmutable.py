# Mutable and Immutable Data Types
# Immutable Data Types
# Example 1: Integer
sugar = 4
print(f"sugar = 4  # initial value")
print(f"id(sugar) before: {id(sugar)} --> value: {sugar}")
sugar += 2
print(f"sugar += 2  # modified value")
print(f"id(sugar) after: {id(sugar)} --> value: {sugar}")
print ("Insight : Integer is immutable, so a new object is created when we modify its value.")

# Example 2: String
greeting = "Hello"
print(f"greeting = 'Hello'  # initial value")
print(f"id(greeting) before: {id(greeting)} --> value: {greeting}")
greeting += ", World!"
print(f"greeting += ', World!'  # modified value")
print(f"id(greeting) after: {id(greeting)} --> value: {greeting}")
print ("Insight : String is immutable, so a new object is created when we modify its value.")   

# Mutable Data Types
# Example 1: List
fruits = ["apple", "banana"]
print(f"fruits = ['apple', 'banana']  # initial value")
print(f"id(fruits) before: {id(fruits)} --> value: {fruits}")
fruits.append("cherry")
print(f"fruits.append('cherry')  # modified value")
print(f"id(fruits) after: {id(fruits)} --> value: {fruits}")
print ("Insight : List is mutable, so the same object is modified when we change its contents.")    

# Example 2: Dictionary
person = {"name": "Alice", "age": 30}   
print(f"person = {{'name': 'Alice', 'age': 30}}  # initial value")
print(f"id(person) before: {id(person)} --> value: {person}")
person["age"] = 31
print(f"person['age'] = 31  # modified value")
print(f"id(person) after: {id(person)} --> value: {person}")    
print ("Insight : Dictionary is mutable, so the same object is modified when we change its contents.")

# Example 3: Set
colors = {"red", "green"}   
print(f"colors = {{'red', 'green'}}  # initial value")
print(f"id(colors) before: {id(colors)} --> value: {colors}")
colors.add("blue")
print(f"colors.add('blue')  # modified value")
print(f"id(colors) after: {id(colors)} --> value: {colors}")
print ("Insight : Set is mutable, so the same object is modified when we change its contents.")

# Conclusion
print("In summary, immutable data types create new objects when modified, while mutable data types modify the existing object.")

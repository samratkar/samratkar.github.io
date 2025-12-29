milk_liters = 7
servings = 4

# standard division
milk_per_serving = milk_liters / servings
print(f"Milk per serving: {milk_per_serving} liters")  # Output: 1.75 liters

# floor division
milk_per_serving = milk_liters // servings
print(f"Milk per serving (floor division): {milk_per_serving} liters")

# modulus
total_cardamom_pods = 10
pods_per_cup = 3
leftover_pods = total_cardamom_pods % pods_per_cup 
print(f"Leftover cardamom pods: {leftover_pods} pods")  # Output: 1 pod

# exponentiation~
base_flavor_strength = 2
scale_factor = 3
powerful_flavor = base_flavor_strength ** scale_factor
print(f"Powerful flavor strength: {powerful_flavor}")  # Output: 8

total_leaves = 1_000_000_000
print(f"Total leaves: {total_leaves}")  # Output: 100000000

import sys
print(f"float info = {sys.float_info}")

from fractions import Fraction
from decimal import Decimal



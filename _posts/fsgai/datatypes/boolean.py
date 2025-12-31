is_boiling = True
stir_count = 5
total_actions = stir_count + is_boiling  # True is treated as 1 - upcasting boolean to integer
print(f"Total actions (stir_count + is_boiling): {total_actions}")  # Output: 6

milk_present = 0 # no milk 
print (f"Is milk present? {bool(milk_present)}")  # Output: False 
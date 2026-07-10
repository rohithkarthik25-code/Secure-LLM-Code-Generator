import os

def convert():
    # Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = current_dir  # Since this script runs inside the data directory
    dataset_path = os.path.join(data_dir, "code_dataset.txt")
    backup_path = os.path.join(data_dir, "code_dataset_prefix.txt")
    
    # Load original text
    with open(dataset_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Backup the original dataset
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    # Split by double newline to get code blocks
    # We clean carriage returns to prevent splitting issues on Windows
    blocks = content.replace("\r\n", "\n").split("\n\n")
    
    # Mappings dictionary for function descriptions
    mappings = {
        "add": "Write a function to add two numbers.",
        "subtract": "Write a function to subtract two numbers.",
        "multiply": "Write a function to multiply two numbers.",
        "divide": "Write a function to divide two numbers.",
        "factorial": "Write a function to calculate the factorial of a number.",
        "is_even": "Write a function to check if a number is even.",
        "is_odd": "Write a function to check if a number is odd.",
        "square": "Write a function to square a number.",
        "cube": "Write a function to cube a number.",
        "power": "Write a function to calculate the power of a base to an exponent.",
        "absolute": "Write a function to get the absolute value of a number.",
        "negate": "Write a function to negate a number.",
        "maximum": "Write a function to find the maximum of two numbers.",
        "minimum": "Write a function to find the minimum of two numbers.",
        "average": "Write a function to calculate the average of a list.",
        "total": "Write a function to sum all elements in a list.",
        "length": "Write a function to get the length of a list.",
        "reverse_list": "Write a function to reverse a list.",
        "reverse_string": "Write a function to reverse a string.",
        "is_palindrome": "Write a function to check if a string is a palindrome.",
        "to_uppercase": "Write a function to convert a string to uppercase.",
        "to_lowercase": "Write a function to convert a string to lowercase.",
        "capitalize_str": "Write a function to capitalize a string.",
        "count_chars": "Write a function to count characters in a string.",
        "count_words": "Write a function to count words in a string.",
        "strip_spaces": "Write a function to strip whitespace from a string.",
        "replace_char": "Write a function to replace characters in a string.",
        "starts_with": "Write a function to check if a string starts with a prefix.",
        "ends_with": "Write a function to check if a string ends with a suffix.",
        "contains": "Write a function to check if a string contains a substring.",
        "split_string": "Write a function to split a string by a separator.",
        "join_list": "Write a function to join a list of strings with a separator.",
        "first_element": "Write a function to get the first element of a list.",
        "last_element": "Write a function to get the last element of a list.",
        "second_element": "Write a function to get the second element of a list.",
        "flatten": "Write a function to flatten a nested list.",
        "unique": "Write a function to find the unique elements in a list.",
        "sort_list": "Write a function to sort a list in ascending order.",
        "sort_desc": "Write a function to sort a list in descending order.",
        "remove_duplicates": "Write a function to remove duplicates from a list preserving order.",
        "list_max": "Write a function to find the maximum value in a list.",
        "list_min": "Write a function to find the minimum value in a list.",
        "append_item": "Write a function to append an item to a list.",
        "remove_item": "Write a function to remove an item from a list.",
        "count_occurrences": "Write a function to count occurrences of an item in a list.",
        "index_of": "Write a function to find the index of an item in a list.",
        "slice_list": "Write a function to slice a list from a start index to an end index.",
        "merge_lists": "Write a function to merge two lists.",
        "zip_lists": "Write a function to zip two lists together.",
        "is_empty": "Write a function to check if a list is empty.",
        "clear_list": "Write a function to clear all elements from a list.",
        "repeat_str": "Write a function to repeat a string n times.",
        "is_digit": "Write a function to check if a string contains only digits.",
        "is_alpha": "Write a function to check if a string contains only letters.",
        "is_alphanumeric": "Write a function to check if a string is alphanumeric.",
        "char_at": "Write a function to get the character at index i in a string.",
        "find_index": "Write a function to find the index of a substring in a string.",
        "count_vowels": "Write a function to count vowels in a string.",
        "count_consonants": "Write a function to count consonants in a string.",
        "is_positive": "Write a function to check if a number is positive.",
        "is_negative": "Write a function to check if a number is negative.",
        "is_zero": "Write a function to check if a number is zero.",
        "clamp": "Write a function to clamp a number between a lower and upper bound.",
        "celsius_to_fahrenheit": "Write a function to convert Celsius to Fahrenheit.",
        "fahrenheit_to_celsius": "Write a function to convert Fahrenheit to Celsius.",
        "km_to_miles": "Write a function to convert kilometers to miles.",
        "miles_to_km": "Write a function to convert miles to kilometers.",
        "is_prime": "Write a function to check if a number is prime.",
        "gcd": "Write a function to find the greatest common divisor (GCD) of two numbers.",
        "lcm": "Write a function to find the least common multiple (LCM) of two numbers.",
        "fibonacci": "Write a function to calculate the nth Fibonacci number.",
        "sum_digits": "Write a function to calculate the sum of digits of a number.",
        "count_digits": "Write a function to count the number of digits in a number.",
        "is_perfect_square": "Write a function to check if a number is a perfect square.",
        "floor_div": "Write a function to perform floor division.",
        "modulo": "Write a function to calculate the modulo of two numbers.",
        "safe_divide": "Write a function to safely divide two numbers and return None if division by zero.",
        "swap": "Write a function to swap two variables.",
        "dict_keys": "Write a function to get the keys of a dictionary.",
        "dict_values": "Write a function to get the values of a dictionary.",
        "dict_get": "Write a function to get a value from a dictionary with a default fallback.",
        "dict_has_key": "Write a function to check if a key exists in a dictionary.",
        "merge_dicts": "Write a function to merge two dictionaries.",
        "invert_dict": "Write a function to invert keys and values of a dictionary.",
        "list_to_dict": "Write a function to create a dictionary from lists of keys and values.",
        "string_to_list": "Write a function to convert a string to a list of characters.",
        "int_to_binary": "Write a function to convert an integer to its binary string representation.",
        "binary_to_int": "Write a function to convert a binary string to an integer.",
        "int_to_hex": "Write a function to convert an integer to its hexadecimal representation.",
        "is_list": "Write a function to check if an object is a list.",
        "is_dict": "Write a function to check if an object is a dictionary.",
        "is_string": "Write a function to check if an object is a string.",
        "is_int": "Write a function to check if an object is an integer.",
        "is_float": "Write a function to check if an object is a float.",
        "truncate": "Write a function to truncate a string to length n.",
        "pad_left": "Write a function to pad a string on the left with a character to a width.",
        "pad_right": "Write a function to pad a string on the right with a character to a width.",
        "title_case": "Write a function to convert a string to title case."
    }
    
    # Process blocks
    new_blocks = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if "# --- Standalone Code Snippets ---" in block:
            continue
            
        instruction = ""
        code = block
        
        if block.startswith("def "):
            # Extract function name
            func_name = block.split("(")[0].replace("def ", "").strip()
            # Map function name to a natural instruction
            instruction = mappings.get(func_name, f"Write a function named {func_name}.")
            
        elif block.startswith("# Snippet "):
            # Extract snippet details
            lines = block.split("\n")
            header = lines[0]
            desc = header.split(":", 1)[1].strip()
            instruction = f"Write a code snippet to {desc.lower()}."
            # Remove the comment header from the code
            code = "\n".join(lines[1:])
            
        else:
            # Default fallback
            instruction = "Write a Python program."
            
        # Format the block according to the instruction format requirements
        formatted_block = f"Instruction: {instruction}\n\nCode:\n{code}\n\n<END>"
        new_blocks.append(formatted_block)
        
    # Join blocks together with double newlines
    new_content = "\n\n".join(new_blocks) + "\n"
    
    with open(dataset_path, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print(f"Dataset successfully converted! Saved to {dataset_path}")
    print(f"Original dataset backed up to {backup_path}")

if __name__ == "__main__":
    convert()

import os
import re

# Use os.getcwd() to get the path to the current directory
directory_path = os.getcwd()

# Regex pattern to match the "Total search time: xxxxx ms" line in your txt files
pattern = re.compile(r'Total search time: (\d+\.\d+) ms')

with open('origin_output.txt', 'w') as output_file:
    # Loop over all txt files in the directory
    output_file.write(f"filename,searchTime\n")
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            with open(os.path.join(directory_path, filename), 'r') as file:
                for line in file:
                    match = pattern.search(line)
                    # If the line matches the pattern, write the filename and the time value to origin_output.txt
                    if match:
                        output_file.write(f"{filename},{match.group(1)}\n")

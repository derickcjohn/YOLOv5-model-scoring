#To get the dict for class label mapping
original_file = r'..\classes.txt' #path to original classes
temp_file = r'path\to\classes.txt' #path to temp class, change the path here

original_lines = []
temp_lines = []

with open(original_file, 'r') as f:
    original_lines = f.readlines()

with open(temp_file, 'r') as f:
    temp_lines = f.readlines()

line_map = {}

for i, line in enumerate(temp_lines):
    match_found = False
    for j, temp_line in enumerate(original_lines):
        if line == temp_line:
            line_map[str(i)] = str(j)
            match_found = True  
            break

    if not match_found:
        line_map[str(i)] = str(len(original_lines) - 1)

print(line_map)

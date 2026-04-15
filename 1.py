import csv

# Define label rules
def get_label(second):
    if 1 <= second <= 10:
        return "Still"
    elif 11 <= second <= 20:
        return "Typing"
    elif 21 <= second <= 35:
        return "Flipping"
    elif 36 <= second <= 45:
        return "Typing"
    elif 46 <= second <= 60:
        return "Flipping"

# Create CSV
output_file_name = 'flip_labels.csv'
with open(output_file_name, "w", newline="") as file:
    writer = csv.writer(file)
    
    # Write header
    writer.writerow(["second", "label"])
    
    # Write rows
    for second in range(1, 61):
        writer.writerow([second, get_label(second)])

print(f"CSV file {output_file_name} created successfully.")

import os

def splitFileWithOverlap(file_path, saved_folder_path, items_per_file=60, overlap=30):
    """
    Splits a file into several new files, each containing a specified number of items,
    with a specified number of items overlapping with the previous file.
    
    Parameters:
    - file_path: The path to the original file to split.
    - saved_folder_path: The folder path for saving the new files.
    - items_per_file: The total number of items each new file should contain.
    - overlap: The number of items to overlap between consecutive files.
    
    Returns:
    - A list of paths to the newly created files.
    """

    # Read all lines from the original file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Initialize variables
    total_items = len(lines)
    new_files = []
    title = lines[0]
    start_index = 1
    
    while start_index < total_items:
        # Calculate end index; it's either `start_index + items_per_file` or the last item
        end_index = min(start_index + items_per_file, total_items)
        
        # Adjust the index for overlapping, after the first file
        if new_files != [] and end_index != total_items:
            start_index -= overlap
            end_index -= overlap
        
        # Ensure the last file has 150 items if possible, by adding items from the previous file
        if end_index - start_index < items_per_file:
            start_index = max(0, total_items - items_per_file)
        
        # Create a new file path
        part_number = len(new_files) + 1
        new_file_path = f"{saved_folder_path}/{part_number}.csv"
        new_files.append(new_file_path)
        
        # Write the subset of lines to the new file
        with open(new_file_path, 'w') as new_file:
            new_file.writelines(title)
            new_file.writelines(lines[start_index:end_index])
        
        # Update the start index for the next file
        start_index = end_index
    
    return new_files

# Splitting the file and getting the paths of the new files
saved_folder_path = './dataset/split'
file_path = './dataset/0050.TW.csv'
new_file_paths = splitFileWithOverlap(file_path, saved_folder_path, 30, 5)

# Print the paths for confirmation
print(new_file_paths)

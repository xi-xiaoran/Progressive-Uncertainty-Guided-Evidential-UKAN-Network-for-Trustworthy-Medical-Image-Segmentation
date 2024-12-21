import os
import shutil

# Set source folder and destination folder
# source_folder = 'data/Kvasir-SEG/images'  # Source folder path
# target_folder = 'data/Kvasir-SEG/Test'  # Target folder path
source_folder = 'data/Kvasir-SEG/masks'  # Source folder path
target_folder = 'data/Kvasir-SEG/GT'  # Target folder path

# Ensure that the target folder exists
os.makedirs(target_folder, exist_ok=True)

# Open the txt file containing the file name
with open('2.txt', 'r') as file_list:
    for file_name in file_list:
        # Remove line breaks at the end of a line
        file_name = file_name.rstrip('\n')
        # Build a complete source file path
        source_path = os.path.join(source_folder, file_name)
        source_path = source_path + '.jpg'
        # Build a complete target file path
        target_path = os.path.join(target_folder, file_name)
        target_path = target_path + '.jpg'
        # move file
        shutil.move(source_path, target_path)
import os
import shutil
from datetime import datetime

def move_old_files(src_dir, dest_dir):
    # Get today's date
    today = datetime.today().date()

    # Walk through the directory structure
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # Get the full file path
            file_path = os.path.join(root, file)
            
            # Get the file creation date
            file_creation_date = datetime.fromtimestamp(os.path.getctime(file_path)).date()
            
            # If the file was not created today, move it
            if file_creation_date != today:
                # Calculate the relative path from the src_dir
                relative_path = os.path.relpath(root, src_dir)
                dest_folder = os.path.join(dest_dir, relative_path)
                
                # Create the destination folder if it doesn't exist
                os.makedirs(dest_folder, exist_ok=True)
                
                # Destination file path
                dest_file = os.path.join(dest_folder, file)
                
                # Move the file
                shutil.move(file_path, dest_file)
                print(f"Moved: {file_path} to {dest_file}")

# Example usage
src_directory = "recordings"
dest_directory = "moved_files"
move_old_files(src_directory, dest_directory)

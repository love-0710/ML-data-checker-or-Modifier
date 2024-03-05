import os

def rename_files(dataset_path):
    for split_folder in os.listdir(dataset_path):
        split_path = os.path.join(dataset_path, split_folder)
        
        if os.path.isdir(split_path):
            for class_folder in os.listdir(split_path):
                class_path = os.path.join(split_path, class_folder)
                
                if os.path.isdir(class_path):
                    for filename in os.listdir(class_path):
                        # Create a new unique filename with prefix
                        new_filename = f"{split_folder}_{class_folder}_{filename}"
                        
                        # Create the full path for the old and new filenames
                        old_filepath = os.path.join(class_path, filename)
                        new_filepath = os.path.join(class_path, new_filename)
                        
                        # Rename the file
                        os.rename(old_filepath, new_filepath)

# Example usage
dataset_path = "Split_Dataset"
rename_files(dataset_path)

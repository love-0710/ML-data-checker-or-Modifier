import os
import shutil
import random

# Set the main directory name
main_dir_name = 'Datasets_project'

# Set the paths
original_dataset_dir = main_dir_name  # Update this with the actual subdirectory name
base_dir = os.path.join(main_dir_name, 'Split_Dataset')

# Create directories for train, validation, and test
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)

validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)

test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# List all classes of the dataset #modify it according to your classes and dataset
class_folders = [
    'Eczema',
    'Melanoma',
    'Atopic_Dermatitis',
    'Basal_Cell_Carcinoma',
    'Melanocytic_Nevi',
    'Benign_Keratosis_lesions',
    'Psoriasis_Lichen_Planus',
    'Seborrheic_Keratoses',
    'Fungal_Infections',
    'Viral_Infections'
]

# Loop through each class folder
for class_folder in class_folders:
    class_path = os.path.join(original_dataset_dir, class_folder)
    
    # List all images in the class folder
    images = [img for img in os.listdir(class_path) if img.endswith('.jpg')]
    
    # Shuffle the images
    random.shuffle(images)
    
    # Calculate the split indices
    test_split = int(0.05 * len(images))
    validation_split = int(0.2 * len(images) + test_split)
    
    # Split the images into train, validation, and test sets
    train_images = images[validation_split:]
    validation_images = images[test_split:validation_split]
    test_images = images[:test_split]
    
    # Create directories for the current class in train, validation, and test
    train_class_dir = os.path.join(train_dir, class_folder)
    os.makedirs(train_class_dir, exist_ok=True)
    
    validation_class_dir = os.path.join(validation_dir, class_folder)
    os.makedirs(validation_class_dir, exist_ok=True)
    
    test_class_dir = os.path.join(test_dir, class_folder)
    os.makedirs(test_class_dir, exist_ok=True)
    
    # Move images to their respective directories
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_class_dir, img)
        shutil.copyfile(src, dst)
        
    for img in validation_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(validation_class_dir, img)
        shutil.copyfile(src, dst)
        
    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_class_dir, img)
        shutil.copyfile(src, dst)

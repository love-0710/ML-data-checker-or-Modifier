import numpy as np
import json

# Load original JSON data from file
with open("development.json", "r") as file:
    original_json_data = json.load(file)

def augment_data(json_data):
    augmented_json_data = json_data.copy()  # Make a copy of the original data
    
    # Augment MFCC vectors and append to the original data
    augmented_mfcc = []
    for mfcc_list in json_data["mfcc"]:
        for mfcc_vector in mfcc_list:
            augmented_mfcc.append(augment_audio_function(mfcc_vector))
    augmented_json_data["mfcc"] += augmented_mfcc
    
    return augmented_json_data

def augment_audio_function(mfcc_vector):
    # Example augmentation function (add random Gaussian noise)
    noise = np.random.normal(0, 0.1, len(mfcc_vector))  # Mean=0, Standard deviation=0.1
    augmented_mfcc = mfcc_vector + noise
    
    return augmented_mfcc.tolist()  # Convert ndarray to list

# Augment the data
augmented_data = augment_data(original_json_data)

# Save augmented data to the original JSON file
with open("development.json", "w") as outfile:
    json.dump(augmented_data, outfile, indent=4)

print("Augmented data appended to development.json")

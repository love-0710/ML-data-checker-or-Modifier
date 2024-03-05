import json
import os
import librosa
import librosa.feature
import math
import pandas as pd

DATASET_PATH = r"development"    #directry path
JSON_PATH = "development.json"

#if you are working with the mfcc astraction the remove caption and metadata then run it.

CAPTIONS_CSV_PATH = "development\clotho_captions_development.csv"
METADATA_CSV_PATH = "development\clotho_metadata_development.csv"
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc_with_captions_and_metadata(dataset_path, json_path, captions_csv_path, metadata_csv_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
        "captions": [],  
        "metadata": []  # Initialize metadata as an empty list
    }
    num_samples_per_segment = SAMPLES_PER_TRACK // num_segments
    expected_n_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # Load captions and metadata CSV files
    captions_df = pd.read_csv(captions_csv_path, encoding='latin-1')
    metadata_df = pd.read_csv(metadata_csv_path, encoding='latin-1')

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            dirpath_comp = dirpath.split("//")
            semantic_label = dirpath_comp[-1]
            data["mapping"].append(semantic_label)
            print(f"Processing {semantic_label}")
            for f in filenames:
                # Getting Files 1 by 1
                file_path = os.path.join(dirpath, f)

                # Calculate Signal for each File
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                for s in range(num_segments):
                    # Divide each file into Segments
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    # Calculate 13 MFCC Vectors for each segment
                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], sr=sr, n_mfcc=n_mfcc,
                                                n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # Discard segments shorter than expected
                    if len(mfcc) == expected_n_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data['labels'].append(i - 1)

                        # Get captions and metadata for the current file and segment
                        caption_row = captions_df.loc[captions_df['file_name'] == f]
                        metadata_row = metadata_df.loc[metadata_df['file_name'] == f]

                        if not caption_row.empty:
                            captions = [caption_row[f'caption_{j}'].iloc[0] for j in range(1, 6)]
                            data['captions'].append(captions)
                        else:
                            data['captions'].append([])

                        if not metadata_row.empty:
                            metadata = {
                                'keywords': metadata_row['keywords'].iloc[0],
                                'sound_id': metadata_row['sound_id'].iloc[0],
                                'sound_link': metadata_row['sound_link'].iloc[0],
                                'start_end_samples': metadata_row['start_end_samples'].iloc[0],
                                'manufacturer': metadata_row['manufacturer'].iloc[0],
                                'license': metadata_row['license'].iloc[0]
                            }
                            data['metadata'].append(metadata)
                        else:
                            data['metadata'].append({})

                        print(f"{file_path},segment:{s + 1}")

    # Put all data in JSON File
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc_with_captions_and_metadata(DATASET_PATH, JSON_PATH, CAPTIONS_CSV_PATH, METADATA_CSV_PATH, num_segments=10)

import pandas as pd
import pickle
import os
from pathlib import Path
import numpy as np

def extract_eda_data(subject_folder):
    """
    Extract EDA data from a subject's pickle file and return as DataFrame
    """
    # Construct path to pickle file
    pickle_path = os.path.join(subject_folder, f"{os.path.basename(subject_folder)}.pkl")
    
    # Load pickle file
    with open(pickle_path, "rb") as f:
        data = pickle.load(f, encoding='latin1')
    
    # Extract EDA data and labels
    eda = data['signal']['wrist']['EDA']
    labels = data['label']
    
    # Resample labels from 700Hz to 4Hz
    # Calculate how many label samples correspond to one EDA sample
    samples_per_eda = len(labels) // len(eda)
    
    # Resample labels by taking the most common label in each window
    resampled_labels = []
    for i in range(0, len(labels), samples_per_eda):
        window = labels[i:i + samples_per_eda]
        # Use mode to get most common label in window
        most_common = np.bincount(window).argmax()
        resampled_labels.append(most_common)
    
    # Ensure we have the same number of samples
    resampled_labels = resampled_labels[:len(eda)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'eda_signal': eda.flatten(),
        'label': resampled_labels
    })
    
    # Add time column (4 Hz sampling rate)
    df['Time (s)'] = df.index / 4
    
    return df

def main():
    # Base directory containing WESAD data
    base_dir = "data/WESAD"
    
    # Get all subject folders (S2 to S17)
    subject_folders = [f for f in os.listdir(base_dir) if f.startswith('S') and f[1:].isdigit()]
    subject_folders = [f for f in subject_folders if f not in ['S1', 'S12']]  # Exclude S1 and S12
    
    # Process each subject
    for subject in subject_folders:
        try:
            print(f"Processing {subject}...")
            subject_path = os.path.join(base_dir, subject)
            
            # Extract EDA data
            eda_df = extract_eda_data(subject_path)
            
            # Save to CSV
            output_path = os.path.join(subject_path, f"{subject}_eda.csv")
            eda_df.to_csv(output_path, index=False)
            print(f"Saved EDA data for {subject} to {output_path}")
            
        except Exception as e:
            print(f"Error processing {subject}: {str(e)}")

if __name__ == "__main__":
    main()



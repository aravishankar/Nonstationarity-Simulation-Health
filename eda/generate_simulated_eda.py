import numpy as np
import pandas as pd
import neurokit2 as nk
import os
from datetime import datetime, timedelta

def generate_eda_signal(duration, scr_peaks, sampling_rate=4):
    """
    Generate EDA signal for a given duration and number of SCR peaks.
    
    Args:
        duration (int): Duration in seconds
        scr_peaks (int): Number of SCR peaks to generate
        sampling_rate (int): Sampling rate in Hz (default 4Hz as per WESAD)
        
    Returns:
        pd.DataFrame: DataFrame with EDA signal and timestamps
    """
    # Generate timestamps
    timestamps = [datetime(2024, 1, 1) + timedelta(seconds=i/sampling_rate) 
                 for i in range(int(duration * sampling_rate))]
    
    # Generate EDA signal using neurokit2
    eda = nk.eda_simulate(duration=duration, 
                         sampling_rate=sampling_rate,
                         scr_number=scr_peaks,
                         drift=0.01)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'eda_signal': eda,
        'label': 0  # Default label (will be updated for stress state)
    })
    
    return df

def generate_subject_data(subject_id):
    """
    Generate EDA data for a single subject with baseline and stress states.
    
    Args:
        subject_id (str): Subject ID (e.g., 'S1')
        
    Returns:
        pd.DataFrame: Combined DataFrame with baseline and stress states
    """
    # Generate baseline state (1174 seconds)
    baseline_scr = np.random.randint(1, 6)  # U(1,5)
    baseline_data = generate_eda_signal(duration=1174, scr_peaks=baseline_scr)
    baseline_data['label'] = 0  # Baseline label
    
    # Generate stress state (664 seconds)
    stress_scr = np.random.randint(6, 21)  # U(6,20)
    stress_data = generate_eda_signal(duration=664, scr_peaks=stress_scr)
    stress_data['label'] = 1  # Stress label
    
    # Combine the data
    combined_data = pd.concat([baseline_data, stress_data], ignore_index=True)
    
    return combined_data

def main():
    # Create output directory if it doesn't exist
    output_dir = 'data/simulated_eda'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load subject IDs from WESAD_learning_ids.csv
    learning_ids = pd.read_csv('data/WESAD_learning_ids.csv')['ID'].tolist()
    
    # Generate data for each subject in learning_ids
    for subject_id in learning_ids:
        print(f"Generating data for {subject_id}")
        
        # Generate subject data
        subject_data = generate_subject_data(subject_id)
        
        # Save to CSV
        output_file = os.path.join(output_dir, f'{subject_id}_eda.csv')
        subject_data.to_csv(output_file, index=False)
        print(f"Saved data to {output_file}")
        
        # Print some statistics
        print(f"Total duration: {len(subject_data)/4:.2f} seconds")
        print(f"Baseline duration: {len(subject_data[subject_data['label']==0])/4:.2f} seconds")
        print(f"Stress duration: {len(subject_data[subject_data['label']==1])/4:.2f} seconds")
        print("---")

if __name__ == "__main__":
    main() 
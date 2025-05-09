import pandas as pd
import os
import glob

def load_and_aggregate_results(folder_path):
    """Load and aggregate results from a folder containing model CSV files."""
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # Initialize dictionary to store aggregated results
    results = {
        'KNN': {'accuracy': [], 'f1': []},
        'LR': {'accuracy': [], 'f1': []},
        'RF': {'accuracy': [], 'f1': []},
        'SVM': {'accuracy': [], 'f1': []}
    }
    
    # Process each CSV file
    for file in csv_files:
        model_name = os.path.basename(file).replace('.csv', '')
        if model_name in results:
            df = pd.read_csv(file)
            results[model_name]['accuracy'].extend(df['Accuracy'].tolist())
            results[model_name]['f1'].extend(df['F1_Score'].tolist())
    
    # Calculate means for each metric
    aggregated = {}
    for model in results:
        aggregated[f"{model}_accuracy"] = sum(results[model]['accuracy']) / len(results[model]['accuracy'])
        aggregated[f"{model}_f1"] = sum(results[model]['f1']) / len(results[model]['f1'])
    
    return aggregated

def main():
    # Define base results folder
    results_folder = 'results'
    
    # Define folders to process within results
    folders = [
        'classification',
        'classification_baseline',
        'classification_normal',
        'classification_real'
    ]
    
    # Initialize list to store results
    all_results = []
    
    # Process each folder
    for folder in folders:
        folder_path = os.path.join(results_folder, folder)
        if os.path.exists(folder_path):
            results = load_and_aggregate_results(folder_path)
            results['Dataset'] = folder
            all_results.append(results)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns
    columns = ['Dataset']
    for model in ['KNN', 'LR', 'RF', 'SVM']:
        columns.extend([f'{model}_accuracy', f'{model}_f1'])
    
    df = df[columns]
    
    # Format numbers to 3 decimal places
    for col in df.columns:
        if col != 'Dataset':
            df[col] = df[col].round(3)
    
    # Create multi-level columns
    df.columns = pd.MultiIndex.from_tuples([
        ('Dataset', ''),
        ('KNN', 'Accuracy'),
        ('KNN', 'F1'),
        ('LR', 'Accuracy'),
        ('LR', 'F1'),
        ('RF', 'Accuracy'),
        ('RF', 'F1'),
        ('SVM', 'Accuracy'),
        ('SVM', 'F1')
    ])
    
    # Save to CSV in results folder
    output_path = os.path.join(results_folder, 'aggregated_results.csv')
    df.to_csv(output_path, index=False)
    
    # Print the table
    print("\nAggregated Classification Results:")
    print("=================================")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main() 
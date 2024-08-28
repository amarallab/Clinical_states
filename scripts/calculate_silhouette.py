from pathlib import Path
import sys
import os

script_dir = os.path.dirname(__file__)  
sys.path.append(os.path.join(script_dir, '../src'))

from preprocess import load_and_prepare_dataset, boot_sil

datasets = ['carpediem']
# datasets = ['mimic']
feature_sets = [
    'sofs', 'sofs+vtls', 'sofs+vtls+dmos', 'sofs+vtls+flgs', 'sofs+vtls+lbs1', 
    'sofs+vtls+lbs1+vents', 'sofs+vtls+vents+lbs2', 'sofs+vtls+vents+lbs'
]
# feature_sets = [
#     'sofs', 'sofs+vtls', 'sofs+vtls+dmos1', 'sofs+vtls+dmos', 'sofs+vtls+dmos1+flgs', 
#     'sofs+vtls+dmos1+lbs1', 
#     'sofs+vtls+dmos1+lbs2', #'sofs+vtls+dmos1+lbs', 
#     'sofs+vtls+dmos1+lbs1+vents'
# ]
# feature_sets = ['sofs', 'sofs+vtls', 'sofs+vtls+dmos', 
                # 'sofs+vtls+dmos+lbs1', 'sofs+vtls+dmos+lbs2', 'sofs+vtls+dmos+lbs']
imputation_strategies = [
    'Complete_cases', 
    'CarryLastForward', 'CarryLastForward_limit1', 'CarryLastForward_limit2',
    'MICE', 'FillZero', 'KNN'
]
# scalers = ['Raw', 'KBD', 'MM', 'SS', 'RS']
scalers = ['Raw', 'KBD+MM', 'KBD', 'MM', 'SS', 'RS']

holdout = 0.1

# Directory to save results
results_dir = Path('../data/calculated/silhouette/240524_holdout')
results_dir.mkdir(parents=True, exist_ok=True)

# Loop over configurations
for dataset in datasets:
    for feature_set in feature_sets:
        for imputation_strategy in imputation_strategies:
            for scaler_name in scalers:
                print(f"Processing: Dataset={dataset}, Feature_set={feature_set}, Imputation={imputation_strategy}, Scaler={scaler_name}")
                
                # Load and prepare the dataset
                try:
                    train_data, test_data = load_and_prepare_dataset(dataset, feature_set, imputation_strategy, scaler_name, holdout=holdout)
                    if test_data.empty:
                        print("Empty dataset after preprocessing. Skipping...")
                        continue

                    # Perform PCA and calculate silhouette scores
                    silhouette_results = boot_sil(dataset=dataset, feature_set=feature_set, \
                                                  imputation_strategy=imputation_strategy, scaler_name=scaler_name, \
                                                    n_boot=50, se=9, holdout=holdout)
                    
                    # Save the results
                    results_file = results_dir / f"sils_{dataset}_{feature_set}_{imputation_strategy}_{scaler_name}_train{(1-holdout)}.csv"
                    silhouette_results.to_csv(results_file, index=False)
                    print(f"Saved results to {results_file}")
                except Exception as e:
                    print(f"Error processing configuration: {e}")

print("Done.")
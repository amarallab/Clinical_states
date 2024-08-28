from pathlib import Path
import sys
import os

import numpy as np

script_dir = os.path.dirname(__file__)  
sys.path.append(os.path.join(script_dir, '../src'))

from dp import bootstrap_dp
from preprocess import PA_pca, load_and_prepare_dataset, perform_pca

datasets = ['carpediem']
# datasets = ['mimic']
# feature_sets = [
#     'sofs', 'sofs+vtls', 'sofs+vtls+dmos1', #'sofs+vtls+dmos', 'sofs+vtls+dmos1+flgs', 
#     'sofs+vtls+dmos1+lbs1', 'sofs+vtls+dmos1+lbs2', #'sofs+vtls+dmos1+lbs', 
#     # 'sofs+vtls+dmos1+lbs1+vents'
# ]
feature_sets = [
    'sofs+vtls+flgs', 
    'sofs+vtls+lbs1', 'sofs+vtls+flgs+lbs1', #'sofs+vtls+dmos1+lbs', 
    # 'sofs+vtls+dmos1+lbs1+vents'
]
# feature_sets = [
#     # 'sofs+vtls'
#     'sofs', 'sofs+vtls', 'sofs+vtls+dmos', 
#     'sofs+vtls+dmos+lbs1', 'sofs+vtls+dmos+lbs2'#, 'sofs+vtls+dmos+lbs'
# ]
imputation_strategies = [
    # 'Complete_cases', 
    # 'CarryLastForward', 'CarryLastForward_limit1', 
    'CarryLastForward_limit2',
    # 'MICE', 'FillZero', 'KNN'
]
# scalers = ['Raw', 'KBD', 'MM', 'SS', 'RS']
scalers = ['KBD']

holdout=.1

# Directory to save results
results_dir = Path('../data/calculated/dp/240521_holdout')
results_dir.mkdir(parents=True, exist_ok=True)

# Loop over configurations
for dataset in datasets:
    for feature_set in feature_sets:
        for imputation_strategy in imputation_strategies:
            for scaler_name in scalers:
                print(f"Processing: Dataset={dataset}, Feature_set={feature_set}, Imputation={imputation_strategy}, Scaler={scaler_name}, holdout={holdout}")
                
                # Load and prepare the dataset
                try:
                    data, _ = load_and_prepare_dataset(dataset, feature_set, imputation_strategy, scaler_name, holdout=holdout, random_state=9)
        
                    pcaX, pca, loadings, top_features = perform_pca(data=data.iloc[:,:-1], n_components=data.shape[1]-1, plot=False)
                    eig_rand = PA_pca(data=data.iloc[:,:-1], k=100, se=9)
                    mean_1 = np.mean(eig_rand, axis=0)
                    exceeds_mean = mean_1 >= pca.explained_variance_ratio_
                    first_exceeding_pc = (np.where(exceeds_mean)[0] + 1).min()
                    n_pc = first_exceeding_pc-1
                    print(f"Number of PCs used: {n_pc}")

                    dp_results = bootstrap_dp(dataset=dataset, featureset=feature_set, \
                                                  imputation_strategy=imputation_strategy, scaler_name=scaler_name, \
                                                    n_pc=n_pc, dp_mode='pdist', seed=9, n_boot=100, eff=False, output_dir=results_dir, holdout=holdout)
                    
                except Exception as e:
                    print(f"Error processing configuration: {e}")

print("Done.")
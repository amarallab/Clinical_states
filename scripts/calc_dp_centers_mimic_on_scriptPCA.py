#%%
from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import KBinsDiscretizer

script_dir = os.path.dirname(__file__)  
sys.path.append(os.path.join(script_dir, '../src'))

from dp import bootstrap_dp, dp
from preprocess import PA_pca, data_bootstrap, load_and_prepare_dataset, perform_pca

#%%
mimic, _ = load_and_prepare_dataset(dataset='mimic', feature_set='sofs+vtls+dmos', \
                                imputation_strategy='Complete_cases', scaler_name='Raw', holdout=.5)

carpediem, _ = load_and_prepare_dataset(dataset='carpediem', feature_set='sofs+vtls+dmos1', \
                                imputation_strategy='CarryLastForward_limit2', scaler_name='Raw', holdout=.1)
# %%
mapping_dict = {
    'respiration_24hours': 'P_F_ratio_points',
    'coagulation_24hours': 'platelet_points',
    'liver_24hours': 'bilirubin_points',
    'cardiovascular_24hours': 'htn_points',
    'gcs_motor': 'GCS_motor_response',
    'gcs_verbal': 'GCS_verbal_response',
    'gcs_eyes': 'GCS_eye_opening',
    'renal_24hours': 'renal_points',
    'heart_rate': 'Heart_rate',
    'sbp': 'Systolic_blood_pressure',
    'dbp': 'Diastolic_blood_pressure',
    'mbp': 'Mean_arterial_pressure',
    'resp_rate': 'Respiratory_rate',
    'temperature': 'Temperature',
    'spo2': 'Oxygen_saturation', 
    'age': 'Age', 
    'bmi': 'BMI'
}
# %%
mimic = mimic.rename(columns=mapping_dict)[carpediem.columns]
#%%
scaler = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy='quantile')
scaler.fit(carpediem.iloc[:, 8:-1])
carpediem.iloc[:, 8:-1] = scaler.transform(carpediem.iloc[:, 8:-1])
mimic.iloc[:, 8:-1] = scaler.transform(mimic.iloc[:, 8:-1])
#%%
pcaX_carpediem, pca_carpediem, loadings_carpediem, top_features_carpediem = perform_pca(data=carpediem.iloc[:,:-1], n_components=carpediem.shape[1]-1, plot=False)

Carpediem_fit_pca_transform_mimic = pca_carpediem.transform(mimic[carpediem.columns].iloc[:,:-1])
Carpediem_fit_pca_transform_mimic = pd.DataFrame(Carpediem_fit_pca_transform_mimic, columns=[f'PC{i}' for i in np.arange(1,carpediem.shape[1])], index=mimic.index)
# %%
n_pc=5
dp_mode='pdist'
X = Carpediem_fit_pca_transform_mimic[[f'PC{k+1}' for k in range(n_pc)]]
bandwidths = np.arange(0.1, 1,0.1)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=KFold(5), verbose=2, n_jobs=-1)
grid.fit(X.values)
bw = grid.best_params_['bandwidth']
print(f"Best bandwidth: {bw}")
#%%
bw=0.4
seed = 9
n_boot = 100
eff = False
output_dir = Path('../data/calculated/dp/mimic_on_scriptPCA')
output_dir.mkdir(parents=True, exist_ok=True)

centers = dp(X, bw, mode=dp_mode, eff=eff)
centers.reset_index().to_csv(os.path.join(output_dir, f'mimic_sofs+vtls+dmos_scriptKBD_{n_pc}PC.csv'))

data_boot = data_bootstrap(X, n_boot, se=seed)

for j in range(n_boot):
    # centers = dp(s1b.loc[(s1b.boot==j), [f'PC{k+1}' for k in range(n_pc)]], bw, mode=dp_mode, eff=eff)
    centers = dp(data_boot[j][[f'PC{k+1}' for k in range(n_pc)]], bw, mode=dp_mode, eff=eff)
    centers.reset_index().to_csv(os.path.join(output_dir, f'mimic_sofs+vtls+dmos_scriptKBD+_{n_pc}PC_boot{j}.csv'))

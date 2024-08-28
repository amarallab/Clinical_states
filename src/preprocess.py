# preprocess.py
#%%
import argparse
from datetime import datetime
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, RobustScaler, KBinsDiscretizer
from settings import curated_dir, carpediem_dir, mimic_dir, calc_dir, plots_dir

def apply_imputation(strategy, data):
    """
    Applies the specified imputation strategy to the data.
    """
    if strategy == 'Complete_cases':
        return data.dropna()
    elif strategy.startswith('CarryLastForward'):
        limit = None if strategy == 'CarryLastForward' else int(strategy[-1])
        return data.groupby(level=[0,1]).apply(
            lambda x: x.reset_index(level=[0,1], drop=True).ffill(limit=limit)
        ).dropna()
    elif strategy == 'MICE':
        imp = IterativeImputer(
            missing_values=np.nan, max_iter=30, verbose=2, imputation_order='roman', random_state=9
        )
        return pd.DataFrame(imp.fit_transform(data), index=data.index, columns=data.columns)
    elif strategy == 'FillZero':
        return data.fillna(0)
    elif strategy == 'KNN':
        knn = KNNImputer()
        return pd.DataFrame(knn.fit_transform(data), index=data.index, columns=data.columns)  

def load_and_prepare_dataset(dataset, feature_set, imputation_strategy='Complete_cases', scaler_name='Raw', holdout=False, random_state=9):
    """
    Loads and prepares a specific dataset based on provided parameters.
    
    Args:
    - dataset (str): Name of the dataset to load ('carpediem' or 'mimic').
    - feature_set (str): Specifies the set of features to include in the returned DataFrame.
    
    Returns:
    - pd.DataFrame: Preprocessed data according to the specified feature set.
    """
    if dataset == 'carpediem':
        return load_carpediem_data(feature_set, imputation_strategy, scaler_name, holdout=holdout, random_state=random_state)
    elif dataset == 'mimic':
        return load_mimic_data(feature_set, imputation_strategy, scaler_name, holdout=holdout, random_state=random_state)
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")

def load_carpediem_data(feature_set, imputation_strategy, scaler_name='Raw', holdout=False, random_state=9):
    """
    Specific function to load and preprocess the 'carpediem' dataset.
    """
    carpediem, carpediem2 = load_carpediem()
    carpediem = carpediem_icu(carpediem, carpediem2)

    carpediem = carpediem_edps(carpediem)

    feature_categories = carpediem_feature_cats(carpediem)

    # Select features based on the specified feature set
    if feature_set == 'org':
        features = [
            'P_F_ratio_points',
            'platelet_points',
            'bilirubin_points',
            'htn_points',
            'GCS_eye_opening',
            'GCS_motor_response',
            'GCS_verbal_response',
            'renal_points',
            'Temperature', 
            'Heart_rate', 
            'Systolic_blood_pressure',
            'Diastolic_blood_pressure',
            'Respiratory_rate', 
            'Oxygen_saturation',
            'Age'
        ]
    else:
        features = sum([feature_categories[cat] for cat in feature_set.split('+') if cat in feature_categories], [])
    carpediem.set_index(['Patient_id', 'ICU_stay', 'ICU_day'], inplace=True)
    data = carpediem.sort_index()[features]

    data = carpediem_onehot(data)

    data = apply_imputation(imputation_strategy, data)

    data = carpediem_scaling(scaler_name, feature_categories, data)
    data = data.merge(carpediem['edps'], how='left', left_index=True, right_index=True)

    if holdout:
        pats = data.index.get_level_values('Patient_id').unique()
        np.random.seed(random_state)
        idxs = np.random.choice(len(pats), int(np.round(len(pats)*holdout)), replace=False)
        holdout = data[data.index.get_level_values('Patient_id').isin(pats[idxs])]
        data = data[~data.index.get_level_values('Patient_id').isin(pats[idxs])]
        
        return data, holdout
    else:
        return data


def carpediem_scaling(scaler_name, feature_categories, data):
    if scaler_name != 'Raw':
        scaler_map = {
            'KBD': KBinsDiscretizer(n_bins=5, encode="ordinal", strategy='quantile'),
            'MM': MinMaxScaler(),
            'SS': StandardScaler(),
            'RS': RobustScaler()
        }
        if scaler_name == "KBD+MM":
            continuous_cols = [col for col in data.columns if col in (feature_categories['dmos1']+feature_categories['vtls']+feature_categories['lbs']+feature_categories['vents1'])] 
            if continuous_cols:
                data[continuous_cols] = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy='quantile').fit_transform(data[continuous_cols])
            data = pd.DataFrame(MinMaxScaler().fit_transform(data), index=data.index, columns=data.columns)
        elif scaler_name in scaler_map:
            scaler = scaler_map[scaler_name]
            if scaler_name == 'KBD':
                continuous_cols = [col for col in data.columns if col in (feature_categories['dmos1']+feature_categories['vtls']+feature_categories['lbs']+feature_categories['vents1'])] 
                if continuous_cols:
                    data[continuous_cols] = scaler.fit_transform(data[continuous_cols])
            else:
                data = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
    return data

def carpediem_onehot(data):
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    if len(categorical_columns)>0:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(sparse_output=False), categorical_columns)
            ],
            remainder='passthrough' 
        )

        data_transformed = preprocessor.fit_transform(data)
        columns_transformed = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)
        new_columns = list(columns_transformed) + [col for col in data.columns if col not in categorical_columns]

        data = pd.DataFrame(data_transformed, index=data.index, columns=new_columns)
    return data

def carpediem_feature_cats(carpediem):
    feature_categories = {
        'dmos': ['Age', 'Ethnicity', 'Gender', 'Race', 'Smoking_status', 'BMI'],
        'flgs': ['ECMO_flag', 'Intubation_flag', 'Hemodialysis_flag', 'CRRT_flag', 'Tracheostomy_flag'],
        'sofs': [
            'P_F_ratio_points',
            'platelet_points',
            'bilirubin_points',
            'htn_points',
            'GCS_eye_opening',
            'GCS_motor_response',
            'GCS_verbal_response',
            'renal_points'
        ],
        'vtls': [
            'Temperature', 
            'Heart_rate', 
            'Systolic_blood_pressure',
            'Diastolic_blood_pressure',
            'Mean_arterial_pressure',
            'Respiratory_rate', 
            'Oxygen_saturation'
        ],
        'lbs': [
            'Urine_output', 'ABG_pH', 'ABG_PaCO2',
            'ABG_PaO2', 'PaO2FIO2_ratio', 'WBC_count', 'Lymphocytes', 'Neutrophils',
            'Hemoglobin', 'Platelets', 'Bicarbonate', 'Creatinine', 'Albumin',
            'Bilirubin', 'CRP', 'D_dimer', 'Ferritin', 'LDH', 'Lactic_acid',
            'Procalcitonin'
        ],
        'vents': [
            'PEEP', 'FiO2', 'Plateau_Pressure', 'Lung_Compliance', 'PEEP_changes',
            'Respiratory_rate_changes', 'FiO2_changes'
        ]
    }

    all_features = sum(feature_categories.values(), [])
    df = carpediem[all_features]

    df = (df.isna().sum()/df.shape[0]).to_frame('na_frac').reset_index(names='features')
    condlist = [df.features.isin(i) for i in feature_categories.values()]
    choicelist = feature_categories.keys()
    df = df.assign(group=np.select(condlist, choicelist))
    df["group"] = pd.Categorical(df["group"], categories=feature_categories.keys())
    df = df.sort_values(['group', 'na_frac'])
    df = df.assign(completeness = 1-df.na_frac)

    feature_categories['lbs1'] = df[(df.group=='lbs')&(df.completeness>0.9)].features.to_list()
    feature_categories['lbs2'] = df[(df.group=='lbs')&(df.completeness>0.5)].features.to_list()
    feature_categories['dmos1'] = ['Age', 'BMI']
    feature_categories['vents1'] = ['PEEP', 'FiO2', 'Plateau_Pressure', 'Lung_Compliance']
    return feature_categories

def carpediem_edps(carpediem):
    carpediem['edps'] = np.nan
    carpediem['edps'] = carpediem['edps'].astype('object')
    f1 = carpediem.days_from_last_ICU_day_to_discharge==0
    f2 = carpediem.Discharge_disposition=='Died'
    f3 = carpediem.ICU_day==carpediem.ICU_len
    f4 = carpediem.ICU_stay==carpediem.max_icu
    carpediem.loc[f1&f2&f3&f4, 'edps'] = 'dying'

    f1 = carpediem.ICU_len - carpediem.ICU_day + carpediem.days_from_last_ICU_day_to_discharge<=10
    f2 = carpediem.Discharge_disposition=='Home'
    f3 = (carpediem.ICU_len-carpediem.ICU_day)/carpediem.ICU_len <=.2
    f4 = carpediem.ICU_stay==carpediem.max_icu
    carpediem.loc[f1&f2&f3&f4, 'edps'] = 'home'
    return carpediem

def carpediem_icu(carpediem, carpediem2):
    carpediem = carpediem.merge(carpediem2[['Patient_id/ICU_stay/ICU_day', 'days_from_last_ICU_day_to_discharge']])
    carpediem = carpediem.merge(
        carpediem.groupby(['Patient_id', 'ICU_stay']).size().to_frame('ICU_len').reset_index()
    )
    carpediem = carpediem.assign(max_icu=carpediem.groupby('Patient_id')['ICU_stay'].transform('max'))
    return carpediem

def load_carpediem():
    carpediem = pd.read_csv(os.path.join(curated_dir, 'carpediem.csv'), index_col=0)
    carpediem2 = pd.read_csv(os.path.join(carpediem_dir, '02data-external_dc_240314_1915.csv'), index_col=0)
    carpediem2 = carpediem2.reset_index(names='Patient_id/ICU_stay/ICU_day')
    return carpediem,carpediem2

#%%
def load_mimic_data(feature_set, imputation_strategy, scaler_name='Raw', holdout=False, random_state=9):
    """
    Loads and preprocesses the 'mimic' dataset based on the specified feature set.
    """
    # Load the basic MIMIC data
    mimic = pd.read_csv(os.path.join(mimic_dir, 'data.csv'), index_col=0)
    mimic_edps = pd.read_csv(os.path.join(curated_dir, 'mimic_edps.csv'), index_col=0)
    mimic = mimic.merge(mimic_edps, how='left')
    mimic['edps'] = np.nan
    mimic['edps'] = mimic['edps'].astype('object')
    f1 = abs(mimic.Itv)<1
    f2 = mimic.discharge=='Died'
    f3 = mimic.icu_day>=mimic.los_icu
    mimic.loc[f1&f2&f3, 'edps'] = 'dying'
    f1 = mimic.los_icu - mimic.icu_day + mimic.Itv<=10
    f2 = mimic.discharge=='Recovered'
    f3 = (mimic.los_icu-mimic.icu_day)/mimic.los_icu <=.2
    mimic.loc[f1&f2&f3, 'edps'] = 'home'
 
    feature_categories = {
        'dmos': ['age', 'bmi'],
        'sofs': ['respiration_24hours', 'coagulation_24hours', 'liver_24hours', 'cardiovascular_24hours',
                 'gcs_motor', 'gcs_verbal', 'gcs_eyes', 'renal_24hours'],
        'vtls': ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature', 'spo2'],
        'lbs': ['uo_24hr', 'po2', 'pco2', 'ph', 'pao2fio2ratio', 'totalco2', 'hemoglobin', 'hematocrit',
                'lactate', 'wbc', 'basophils_abs', 'eosinophils_abs', 'lymphocytes_abs', 'monocytes_abs',
                'neutrophils_abs', 'mch', 'mchc', 'mcv', 'platelet', 'rbc', 'rdw', 'albumin', 'aniongap',
                'bicarbonate', 'bun', 'calcium', 'chloride', 'creatinine', 'glucose', 'sodium', 'potassium',
                'bilirubin_total', 'alt', 'alp', 'ast', 'ld_ldh', 'crp', 'd_dimer', 'inr', 'pt', 'ptt', 'ferritin']
    }

    all_features = sum(feature_categories.values(), [])
    df = mimic[all_features]

    df = (df.isna().sum()/df.shape[0]).to_frame('na_frac').reset_index(names='features')
    condlist = [df.features.isin(i) for i in feature_categories.values()]
    choicelist = feature_categories.keys()
    df = df.assign(group=np.select(condlist, choicelist))
    df["group"] = pd.Categorical(df["group"], categories=feature_categories.keys())
    df = df.sort_values(['group', 'na_frac'])
    df = df.assign(completeness = 1-df.na_frac)

    feature_categories['lbs1'] = df[(df.group=='lbs')&(df.completeness>0.9)].features.to_list()
    feature_categories['lbs2'] = df[(df.group=='lbs')&(df.completeness>0.5)].features.to_list()

    # Select features based on the specified feature set
    features = sum([feature_categories[cat] for cat in feature_set.split('+') if cat in feature_categories], [])
    mimic.set_index(['subject_id', 'stay_id', 'icu_day'], inplace=True)
    data = mimic.sort_index()[features]

    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    if len(categorical_columns)>0:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(sparse_output=False), categorical_columns)
            ],
            remainder='passthrough' 
        )

        data_transformed = preprocessor.fit_transform(data)
        columns_transformed = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)
        new_columns = list(columns_transformed) + [col for col in data.columns if col not in categorical_columns]

        data = pd.DataFrame(data_transformed, index=data.index, columns=new_columns)

    data = apply_imputation(imputation_strategy, data)

    if scaler_name != 'Raw':
        scaler_map = {
            'KBD': KBinsDiscretizer(n_bins=5, encode="ordinal", strategy='quantile'),
            'MM': MinMaxScaler(),
            'SS': StandardScaler(),
            'RS': RobustScaler()
        }
        if scaler_name == "KBD+MM":
            continuous_cols = [col for col in data.columns if col in (feature_categories['dmos']+feature_categories['vtls']+feature_categories['lbs'])] 
            if continuous_cols:
                data[continuous_cols] = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy='quantile').fit_transform(data[continuous_cols])
            data = pd.DataFrame(MinMaxScaler().fit_transform(data), index=data.index, columns=data.columns)
        elif scaler_name in scaler_map:
            scaler = scaler_map[scaler_name]
            if scaler_name == 'KBD':
                continuous_cols = [col for col in data.columns if col in (feature_categories['dmos']+feature_categories['vtls']+feature_categories['lbs'])]
                if continuous_cols:
                    data[continuous_cols] = scaler.fit_transform(data[continuous_cols])
            else:
                data = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
    data = data.merge(mimic['edps'], how='left', left_index=True, right_index=True)

    if holdout:
        pats = data.index.get_level_values('subject_id').unique()
        np.random.seed(random_state)
        idxs = np.random.choice(len(pats), int(np.round(len(pats)*holdout)), replace=False)
        holdout = data[data.index.get_level_values('subject_id').isin(pats[idxs])]
        data = data[~data.index.get_level_values('subject_id').isin(pats[idxs])]
        
        return data, holdout
    else:
        return data
   

def perform_pca(
        data, n_components=2, top_n=5, n_rnd=100, se=9, 
        plot=False, save_plot=None
):
    pca = PCA(n_components=n_components)
    pcaX = pca.fit_transform(data)

    loadings = pca.components_
    num_pc = pca.n_components_
    pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
    pcaX = pd.DataFrame(pcaX, columns=pc_list, index=data.index)
    loadings = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
    loadings['variable'] = data.columns.values
    loadings = loadings.set_index('variable')
    
    top_features = {}
    for pc in pc_list:
        sorted_loadings = loadings[pc].abs().sort_values(ascending=False)
        top_features[pc] = sorted_loadings.head(top_n).index.tolist()

    if plot:
        eig_rand = PA_pca(pd.DataFrame(data), n_rnd, se)

        x_ = list(range(1, pca.n_components_ + 1))
        mean_1 = np.mean(eig_rand, axis=0)
        std_1 = np.std(eig_rand, axis=0)
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(x_, pca.explained_variance_ratio_ * 100, linewidth=2, label='Data')
        ax.plot(x_, mean_1 * 100, linewidth=2, label='Random')
        ax.fill_between(x_, (mean_1-std_1)*100, (mean_1+std_1)*100, facecolor = "red")
        ax.fill_between(x_, np.percentile(eig_rand, 5, axis=0), np.percentile(eig_rand, 95, axis=0), facecolor = "red")

        ax.legend(frameon=False)
        ax.set_ylabel("Explained variance %")
        ax.set_xlabel("Principal component")

        exceeds_mean = mean_1 >= pca.explained_variance_ratio_
        first_exceeding_pc = (np.where(exceeds_mean)[0] + 1).min()
        ax.set_title(f"1st PC noise>=data: {first_exceeding_pc}")
        if save_plot:
            plt.savefig(os.path.join(plots_dir, f'pca/pa_{save_plot}.png'), dpi=150, transparent=False)

        top_features_combined = list(set().union(*(top_features[pc] for pc in pc_list[:max(4, (first_exceeding_pc-1))])))
        cluster_data = abs(loadings.loc[top_features_combined, pc_list[:max(4, (first_exceeding_pc-1))]])
        sns.clustermap(cluster_data, annot=False, cmap='Reds', cbar_kws={'label': 'Loading'}, col_cluster=False)
        if save_plot:
            plt.savefig(os.path.join(plots_dir, f'pca/loadings_{save_plot}.png'), dpi=150, transparent=False)

        replacements = {
            'PC1': 'PC 1 [a.u.]', 'PC2': 'PC 2 [a.u.]', 'PC3': 'PC 3 [a.u.]', 'PC4': 'PC 4 [a.u.]'
        }
        g = sns.PairGrid(pcaX[[f'PC{i+1}' for i in range(4)]], diag_sharey=False)
        g.map_upper(hexbin)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.histplot)

        for i in range(4):
            for j in range(4):
                g.axes[i][j].set_xticklabels([])
                g.axes[i][j].set_yticklabels([])
                g.axes[i][j].set_xticks([])
                g.axes[i][j].set_yticks([])
                xlabel = g.axes[i][j].get_xlabel()
                ylabel = g.axes[i][j].get_ylabel()
                if xlabel in replacements.keys():
                    g.axes[i][j].set_xlabel(replacements[xlabel], fontsize=20)
                if ylabel in replacements.keys():
                    g.axes[i][j].set_ylabel(replacements[ylabel], fontsize=20)
        if save_plot:     
            plt.savefig(os.path.join(plots_dir, f'pca/distplot_{save_plot}.png'), dpi=150, transparent=False)
    return pcaX, pca, loadings, top_features

def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)

def data_randomize(data, m_replace=False):

    data_rand_ = np.zeros(data.shape)
    for i in range(data.shape[1]):
        p_ = 1.0*data.iloc[:,i].values
        if m_replace == False:
            np.random.shuffle(p_)
        else:
            p_ = np.random.choice(p_,size=len(p_),replace=True)
        data_rand_[:,i] = p_
    return data_rand_

def PA_pca(data, k, se):
    np.random.seed(se)
    eig_rand_ = np.zeros((k, data.shape[1]))
    for i in range(k):
        pca_rand_ = PCA().fit(data_randomize(data))
        eig_rand_[i,:] = pca_rand_.explained_variance_ratio_
    return eig_rand_

def data_bootstrap(data, n_boot, se):
    """
    Bootstrap data on the patient level.
    """
    np.random.seed(se)
    data_boot = []    
    for i in range(n_boot):
        idx = np.random.choice(data.index.get_level_values(0).unique(), size=data.index.get_level_values(0).nunique(), replace=True)
        data_boot.append(data.loc[idx])
    return data_boot

def boot_sil(
        se=9, n_boot=50, 
        dataset='carpediem', feature_set='sofs', 
        imputation_strategy='Complete_cases', 
        scaler_name='Raw', holdout=False
):
    if (holdout!=False):
        data, _ = load_and_prepare_dataset(dataset, feature_set, imputation_strategy, scaler_name, holdout=holdout)
    else:
        data = load_and_prepare_dataset(dataset, feature_set, imputation_strategy, scaler_name, holdout=holdout)
    n_pairs = data.shape[0]

    data_boot = data_bootstrap(data, n_boot, se)
    sils = []
    for boot in range(n_boot):
        ICU_stay
        pcaX, _, _, _ = perform_pca(data_boot_.iloc[:,:-1])
        data_boot_ = data_boot_.merge(pcaX, left_index=True, right_index=True)
        data_boot_.dropna(subset='edps', inplace=True)
        labels = data_boot_['edps'].map({'home': 0, 'dying': 1}).values
        sil_score = silhouette_score(data_boot_['PC1'].values.reshape(-1, 1), labels)
        sils.append({
            'n_pairs': n_pairs,
            'imputer': imputation_strategy,
            'scaler': scaler_name,  
            'boot': boot,
            'sil': sil_score, 
            'feature_set': feature_set
        })
    # pcaX, _, _, _ = perform_pca(data.iloc[:,:-1])
    # data = data.merge(pcaX, left_index=True, right_index=True)
    
    # np.random.seed(se)
    # sils = []
    # for boot in range(n_boot):
    #     sampled_indices = data[data['edps'].isin(['home', 'dying'])].index
    #     sampled_indices = np.random.choice(sampled_indices, size=n_smp, replace=True)

    #     sampled_data = data.loc[sampled_indices]   
    #     labels = sampled_data['edps'].map({'home': 0, 'dying': 1}).values
       
    #     sil_score = silhouette_score(sampled_data['PC1'].values.reshape(-1, 1), labels)
        
    #     sils.append({
    #         'n_pairs': n_pairs,
    #         'imputer': imputation_strategy,
    #         'scaler': scaler_name,  
    #         'boot': boot,
    #         'sil': sil_score, 
    #         'feature_set': feature_set
    #     })

    return pd.DataFrame(sils)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and analyze datasets.")
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to load and process (e.g., "carpediem", "mimic").')
    parser.add_argument('--feature_set', type=str, default='sofs',
                        help='Feature set to include in the analysis.')
    parser.add_argument('--imputation_strategy', type=str, default='Complete_cases',
                        help='Imputation strategy to apply to the dataset.')
    parser.add_argument('--scaler_name', type=str, default='Raw',
                        help='Scaler to use on the dataset.')
    parser.add_argument('--n_boot', type=int, default=50,
                        help='Number of bootstrapping iterations.')
    parser.add_argument('--n_smp', type=int, default=100,
                        help='Number of samples per bootstrap.')
    parser.add_argument('--seed', type=int, default=9,
                        help='Random seed for reproducibility.')
    parser.add_argument('--n_components', type=int, default=2,
                        help='Number of principal components to compute.')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Number of top features to extract per component.')
    parser.add_argument('--holdout', type=float, default=0.1)

    args = parser.parse_args()

    def main(args):
        silhouette_results = boot_sil(
            se=args.seed, n_smp=args.n_smp, n_boot=args.n_boot,
            dataset=args.dataset, feature_set=args.feature_set,
            imputation_strategy=args.imputation_strategy, scaler_name=args.scaler_name, 
            holdout=args.holdout
        )

        today = datetime.now().strftime("%Y%m%d")
        silhouette_results.to_csv(os.path.join(calc_dir, f'sils_{today}.csv'))
        print(f"Silhouette results saved to {os.path.join(calc_dir, f'sils_{today}.csv')}")
    
    main(args)
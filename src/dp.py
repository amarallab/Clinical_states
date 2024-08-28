# dp.py
import argparse
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KDTree, KernelDensity, NearestNeighbors

from preprocess import data_bootstrap, load_and_prepare_dataset, perform_pca
from settings import calc_dir

def avg_shortest_dist(X, N_samples):

    N = len(X)
    kdt = KDTree(X, metric='euclidean')

    list_d = []
    for i in range(N_samples):
        i1 = np.random.choice(N,replace=False,size=1)
        x1 = X[i1,:]
        d = kdt.query(x1, k=2, return_distance=True)[0][0][1]
        list_d += [d]
    return np.mean(list_d)

def dp_nn(X, bw=0.1, eff=10000):
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X.values)
    rho = np.exp(kde.score_samples(X.values))
    sort_rho_idx = np.argsort(-np.array(rho))
    
    delta = np.full(X.shape[0], np.inf)
    nneigh = np.full(X.shape[0], -1, dtype=int)

    # Prepare NearestNeighbors search (note: fitting is delayed to use partial data)
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    
    if eff:
        th = min(X.shape[0], eff)
    else:
        th = X.shape[0]
    for i in range(1, th):  # Skip the highest density point
        # Fit NN on points with higher density than the current point
        higher_density_points = X.iloc[sort_rho_idx[:i]]
        nn.fit(higher_density_points)
        
        # Find the nearest neighbor with higher density for the current point
        current_point = X.iloc[[sort_rho_idx[i]]]
        distance, index = nn.kneighbors(current_point)
        
        # Update delta and nneigh for the current point
        delta[sort_rho_idx[i]] = distance[0][0]
        # Map local index back to the global index
        nneigh[sort_rho_idx[i]] = sort_rho_idx[:i][index[0][0]]

    # The highest density point gets the maximum delta among others
    delta[sort_rho_idx[0]] = np.max(delta[sort_rho_idx[1:th]])

    # Construct the output DataFrame
    df = X.copy()
    df['rho'] = rho
    df['delta'] = delta
    df['nneigh'] = nneigh
    df['ordrho'] = sort_rho_idx
    df['rd'] = df['rho'] * df['delta']
    df['rd_rank'] = df['rd'].rank(ascending=False)
    
    return df

def dp_pdist(X, bw=0.1):   
    distance_matrix = pdist(X, metric='euclidean')
    distance_matrix = squareform(distance_matrix)
    
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X.values)
    rho = np.exp(kde.score_samples(X.values))
    sort_rho_idx = np.argsort(-np.array(rho))
    
    delta, nneigh = [float(distance_matrix.max())] * X.shape[0], [0] * X.shape[0] 
    delta[sort_rho_idx[0]] = -1.
    for i in range(X.shape[0]):
        for j in range(0, i):
            old_i, old_j = sort_rho_idx[i], sort_rho_idx[j]
            if distance_matrix[old_i, old_j] < delta[old_i]:
                delta[old_i] = distance_matrix[old_i, old_j]
                nneigh[old_i] = old_j
    delta[sort_rho_idx[0]] = max(delta)
    
    df = X.copy(deep=True)
    df['rho'] = rho
    df['delta'] = delta
    df['nneigh'] = nneigh
    df['ordrho'] = sort_rho_idx
    df['rd'] = df['rho'] * df['delta']
    df['rd_rank'] = df.rd.rank(ascending=False)

    return df

def dp(X, bw=0.1, mode='pdist', eff=10000):
    if mode == 'pdist':
        return dp_pdist(X, bw)
    elif mode == 'nn':
        return dp_nn(X, bw, eff)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def bootstrap_dp(dataset='carpediem', featureset='sofs', scaler_name='KBD', imputation_strategy='Complete_cases', n_pc=4, dp_mode='pdist', seed=9, n_boot=100, eff=False, output_dir=os.path.join(calc_dir, 'dp'), holdout=False):
    if holdout!=False:
        data, _ = load_and_prepare_dataset(dataset=dataset, feature_set=featureset, imputation_strategy=imputation_strategy, scaler_name=scaler_name, holdout=holdout, random_state=seed)
    else:
        data = load_and_prepare_dataset(dataset=dataset, feature_set=featureset, imputation_strategy=imputation_strategy, scaler_name=scaler_name, random_state=seed)
    pcaX, pca, _, _ = perform_pca(data=data.iloc[:,:-1], n_components=data.shape[1]-1, plot=False)

    X = pcaX[[f'PC{k+1}' for k in range(n_pc)]]
    bandwidths = np.arange(0.1,2,0.1)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=KFold(5), verbose=2, n_jobs=-1)
    grid.fit(X.values)
    bw = grid.best_params_['bandwidth']
    print(f"Best bandwidth: {bw}")
    centers = dp(X, bw, mode=dp_mode, eff=eff)
    centers.reset_index().to_csv(os.path.join(output_dir, f'{dataset}_{featureset}_{scaler_name}_{imputation_strategy}_{n_pc}PC_{dp_mode}.csv'))

    data_boot = data_bootstrap(X, n_boot, se=seed)
    # np.random.seed(seed)

    # s1b = pd.DataFrame()
    # for j in range(n_boot):
    #     sib = X.iloc[np.random.randint(X.shape[0], size=X.shape[0]), :]
    #     sib = sib.assign(boot=j)
    #     s1b = pd.concat((s1b,sib)).reset_index(drop=True)

    for j in range(n_boot):
        # centers = dp(s1b.loc[(s1b.boot==j), [f'PC{k+1}' for k in range(n_pc)]], bw, mode=dp_mode, eff=eff)
        centers = dp(data_boot[j][[f'PC{k+1}' for k in range(n_pc)]], bw, mode=dp_mode, eff=eff)
        centers.reset_index().to_csv(os.path.join(output_dir, f'{dataset}_{featureset}_{scaler_name}_{imputation_strategy}_{n_pc}PC_{dp_mode}_boot{j}.csv'))

def main():
    parser = argparse.ArgumentParser(description="Perform PCA and decision point analysis on clinical data.")
    parser.add_argument('--dataset', type=str, default='mimic', help='Dataset name')
    parser.add_argument('--feature_set', type=str, default='sofs+vtls+dmos', help='Feature set to use')
    parser.add_argument('--scaler_name', type=str, default='KBD', help='Scaler name')
    parser.add_argument('--imputation_strategy', type=str, default='Complete_cases', help='Imputation strategy')
    parser.add_argument('--n_pc', type=int, default=4, help='Number of principal components to analyze')
    parser.add_argument('--dp_mode', type=str, default='pdist', help='density peak mode (pdist or nn)')
    parser.add_argument('--seed', type=int, default=9, help='Random seed for reproducibility.')
    parser.add_argument('--n_boot', type=int, default=100, help='Number of bootstrap samples')
    parser.add_argument('--eff',type=int|bool, default=False, help='Efficient mode for density peak analysis')
    parser.add_argument('--output_dir', type=str, default=os.path.join(calc_dir, 'dp'), help='Output directory')
    
    args = parser.parse_args()

    bootstrap_dp(dataset=args.dataset, featureset=args.feature_set, scaler_name=args.scaler_name, imputation_strategy=args.imputation_strategy, n_pc=args.n_pc, dp_mode=args.dp_mode, seed=args.seed, n_boot=args.n_boot, eff=args.eff, output_dir=args.output_dir)

if __name__ == '__main__':
    main()
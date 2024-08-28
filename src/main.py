# main.py
#%%
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform
from preprocess import load_and_prepare_dataset, perform_pca

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

def pa(data, k, se):
    """Perform parallel analysis on PCA components."""
    np.random.seed(se)
    eig_rand_ = np.zeros((k, data.shape[1]))
    for i in range(k):
        pca_rand_ = PCA().fit(data_randomize(data))
        eig_rand_[i, :] = pca_rand_.explained_variance_ratio_
    return eig_rand_

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

def dp(X, bw=0.1):   
    distance_matrix = pdist(X, metric='euclidean')
    distance_matrix = squareform(distance_matrix)
    
    # kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X.values)
    # rho = np.exp(kde.score_samples(X.values))
    # sort_rho_idx = np.argsort(-np.array(rho))
    
    # delta, nneigh = [float(distance_matrix.max())] * X.shape[0], [0] * X.shape[0] 
    # delta[sort_rho_idx[0]] = -1.
    # for i in range(X.shape[0]):
    #     for j in range(0, i):
    #         old_i, old_j = sort_rho_idx[i], sort_rho_idx[j]
    #         if distance_matrix[old_i, old_j] < delta[old_i]:
    #             delta[old_i] = distance_matrix[old_i, old_j]
    #             nneigh[old_i] = old_j
    #     if i//100 == 0:
    #         print(i)
    # delta[sort_rho_idx[0]] = max(delta)
    
    # df = X.copy(deep=True)
    # df['rho'] = rho
    # df['delta'] = delta
    # df['nneigh'] = nneigh
    # df['ordrho'] = sort_rho_idx
    # df['rd'] = df['rho'] * df['delta']
    # df['rd_rank'] = df.rd.rank(ascending=False)

    return None

def main():
    parser = argparse.ArgumentParser(description='Data preprocessing, imputation, and PCA script.')
    parser.add_argument('--dataset', type=str, required=True, \
                        help='Dataset to process (e.g., "carpediem", "mimic").')
    parser.add_argument('--feature_set', type=str, required=True, \
                        help='Set of features to include (e.g., "sofs", "sofs+vtls").')
    parser.add_argument('--imputation_strategy', type=str, default='Complete_cases', \
                        help='Imputation strategy to apply (e.g., "Complete_cases", "CarryLastForward_limit1", "MICE").')
    parser.add_argument('--scaler_name', type=str, default='Raw',
                        help='Scaler to use on the dataset.')
    args = parser.parse_args()

    # Load and preprocess dataset
    data = load_and_prepare_dataset(args.dataset, args.feature_set, args.imputation_strategy, args.scaler_name)
    
    pcaX, pca, _, _ = perform_pca(data=data.iloc[:,:-1], n_components=data.shape[1]-1, plot=False)

    Xlab = pd.DataFrame(pcaX, columns=["PC"+str(i) for i in list(range(1, pca.n_components_+1))], index=data.index)

    n_pc=4
    X = Xlab[[f'PC{k+1}' for k in range(n_pc)]]
    # bandwidths = 10 ** np.linspace(-2, 0, 10)
    # grid = GridSearchCV(KernelDensity(kernel='gaussian'),
    #                     {'bandwidth': bandwidths},
    #                     cv=KFold(5), verbose=2, n_jobs=-1)
    # grid.fit(X.values)
    # print(grid.best_params_)
    # bandwidths = np.arange(0.1,1,0.1)
    # grid = GridSearchCV(KernelDensity(kernel='gaussian'),
    #                     {'bandwidth': bandwidths},
    #                     cv=KFold(5), verbose=2, n_jobs=-1)
    # grid.fit(X.values)
    # print(grid.best_params_)

    bw = 0.4
    print(X.shape)
    centers_4 = dp(X, bw)
    # n_pc = 4
    # n_boot = 100
    # np.random.seed(9)

    # s1b = pd.DataFrame()
    # si = Xlab[[f'PC{k+1}' for k in range(n_pc)]]
    # for j in range(n_boot):
    #     sib = si.iloc[np.random.randint(si.shape[0], size=si.shape[0]), :]
    #     sib = sib.assign(boot=j)
    #     s1b = pd.concat((s1b,sib)).reset_index(drop=True)
    #     print(j)
    # centers = {}
    # for j in range(n_boot):
    #     centers[f'pc4boot{j}'] = dp(s1b.loc[(s1b.boot==j), [f'PC{k+1}' for k in range(n_pc)]], bw)
    #     print(j)
#%%
if __name__ == "__main__":
    main()



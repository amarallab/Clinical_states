
from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity

script_dir = os.path.dirname(__file__)  
sys.path.append(os.path.join(script_dir, '../src'))

from ClusterEnsembles import *
from dp import *

# Directory
syn_dir = Path('../data/calculated/dp/synthetic')
syn_dir.mkdir(parents=True, exist_ok=True)

n_smp = 20
n_feats = [4,10]
n_classes = [2,4,8]
noise_ratios = [0.1, 0.5]

X = pd.read_csv(os.path.join(syn_dir, 'simulated_data.csv'))
sls = pd.read_csv(os.path.join(syn_dir, 'dps_silhouette_scores.csv'))

dfs = []
for file in os.listdir(syn_dir):
    if file.endswith('.csv') and file.startswith('smp'):
        patterns = file.rstrip('.csv').split('_') 
        smp = int(patterns[0].strip('smp'))
        n_feat = int(patterns[1].strip('feat'))
        n_class = int(patterns[2].strip('class'))
        noise_ratio = float(patterns[3].strip('noise'))
        if len(patterns) == 5:
            boot = int(patterns[4].strip('boot'))
        else:
            boot = 'all'
        df = pd.read_csv(os.path.join(syn_dir, file))
        df = df.assign(smp=smp)
        df = df.assign(n_feat=n_feat)
        df = df.assign(n_class=n_class)
        df = df.assign(noise_ratio=noise_ratio)
        df = df.assign(boot=boot)
        dfs.append(df)
dps = pd.concat(dfs, ignore_index=True)

dfs = []
for i in range(n_smp):
    for n_feat in n_feats:
        for n_class in n_classes:
            for noise_ratio in noise_ratios:
                xi = X[(X.smp==i) & (X.n_feat==n_feat) & (X.n_class==n_class) & (X.noise_ratio==noise_ratio)][[f'x{j}' for j in np.arange(1,n_feat+1)]]
                
                Kms = [
                    KMeans(nn, random_state=9, n_init=10).fit_predict(xi.values)
                    for nn in range(2,20)
                ]
                Aggs = [
                    AgglomerativeClustering(n_clusters=nn).fit_predict(xi.values)
                    for nn in range(2,20)
                ]
                kms_mcla = cluster_ensembles(np.array(Kms), solver='mcla')
                ags_mcla = cluster_ensembles(np.array(Aggs), solver='mcla')

                sil = sls[(sls.smp==i) & (sls.n_feat==n_feat) & (sls.n_class==n_class) & (sls.noise_ratio==noise_ratio)]
                sil = sil.groupby('n_cmp_assume').sl.agg(lambda x: x.argmax()+2).to_frame('mx_cmp').reset_index()
                if sum(sil.n_cmp_assume==sil.mx_cmp)>0:
                    # dp_cmp = np.random.choice(sil[sil.n_cmp_assume==sil.mx_cmp].n_cmp_assume.unique())
                    dp_cmp = sil[sil.mx_cmp.isin(sil[sil.n_cmp_assume==sil.mx_cmp].n_cmp_assume.unique())].mx_cmp.value_counts().sort_values(ascending=False).index[0]
                else:
                    dp_cmp = np.round(sil.mx_cmp.mean())
                dp_cmp = int(dp_cmp)
                cc = dps[(dps.smp==i) & (dps.n_feat==n_feat) & (dps.n_class==n_class) & (dps.noise_ratio==noise_ratio)].sort_values('rd', ascending=False).iloc[:(dp_cmp*101),:][[f'x{ff}' for ff in np.arange(1,n_feat+1)]].drop_duplicates()
                km = KMeans(n_clusters=dp_cmp, random_state=0, n_init=5).fit(cc.values)
                gm = GaussianMixture(n_components=dp_cmp, means_init=km.cluster_centers_, fix_means=True).fit(xi.values)
                edp_pred=gm.predict(xi.values)
                edp_scr=gm.score_samples(xi.values)
                edp_pred[edp_scr<np.quantile(edp_scr, noise_ratio/(1+noise_ratio))] = -1

                xi = xi.assign(edp_pred=edp_pred)
                xi = xi.assign(edp_scr=edp_scr)
                xi = xi.assign(km_mcla=kms_mcla)
                xi = xi.assign(agg_mcla=ags_mcla)
                xi = xi.assign(smp=i)
                xi = xi.assign(n_feat=n_feat)
                xi = xi.assign(n_class=n_class)
                xi = xi.assign(noise_ratio=noise_ratio)
                xi = xi.assign(y_true=X[(X.smp==i) & (X.n_feat==n_feat) & (X.n_class==n_class) & (X.noise_ratio==noise_ratio)].y_true)
                dfs.append(xi)
                print(f"Done: smp: {i}, n_feat: {n_feat}, n_class: {n_class}, noise_ratio: {noise_ratio}")
                out = pd.concat(dfs, ignore_index=True)
                out.to_csv(os.path.join(syn_dir, 'synthetic_bench.csv'), index=False)

out = pd.concat(dfs, ignore_index=True)
out.to_csv(os.path.join(syn_dir, 'synthetic_bench.csv'), index=False)



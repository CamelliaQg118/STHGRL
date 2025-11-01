import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn import metrics
import scipy as sp
import numpy as np
import torch
import copy
import os
import STHGRL

import utils

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


ARI_list = []
random_seed = 42
STHGRL.fix_seed(random_seed)
os.environ['R_HOME'] = 'D:/R/R-4.3.3/R-4.3.3'
os.environ['R_USER'] = 'D:/Anaconda3/Anaconda3202303/envs/STHGRL/Lib/site-pac kages/rpy2'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

dataset = 'DLPFC'
slice = '151676'
platform = '10X'
file_fold = os.path.join('../Data', platform, dataset, slice)
adata, adata_X = utils.load_data(dataset, file_fold)
print("adata", adata)
df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
adata = utils.label_process_DLPFC(adata, df_meta)

savepath = '../Result/test/DLPFC/' + str(slice) + '/'

if not os.path.exists(savepath):
    os.mkdir(savepath)
n_clusters = 5 if slice in ['151669', '151670', '151671', '151672'] else 7



print("adata_adj", adj)
print("hypergraph", hypergraph)
print("hypergraph_drop", hypergraph_drop)
sthgrl_net = STHGRL.sthgrl(adata.obsm['X_pca'], adata, adj, hypergraph, hypergraph_drop, n_clusters, dataset, device=device)

tool = None
if tool == 'mclust':
    emb = sthgrl_net.train()
    adata.obsm['STHGRL'] = emb
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    STHGRL.mclust_R(adata, n_clusters, use_rep='STHGRL', key_added='STHGRL', random_seed=random_seed)
elif tool == 'leiden':
    emb = sthgrl_net.train()
    adata.obsm['STHGRL'] = emb
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    STHGRL.leiden(adata, n_clusters, use_rep='STHGRL', key_added='STHGRL', random_seed=random_seed)
elif tool == 'louvain':
    emb = sthgrl_net.train()
    adata.obsm['STHGRL'] = emb
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    STHGRL.louvain(adata, n_clusters, use_rep='STHGRL', key_added='STHGRL', random_seed=random_seed)
else:
    emb, idx = sthgrl_net.train()
    print("emb", emb)
    adata.obsm['STHGRL'] = emb
    adata.obs['STHGRL'] = idx
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

print("adata", adata)
new_type = utils.refine_label(adata, radius=15, key='STHGRL')
adata.obs['STHGRL'] = new_type
ARI = metrics.adjusted_rand_score(adata.obs['ground_truth'], adata.obs['STHGRL'])
NMI = metrics.normalized_mutual_info_score(adata.obs['ground_truth'], adata.obs['STHGRL'])
adata.uns["ARI"] = ARI
adata.uns["NMI"] = NMI
print('===== Project: {}_{} ARI score: {:.4f}'.format(str(dataset), str(slice), ARI))
print('===== Project: {}_{} NMI score: {:.4f}'.format(str(dataset), str(slice), NMI))
print(str(slice))
print(n_clusters)
ARI_list.append(ARI)


plt.rcParams["figure.figsize"] = (3, 3)
title = "Manual annotation (" + dataset + "#" + slice + ")"
sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title, show=False)
plt.savefig(savepath + 'Manual Annotation.jpg', bbox_inches='tight', dpi=300)
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
sc.pl.spatial(adata, color='ground_truth', ax=axes[0], show=False)
sc.pl.spatial(adata, color=['STHGRL'], ax=axes[1], show=False)
axes[0].set_title("Manual annotation (" + dataset + "#" + slice + ")")
axes[1].set_title('STHGRL_Clustering: (ARI=%.4f)' % ARI)


plt.subplots_adjust(wspace=0.5)  
plt.subplots_adjust(hspace=0.5)  
plt.savefig(savepath + 'STHGRL.jpg', dpi=300)  


sc.pp.neighbors(adata, use_rep='STHGRL', metric='cosine')
sc.tl.umap(adata)
sc.pl.umap(adata, color='STHGRL', title='STHGRL', show=False)
plt.savefig(savepath + 'umap.jpg', bbox_inches='tight', dpi=300)

for ax in axes:
    ax.set_aspect(1)
plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(hspace=0.5)


title = 'STHGRL:{}_{} ARI={:.4f} NMI={:.4f}'.format(str(dataset), str(slice), adata.uns['ARI'], adata.uns['NMI'])
sc.pl.spatial(adata, img_key="hires", color=['STHGRL'], title=title, show=False)
plt.savefig(savepath + 'STHGRL_NMI_ARI_gat.tif', bbox_inches='tight', dpi=300)

plt.rcParams["figure.figsize"] = (3, 3)
sc.tl.paga(adata, groups='STHGRL')
sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=title, legend_fontoutline=2, show=False)
plt.savefig(savepath + 'STSGCL_PAGA_domain.tif', bbox_inches='tight', dpi=300)

sc.tl.paga(adata, groups='ground_truth')
sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=title, legend_fontoutline=2, show=False)
plt.savefig(savepath + 'STSGCL_PAGA_ground_truth.png', bbox_inches='tight', dpi=300)



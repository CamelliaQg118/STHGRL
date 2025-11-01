import os
import ot
import torch
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn import metrics
from torch_sparse import SparseTensor
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from matplotlib.image import imread


def adata_hvg(adata):
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] ==True]
    sc.pp.scale(adata)
    return adata


def adata_hvg_MSC(adata):
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] ==True]
    sc.pp.scale(adata)
    return adata


def adata_hvg_process(adata):
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.scale(adata)
    return adata

def adata_hvg_slide(adata):
    # sc.pp.filter_genes(adata, min_cells=50)
    # sc.pp.normalize_total(adata, target_sum=1e6)
    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    # adata = adata[:, adata.var['highly_variable'] ==True]
    # sc.pp.scale(adata)
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)
    return adata


def fix_seed(seed):
    import random
    import torch
    from torch.backends import cudnn

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def load_data(dataset, file_fold):
    if dataset == "DLPFC":
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
        # print("adata", adata)
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X

    elif dataset == "Human_Breast_Cancer":
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
    elif dataset == "Adult_Mouse_Brain_Section_1":
        adata = sc.read_visium(file_fold, count_file='V1_Adult_Mouse_Brain_Coronal_Section_1_filtered_feature_bc_matrix.h5', load_images=True)
        # print('adata', adata)
        adata.var_names_make_unique()
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
    elif dataset == "Mouse_Brain_Anterior_Section1":
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        # print('adata', adata)
        adata.var_names_make_unique()
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
    elif dataset == "ME":
        adata = sc.read_h5ad(file_fold + 'E9.5_E1S1.MOSTA.h5ad')
        print('adata', adata)
        adata.var_names_make_unique()
        # # print("adata", adata)
        # adata.obs['x'] = adata.obs["array_row"]
        # adata.obs['y'] = adata.obs["array_col"]
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X

    elif dataset == 'MOB':
        # savepath = '../Result/MOB_Stereo/'
        # if not os.path.exists(savepath):
        #     os.mkdir(savepath)
        counts_file = os.path.join(file_fold, 'RNA_counts.tsv')
        counts = pd.read_csv(counts_file, sep='\t', index_col=0).T
        counts.index = [f'Spot_{i}' for i in counts.index]
        adata = sc.AnnData(counts)
        adata.X = csr_matrix(adata.X, dtype=np.float32)
        adata.var_names_make_unique()

        pos_file = os.path.join(file_fold, 'position.tsv')
        coor_df = pd.read_csv(pos_file, sep='\t')
        coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
        coor_df = coor_df.loc[:, ['x', 'y']]
        # print('adata.obs_names', adata.obs_names)
        coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
        adata.obs['x'] = coor_df['x'].tolist()
        adata.obs['y'] = coor_df['y'].tolist()
        adata.obsm["spatial"] = coor_df.to_numpy()
        print(adata)

        # hires_image = os.path.join(file_fold, 'crop1.png')
        # adata.uns["spatial"] = {}
        # adata.uns["spatial"][dataset] = {}
        # adata.uns["spatial"][dataset]['images'] = {}
        # adata.uns["spatial"][dataset]['images']['hires'] = imread(hires_image)

        # label_file = pd.read_csv(os.path.join(file_fold, 'Cell_GetExp_gene.txt'), sep='\t', header=None)
        # used_barcode = label_file[0]

        barcode_file = pd.read_csv(os.path.join(file_fold, 'used_barcodes.txt'), sep='\t', header=None)
        used_barcode = barcode_file[0]
        adata = adata[used_barcode]
        adata.var_names_make_unique()

        # adata.obs['total_exp'] = adata.X.sum(axis=1)
        # fig, ax = plt.subplots()
        # sc.pl.spatial(adata, color='total_exp', spot_size=40, show=False, ax=ax)
        # ax.invert_yaxis()


        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg_process(adata)

        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
        adata_X = torch.FloatTensor(np.array(adata_X))

    elif dataset == 'MOB_V2':
        # savepath = '../Result/MOB_Slide/_0.005/'
        # if not os.path.exists(savepath):
        #     os.mkdir(savepath)
        # counts_file = os.path.join(input_dir, '')
        # coor_file = os.path.join(input_dir, '')
        counts_file = os.path.join(file_fold, 'Puck_200127_15.digital_expression.txt')
        counts = pd.read_csv(counts_file, sep='\t', index_col=0)
        adata = sc.AnnData(counts.T)
        adata.X = csr_matrix(adata.X, dtype=np.float32)
        adata.var_names_make_unique()
        print(adata)

        coor_file = os.path.join(file_fold, 'Puck_200127_15_bead_locations.csv')
        coor_df = pd.read_csv(coor_file, index_col=0)
        # coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
        coor_df = coor_df.set_index('barcode')
        coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
        adata.obs['x'] = coor_df['xcoord'].tolist()
        adata.obs['y'] = coor_df['ycoord'].tolist()
        adata.obsm["spatial"] = coor_df.to_numpy()
        sc.pp.calculate_qc_metrics(adata, inplace=True)

        # print("adata", adata)
        # plt.rcParams["figure.figsize"] = (6, 5)
        # # Original tissue area, some scattered spots
        # sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts", s=6, show=False, save='_MOB01_slide.png')
        # plt.title('')
        # plt.axis('off')


        barcode_file = pd.read_csv(os.path.join(file_fold, 'used_barcodes.txt'), sep='\t', header=None)
        used_barcode = barcode_file[0]
        adata = adata[used_barcode]
        adata.var_names_make_unique()
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg_process(adata)
   
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
        adata_X = torch.FloatTensor(np.array(adata_X))

    elif dataset == 'hip':
        savepath = '../Result/hip/_0.005/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        # counts_file = os.path.join(input_dir, '')
        # coor_file = os.path.join(input_dir, '')
        counts_file = os.path.join(file_fold, 'Puck_200115_08.digital_expression.txt')
        counts = pd.read_csv(counts_file, sep='\t', index_col=0)
        adata = sc.AnnData(counts.T)
        adata.X = csr_matrix(adata.X, dtype=np.float32)
        adata.var_names_make_unique()
        print(adata)

        coor_file = os.path.join(file_fold, 'Puck_200115_08_bead_locations.csv')
        coor_df = pd.read_csv(coor_file, index_col=0)
        # coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
        coor_df = coor_df.set_index('barcode')
        coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
        adata.obs['x'] = coor_df['xcoord'].tolist()
        adata.obs['y'] = coor_df['ycoord'].tolist()
        adata.obsm["spatial"] = coor_df.to_numpy()
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        print(adata)
        plt.rcParams["figure.figsize"] = (6, 5)
        sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts", s=6, show=False)
        plt.title('prime')
        plt.axis('off')
        plt.savefig(savepath + 'STMCCL_hip_1.jpg', dpi=600)

        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg_slide(adata)
   
        print("adata", adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
        adata_X = torch.FloatTensor(np.array(adata_X))


    elif dataset == 'ISH':
        adata = sc.read(file_fold + '/STARmap_20180505_BY3_1k.h5ad')
        # print(adata)
        adata.obs['x'] = adata.obs["X"]
        adata.obs['y'] = adata.obs["Y"]
        adata.layers['count'] = adata.X
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X

    elif dataset == 'mouse_somatosensory_cortex':
        adata = sc.read(file_fold + '/osmFISH_cortex.h5ad')
        print(adata)
        adata.var_names_make_unique()
        adata = adata[adata.obs["Region"] != "Excluded"]

        adata.layers['count'] = adata.X
        adata = adata_hvg_MSC(adata)
        sc.pp.scale(adata)
        # print("adata1", adata)
        # adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata_X = adata.X
        adata.obsm['X_pca'] = adata.X
    else:
        platform = '10X'
        file_fold = os.path.join('../Data', platform, dataset)
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5')
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        df_meta = pd.read_csv(os.path.join('../Data', dataset,  'metadata.tsv'), sep='\t', header=None, index_col=0)
        adata.obs['layer_guess'] = df_meta['layer_guess']
        df_meta.columns = ['over', 'ground_truth']
        adata.obs['ground_truth'] = df_meta.iloc[:, 1]

        adata.var_names_make_unique()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X

    return adata, adata_X


def label_process_DLPFC(adata, df_meta):
    labels = df_meta["layer_guess_reordered"].copy()
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    ground.replace('WM', '0', inplace=True)
    ground.replace('Layer1', '1', inplace=True)
    ground.replace('Layer2', '2', inplace=True)
    ground.replace('Layer3', '3', inplace=True)
    ground.replace('Layer4', '4', inplace=True)
    ground.replace('Layer5', '5', inplace=True)
    ground.replace('Layer6', '6', inplace=True)
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground
    return adata



def graph_build(adata, adata_X, dataset):
    if dataset == 'DLPFC':
        n1 = 12
        adj, edge_index = load_adj(adata, n1)

    elif dataset == 'MBO':
        n1 = 10
        adj, edge_index = load_adj(adata, n1)  

    elif dataset =='MOB_V2':
        n1 = 7
        adj, edge_index = load_adj(adata, n1)  

    elif dataset == 'Adult_Mouse_Brain_Section_1':
        n1 = 5
        n2 = [5, 10]
        adj, edge_index = load_adj(adata, n1) 

    elif dataset == 'Mouse_Brain_Anterior_Section1':
        n1 = 10
        adj, edge_index = load_adj(adata, n1)  

    elif dataset == 'ISH':
        n1 = 7
        adj, edge_index = load_adj(adata, n1)  

    elif dataset == 'mouse_somatosensory_cortex':
        n1 = 10
        adj, edge_index = load_adj(adata, n1)  

    else:
        n1 = 10
        adj, edge_index = load_adj(adata, n1) 

    return adata, adj, edge_index



def dropout_edge(edge_index, p=0.5, force_undirected=False):
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


def edge_index_to_numpy_adj(edge_index, num_nodes, is_undirected=True):

    edge_index = edge_index.cpu().numpy() 
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for src, dst in edge_index.T:
        adj[src, dst] = 1
        if is_undirected:
            adj[dst, src] = 1

    return adj


def load_adj(adata, n1):
    adj = generate_adj(adata, include_self=False, n=n1)
    # print("adj", adj)

    adj = sp.coo_matrix(adj)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_norm, edge_index = preprocess_adj(adj)
    return adj_norm, edge_index,


def load_adj_hyper(adata, n1, n2):
    adj = generate_adj(adata, include_self=False, n=n1)
    # print("adj", adj)


    adj_hp, H = load_hpgraph(adj, n2)
    adj = sp.coo_matrix(adj)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_norm, edge_index = preprocess_adj(adj)

    adj_drop, H_drop = edge_process(adj, edge_index, n2)
    return adj_norm, edge_index, H, H_drop


def edge_process(adj, edge_index, n2):

    drop_edge_rate = 0.1
    edge_drop, mask_index = dropout_edge(edge_index, p=drop_edge_rate, force_undirected=False)
    num_nodes = adj.shape[0]
    adj_drop = edge_index_to_numpy_adj(edge_drop, num_nodes)
    adj_drop = adj_drop.astype(np.int64)
    fea_1, H_drop = load_hpgraph(adj_drop, n2)

    return adj_drop, H_drop


def load_hpgraph(adj, n):
    adj_hp, H = load_feature_construct_H(adj, m_prob=1, K_neigs=n, is_probH=False)
    return adj_hp, H


def adj_to_edge_index(adj):
    dense_adj = adj.toarray()
    edge_index = torch.nonzero(torch.tensor(dense_adj), as_tuple=False).t()
    return edge_index


def generate_adj(adata, include_self=False, n=6):
    dist = metrics.pairwise_distances(adata.obsm['spatial'])
    adj = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n+1]
        adj[i, n_neighbors] = 1
    if not include_self:
        x, y = np.diag_indices_from(adj)
        adj[x, y] = 0
    adj = adj + adj.T
    adj = adj > 0
    adj = adj.astype(np.int64)
    return adj


def preprocess_adj(adj):
    adj = adj + sp.eye(adj.shape[0])
    # edge_index = adj_to_edge_index(adj)
    rowsum = np.array(adj.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    edge_index = adj_to_edge_index(adj_normalized)

    return sparse_mx_to_torch_sparse_tensor(adj_normalized), edge_index


def generate_adj2(adata, include_self=True):
    dist = metrics.pairwise_distances(adata.obsm['spatial'])
    dist = dist / np.max(dist)
    adj = dist.copy()
    if not include_self:
        np.fill_diagonal(adj, 0)
    # print('adj', adj)
    return adj


# def Eu_dis(x):
#     x = np.mat(x)
#     # x = np.array(x)
#     aa = np.sum(np.multiply(x, x), 1)
#     ab = x * x.T
#     dist_mat = aa + aa.T - 2 * ab
#     dist_mat[dist_mat < 0] = 0
#     dist_mat = np.sqrt(dist_mat)
#     dist_mat = np.maximum(dist_mat, dist_mat.T)
#     return dist_mat

# def Eu_dis(x):
#     x = np.array(x)
#     aa = np.sum(np.square(x), axis=1, keepdims=True)  # (N, 1)
#     dist = aa + aa.T - 2 * (x @ x.T)
#     dist[dist < 0] = 0
#     return np.sqrt(dist)


def Eu_dis(x):
    if isinstance(x, torch.Tensor):
      
        aa = torch.sum(x ** 2, dim=1, keepdim=True)  # (N, 1)
        dist = aa + aa.T - 2 * torch.matmul(x, x.T)
        dist = torch.clamp(dist, min=0.0)  
        return torch.sqrt(dist)
    else:

        x = np.array(x)
        aa = np.sum(np.square(x), axis=1, keepdims=True)
        dist = aa + aa.T - 2 * (x @ x.T)
        dist[dist < 0] = 0
        return np.sqrt(dist)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def refine_label(adata, radius=50, key='label'):  
    n_neigh = radius  
    new_type = []  
    old_type = adata.obs[key].values  

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')  

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    return new_type


def scale(z: torch.Tensor):
    zmax = z.max(dim=1, keepdim=True)[0]
    zmin = z.min(dim=1, keepdim=True)[0]
    z_std = (z - zmin) / ((zmax - zmin) + 1e-20)
    z_scaled = z_std
    return z_scaled

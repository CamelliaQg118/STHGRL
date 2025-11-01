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
    #种子为2023
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
        # plt.savefig(savepath + 'STMCCL_stereo_MOB1.jpg', dpi=600)

        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg_process(adata)
        # print("adata是否降维", adata)
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
        # plt.savefig(savepath + 'STNMAE_MOBV2.jpg', dpi=300)

        barcode_file = pd.read_csv(os.path.join(file_fold, 'used_barcodes.txt'), sep='\t', header=None)
        used_barcode = barcode_file[0]
        adata = adata[used_barcode]
        adata.var_names_make_unique()
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg_process(adata)
        # print("adata是否降维", adata)
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
        # print("adata是否降维", adata)
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
        adata.obs['ground_truth'] = df_meta.iloc[:, 1]#同理获取获取数据

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


def label_process_HBC(adata, df_meta):
    labels = df_meta["ground_truth"].copy()
    # print("labels", labels)
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    ground = ground.replace('DCIS/LCIS_1', '0')
    ground = ground.replace('DCIS/LCIS_2', '1')
    ground = ground.replace('DCIS/LCIS_4', '2')
    ground = ground.replace('DCIS/LCIS_5', '3')
    ground = ground.replace('Healthy_1', '4')
    ground = ground.replace('Healthy_2', '5')
    ground = ground.replace('IDC_1', '6')
    ground = ground.replace('IDC_2', '7')
    ground = ground.replace('IDC_3', '8')
    ground = ground.replace('IDC_4', '9')
    ground = ground.replace('IDC_5', '10')
    ground = ground.replace('IDC_6', '11')
    ground = ground.replace('IDC_7', '12')
    ground = ground.replace('IDC_8', '13')
    ground = ground.replace('Tumor_edge_1', '14')
    ground = ground.replace('Tumor_edge_2', '15')
    ground = ground.replace('Tumor_edge_3', '16')
    ground = ground.replace('Tumor_edge_4', '17')
    ground = ground.replace('Tumor_edge_5', '18')
    ground = ground.replace('Tumor_edge_6', '19')
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground.values.astype(int)
    # print("ground", adata.obs['ground'])
    return adata


def label_process_Mouse_brain_anterior(adata, df_meta):
    labels = df_meta["ground_truth"].copy()
    # print("labels", labels)
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    ground = ground.replace('AOB::Gl', '0')
    ground = ground.replace('AOB::Gr', '1')
    ground = ground.replace('AOB::Ml', '2')
    ground = ground.replace('AOE', '3')
    ground = ground.replace('AON::L1_1', '4')
    ground = ground.replace('AON::L1_2', '5')
    ground = ground.replace('AON::L2', '6')
    ground = ground.replace('AcbC', '7')
    ground = ground.replace('AcbSh', '8')
    ground = ground.replace('CC', '9')
    ground = ground.replace('CPu', '10')
    ground = ground.replace('Cl', '11')
    ground = ground.replace('En', '12')
    ground = ground.replace('FRP::L1', '13')
    ground = ground.replace('FRP::L2/3', '14')
    ground = ground.replace('Fim', '15')
    ground = ground.replace('Ft', '16')
    ground = ground.replace('HY::LPO', '17')
    ground = ground.replace('Io', '18')
    ground = ground.replace('LV', '19')
    ground = ground.replace('MO::L1', '20')
    ground = ground.replace('MO::L2/3', '21')
    ground = ground.replace('MO::L5', '22')
    ground = ground.replace('MO::L6', '23')
    ground = ground.replace('MOB::Gl_1', '24')
    ground = ground.replace('MOB::Gl_2', '25')
    ground = ground.replace('MOB::Gr', '26')
    ground = ground.replace('MOB::MI', '27')
    ground = ground.replace('MOB::Opl', '28')
    ground = ground.replace('MOB::lpl', '29')
    ground = ground.replace('Not_annotated', '30')
    ground = ground.replace('ORB::L1', '31')
    ground = ground.replace('ORB::L2/3', '32')
    ground = ground.replace('ORB::L5', '33')
    ground = ground.replace('ORB::L6', '34')
    ground = ground.replace('OT::Ml', '35')
    ground = ground.replace('OT::Pl', '36')
    ground = ground.replace('OT::PoL', '37')
    ground = ground.replace('Or', '38')
    ground = ground.replace('PIR', '39')
    ground = ground.replace('Pal::GPi', '40')
    ground = ground.replace('Pal::MA', '41')
    ground = ground.replace('Pal::NDB', '42')
    ground = ground.replace('Pal::Sl', '43')
    ground = ground.replace('Py', '44')
    ground = ground.replace('SLu', '45')
    ground = ground.replace('SS::L1', '46')
    ground = ground.replace('SS::L2/3', '47')
    ground = ground.replace('SS::L5', '48')
    ground = ground.replace('SS::L6', '49')
    ground = ground.replace('St', '50')
    ground = ground.replace('TH::RT', '51')
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground.values.astype(int)
    # print("ground", adata.obs['ground'])
    return adata


def graph_build(adata, adata_X, dataset):
    if dataset == 'DLPFC':
        n1 = 12
        adj, edge_index = load_adj(adata, n1)#加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式
        ###基于特征关系构建超边

    elif dataset == 'MBO':
        n1 = 10
        adj, edge_index = load_adj(adata, n1)  # 加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式

    elif dataset =='MOB_V2':
        n1 = 7
        adj, edge_index = load_adj(adata, n1)  # 加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式

    elif dataset == 'Adult_Mouse_Brain_Section_1':
        n1 = 5
        n2 = [5, 10]
        adj, edge_index = load_adj(adata, n1)  # 加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式

    elif dataset == 'Mouse_Brain_Anterior_Section1':
        n1 = 10
        adj, edge_index = load_adj(adata, n1)  # 加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式

    elif dataset == 'ISH':
        n1 = 7
        adj, edge_index = load_adj(adata, n1)  # 加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式

    elif dataset == 'mouse_somatosensory_cortex':
        n1 = 10
        adj, edge_index = load_adj(adata, n1)  # 加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式

    else:
        n1 = 10
        adj, edge_index = load_adj(adata, n1)  # 加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式

    return adata, adj, edge_index


def graph_hypergraph_build(adata, adata_X, dataset):
    if dataset == 'DLPFC':
        n1 = 10
        n2 = [5, 10]
        adj, edge_index, H, H_drop = load_adj_hyper(adata, n1, n2)#加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式
        ###基于特征关系构建超边
        hypergraph = generate_G_from_H(H, variable_weight=False)
        print("hypergraph", hypergraph)
        hypergraph_drop = generate_G_from_H(H_drop, variable_weight=False)

    elif dataset == 'Human_Breast_Cancer':
        n1 = 10
        n2 = [5, 10]
        adj, edge_index, H, H_drop = load_adj_hyper(adata, n1, n2)  # 加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式
        ###基于特征关系构建超边
        hypergraph = generate_G_from_H(H, variable_weight=False)
        print("hypergraph", hypergraph)
        hypergraph_drop = generate_G_from_H(H_drop, variable_weight=False)

    elif dataset == 'Mouse_Brain_Anterior_Section1':
        n1 = 10
        n2 = [5, 10]
        adj, edge_index, H, H_drop = load_adj_hyper(adata, n1, n2)  # 加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式
        ###基于特征关系构建超边
        hypergraph = generate_G_from_H(H, variable_weight=False)
        print("hypergraph", hypergraph)
        hypergraph_drop = generate_G_from_H(H_drop, variable_weight=False)

    elif dataset == 'mouse_somatosensory_cortex':
        n1 = 10
        n2 = [5, 10]
        adj, edge_index, H, H_drop = load_adj_hyper(adata, n1, n2)  # 加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式
        ###基于特征关系构建超边
        hypergraph = generate_G_from_H(H, variable_weight=False)
        print("hypergraph", hypergraph)
        hypergraph_drop = generate_G_from_H(H_drop, variable_weight=False)

    elif dataset == 'ME':
        n1 = 10
        n2 = [5, 10]
        adj, edge_index, H, H_drop = load_adj_hyper(adata, n1, n2)  # 加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式
        ###基于特征关系构建超边
        hypergraph = generate_G_from_H(H, variable_weight=False)
        print("hypergraph", hypergraph)
        hypergraph_drop = generate_G_from_H(H_drop, variable_weight=False)

    elif dataset == 'MBO':
        n1 = 10
        n2 = [5, 10]
        adj, edge_index, H, H_drop = load_adj_hyper(adata, n1, n2)  # 加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式
        ###基于特征关系构建超边
        hypergraph = generate_G_from_H(H, variable_weight=False)
        print("hypergraph", hypergraph)
        hypergraph_drop = generate_G_from_H(H_drop, variable_weight=False)

    elif dataset == 'MOB_V2':
        n1 = 7
        n2 = [5, 10]
        adj, edge_index, H, H_drop = load_adj_hyper(adata, n1, n2)  # 加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式
        ###基于特征关系构建超边
        hypergraph = generate_G_from_H(H, variable_weight=False)
        print("hypergraph", hypergraph)
        hypergraph_drop = generate_G_from_H(H_drop, variable_weight=False)

    elif dataset == 'ISH':
        n1 = 7
        n2 = [5, 10]
        adj, edge_index, H, H_drop = load_adj_hyper(adata, n1, n2)  # 加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式
        ###基于特征关系构建超边
        hypergraph = generate_G_from_H(H, variable_weight=False)
        print("hypergraph", hypergraph)
        hypergraph_drop = generate_G_from_H(H_drop, variable_weight=False)

    else:
        n1 = 10
        n2 = [5, 10]
        adj, edge_index, H, H_drop = load_adj_hyper(adata, n1, n2)  # 加载邻接矩阵，返回的是加上自环并且归一化的邻接矩阵，且返回邻接矩阵的索引形式
        ###基于特征关系构建超边
        hypergraph = generate_G_from_H(H, variable_weight=False)
        print("hypergraph", hypergraph)
        hypergraph_drop = generate_G_from_H(H_drop, variable_weight=False)

    return adata, adj, edge_index, hypergraph, hypergraph_drop


def dropout_edge(edge_index, p=0.5, force_undirected=False):
    if p < 0. or p > 1.:#检查边的取值范围
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

    edge_index = edge_index.cpu().numpy()  # 转为 numpy
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for src, dst in edge_index.T:
        adj[src, dst] = 1
        if is_undirected:
            adj[dst, src] = 1

    return adj


def load_adj(adata, n1):
    adj = generate_adj(adata, include_self=False, n=n1)#生成邻接矩阵
    # print("adj", adj)
    # 邻接矩阵处理(基于特征，则无adj的事)
    adj = sp.coo_matrix(adj)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_norm, edge_index = preprocess_adj(adj)
    return adj_norm, edge_index,


def load_adj_hyper(adata, n1, n2):
    adj = generate_adj(adata, include_self=False, n=n1)#生成邻接矩阵
    # print("adj", adj)
    # 邻接矩阵处理(基于特征，则无adj的事)

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


def load_feature_construct_H(adj, m_prob=1, K_neigs=[10], is_probH=False,):
    adj = adj
    H = None
    tmp = construct_H_with_KNN(adj, K_neigs=K_neigs, split_diff_scale=False, is_probH=is_probH, m_prob=m_prob)
    H = hyperedge_concat(H, tmp)
    return adj, H


def construct_H_with_KNN(adj, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    if len(adj.shape) != 2:
        adj = adj.reshape(-1, adj.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(adj)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)

        else:
            H.append(H_tmp)
    return H


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    def _is_empty(h):
        if isinstance(h, list):
            return len(h) == 0
        elif isinstance(h, np.ndarray):
            return h.size == 0
        else:
            return False
    H = None
    for h in H_list:
        if h is not None and not _is_empty(h):
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    # 如果是 numpy，则转成 torch tensor
    is_numpy_input = False
    if isinstance(dis_mat, np.ndarray):
        is_numpy_input = True
        dis_mat = torch.from_numpy(dis_mat)

    dis_mat = dis_mat.clone()
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    device = dis_mat.device
    H = torch.zeros((n_obj, n_edge), device=device)

    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0.0
        dis_vec = dis_mat[center_idx]  # shape: (N,)

        nearest_idx = torch.argsort(dis_vec)  # 返回排序后的索引
        avg_dis = torch.mean(dis_vec)

        if center_idx not in nearest_idx[:k_neig]:
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                dist = dis_vec[node_idx]
                weight = torch.exp(-dist**2 / (m_prob * avg_dis)**2)
                H[node_idx, center_idx] = weight
            else:
                H[node_idx, center_idx] = 1.0

    return H.numpy() if is_numpy_input else H



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
#     dist[dist < 0] = 0  # 避免数值误差出现负值
#     return np.sqrt(dist)


def Eu_dis(x):
    if isinstance(x, torch.Tensor):
        # Tensor 类型
        aa = torch.sum(x ** 2, dim=1, keepdim=True)  # (N, 1)
        dist = aa + aa.T - 2 * torch.matmul(x, x.T)
        dist = torch.clamp(dist, min=0.0)  # 避免负值
        return torch.sqrt(dist)
    else:
        # 默认 NumPy 类型
        x = np.array(x)
        aa = np.sum(np.square(x), axis=1, keepdims=True)
        dist = aa + aa.T - 2 * (x @ x.T)
        dist[dist < 0] = 0
        return np.sqrt(dist)


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    # print(H)
    H = np.array(H)
    n_edge = H.shape[1]#获取超边数
    # the weight of the hyperedge
    W = np.ones(n_edge)#默认所有超边权重初始化为 1
    # the degree of the node
    DV = np.sum(H * W, axis=1)#计算每个节点的度数（每行求和）
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)#计算超边的度数（每列求和）

    invDE = np.mat(np.diag(np.power(DE, -1)))#构建超边逆对角矩阵D^-1
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))#构建对角矩阵D^-1/2
    W = np.mat(np.diag(W))#构建超边权重矩阵
    H = np.mat(H)#将 H 显式转为矩阵格式
    HT = H.T#超边关联矩阵的转置

    if variable_weight:#是否使用可变权重，如果使用可变权重
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2#则返回三个值
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G #不使用可变权重，则返回一个超图卷积核


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def refine_label(adata, radius=50, key='label'):  # 修正函数相当于强制修正，
    # 功能，使得每个spot半径小于50的范围内，其他spot 的大部分是哪一类就把这个spot 强制归为这一类。
    n_neigh = radius  # 定义半径
    new_type = []  # spot新的类型
    old_type = adata.obs[key].values  ##读入数据的原始类型

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')  # 用欧氏距离

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
    z_scaled = z_std#取方差为输出结果
    return z_scaled

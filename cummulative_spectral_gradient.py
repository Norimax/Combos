import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.neighbors import KNeighborsClassifier
import umap
from scipy.sparse.csgraph import connected_components

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.collections import LineCollection


class CummulativeSpectralGradient:
  	"""
    This class computes the complexity of a dataset, which is
    called cummulative-spectral-gradient, CSG.
    Original Paper: https://arxiv.org/abs/1905.07299
    """
    def __init__(self, class_nms, emb_method='tsne',
                 emb_dim=2, n_neighbors=3):
        """
        Args:
            class_nms (array-like): names of classes
            emb_method (str): method of embedding, 'pca', 'tsne' or 'umap'
            emb_dim (int): dimention of output of embedding, 2 or 3
            n_neighbors (int):
            	number of neighbors to calculate similarities in
                k-neighbors clustering
        """
        self.class_nms = class_nms
        self.n_classes = len(class_nms)
        self.emb_method = emb_method
        self.emb_dim = emb_dim
        self.n_neighbors = n_neighbors

    def compute_CSG(self, X, y):
        if self.emb_method == 'tsne':
            perp = int(len(y) ** 0.5)
            self.emb_df = self.embed_data(X, y, perplexity=perp)
        else:
            self.emb_df = self.embed_data(X, y)
        self.S = self.compute_S()
        self.W = self.compute_W(self.S)
        self.CSG = self.compute_CSG_from_W(self.W)

    def embed_data(self, X, y, **params):
        """
        Args:
            X: 1-dimentinal data (must be flattened)
            y: array of labels of X
            method: algorithm of the reduction. ('umap' or 'tsne')
            params: other arguments for embedding function
        """
        if self.emb_method == 'umap':
            reducer = umap.UMAP(n_components=self.emb_dim,
                                random_state=42, **params)
        elif self.emb_method == 'tsne':
            reducer = TSNE(n_components=self.emb_dim,
                           random_state=42, **params)
        elif self.emb_method == 'pca':
            reducer = PCA(n_components=self.emb_dim,
                          random_state=42, **params)
        val_cols = [f'val{i}' for i in range(1, self.emb_dim + 1)]
        emb_df = pd.DataFrame(reducer.fit_transform(X), columns=val_cols)
        emb_df['label'] = y
        return emb_df

    def compute_S(self):
        """
        S: similarity matrix
        """
        val_cols = [f'val{i}' for i in range(1, self.emb_dim + 1)]
        S = np.eye(self.n_classes)
        for i, j in combinations(list(range(self.n_classes)), 2):
            X_i = self.emb_df[self.emb_df.label == i][val_cols].values
            X_j = self.emb_df[self.emb_df.label == j][val_cols].values
            S[i, j] = self.compute_similarity(X_i, X_j, self.n_neighbors)
            S[j ,i] = self.compute_similarity(X_j, X_i, self.n_neighbors)
        return S

    def compute_similarity(self, X_i, X_j, n_neighbors=3):
        """
        Calculate similarity of class-i to class-j by computing multiplicity.
        """
        X = np.concatenate([X_i, X_j])
        y = np.concatenate([np.zeros(len(X_i)), np.ones(len(X_j))])
        knc = KNeighborsClassifier(n_neighbors)
        knc.fit(X, y)
        proba = knc.predict_proba(X)
        m_ij = proba[:len(X_i), 1].mean()
        return m_ij

    @staticmethod
    def compute_W(S):
        """
        W: weighted adjacency matrix
        """
        W = np.ones(S.shape)
        for i, j in combinations(list(range(S.shape[0])), 2):
            w_ij = 1 - np.abs(S[i] - S[j]).sum() / np.abs(S[i] + S[j]).sum()
            W[i, j] = W[j, i] = w_ij
        return W

    @staticmethod
    def compute_CSG_from_W(W):
        n_classes = W.shape[0]
        D = np.eye(n_classes) * W.sum(0)
        L = D - W
        eigvals = np.sort(np.linalg.eigvals(L))
        eigvals_norm = [(eigvals[i+1] - eigvals[i]) / (n_classes - i)
                        for i in range(n_classes - 1)]
        CSG = pd.Series(eigvals_norm).cummax().sum()
        return CSG

    def plot_emb_scatter(self, save_path=None):
        if self.emb_dim == 2:
            plt.figure(figsize=(10, 8))
            for i, cls in enumerate(self.class_nms):
                tmp = self.emb_df[self.emb_df.label == i]
                x, y = tmp.val1, tmp.val2
                plt.scatter(x, y, label=cls, s=5)
        elif self.emb_dim == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection='3d')
            for i, cls in enumerate(self.class_nms):
                tmp = self.emb_df[self.emb_df.label == i]
                x, y, z = tmp.val1, tmp.val2, tmp.val3
                ax.scatter3D(x, y, z, label=cls, s=5)
        plt.legend()
        if save_path != None:
            plt.savefig(save_path)
        plt.show()
        plt.close()

    def plot_mds(self, save_path=None):
        mds = MDS(n_components=2, max_iter=3000, eps=1e-9,
                  dissimilarity='precomputed', n_jobs=1,
                  random_state=42)
        pos = mds.fit(1 - self.W).embedding_

        plt.figure(figsize=(8, 8))
        ax = plt.axes([0., 0., 1., 1.])

        for (pos_x, pos_y), cls in zip(pos, self.class_nms):
            plt.text(pos_x - 0.03, pos_y - 0.03, cls, fontsize=25)

        segments = [[pos[i, :], pos[j, :]]
                    for i in range(len(pos)) for j in range(len(pos))]
        lc = LineCollection(segments,
                            zorder=0, cmap=plt.cm.Blues,
                            norm=plt.Normalize(0, 0.5))
        lc.set_linewidths(np.full(len(segments), 0.5))
        ax.add_collection(lc)
        plt.scatter(pos[:, 0], pos[:, 1], color='turquoise')
        plt.axis('off')
        if save_path != None:
            plt.savefig(save_path)
        plt.show()
        plt.close()

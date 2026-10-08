from functools import partial
from typing import List

import numpy as np
import torch
from freerec.data.datasets import RecDataSet
from freerec.data.tags import ITEM, ID
from scipy.sparse import coo_array, csr_array


class Mope:
    def __init__(
        self,
        dataset: RecDataSet,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        method: str = "metis",
        ngroups: int = 16,
        ufactor: int = 7999,
        resolution: float = 1.0,
    ):
        Item = dataset.fields[ITEM, ID]
        edge_index, edge_weight = edge_index.numpy(), edge_weight.numpy()
        row, col = edge_index[0], edge_index[1]
        self._adj = coo_array((edge_weight, (row, col)), shape=(Item.count, Item.count))

        if method == "full":
            self.solver = self.full
        elif method == "random":
            self.solver = partial(self.random, ngroups=ngroups)
        elif method == "spectral":
            self.solver = partial(self.spectral, ngroups=ngroups)
        elif method == "metis":
            self.solver = partial(self.metis, ngroups=ngroups, ufactor=ufactor)
        elif method == "louvain":
            self.solver = partial(self.louvain, resolution=resolution)
        elif method == "leiden":
            self.solver = partial(self.leiden, resolution=resolution)
        else:
            raise NotImplementedError

    def full(self, adj: coo_array, seed: int = 0):
        groups = np.zeros(adj.shape[0], dtype=np.int64)
        return groups

    def random(self, adj: coo_array, ngroups: int = 16, seed: int = 0):
        nums = adj.shape[0]
        group_size = nums // ngroups
        groups = [[ngroups - i] * group_size for i in range(ngroups)]
        groups.append([0] * (nums % ngroups))
        groups = np.concatenate(groups)
        rng = np.random.default_rng(seed)
        return rng.permutation(groups)

    def spectral(self, adj, ngroups: int = 16, seed: int = 0):
        from sklearn.cluster import SpectralClustering

        clustering = SpectralClustering(
            n_clusters=ngroups, affinity="precomputed", random_state=seed
        )

        adj_coo = adj.tocoo()
        row = adj_coo.row.astype(np.int32)
        col = adj_coo.col.astype(np.int32)
        data = adj_coo.data

        adj = csr_array(
            (data, (row, col)),
            shape=adj_coo.shape,
            dtype=np.float32,
        )

        clustering.fit(adj)
        return clustering.labels_

    def metis(self, adj, ngroups: int = 16, ufactor: int = 7999, seed: int = 0):
        import pymetis

        adj = adj.tocsr()
        xadj = adj.indptr
        adjncy = adj.indices
        eweights = adj.data

        n_cuts, groups = pymetis.part_graph(
            ngroups,
            xadj=xadj,
            adjncy=adjncy,
            eweights=eweights,
            options=pymetis.Options(seed=seed, ufactor=ufactor),
        )
        return np.array(groups)

    def louvain(self, adj, resolution: float = 1.0, seed: int = 0):
        import networkx as nx

        row = adj.row
        col = adj.col
        data = adj.data
        graph = nx.Graph()
        for i, j, weight in zip(row, col, data):
            graph.add_edge(i, j, weight=weight)
        communities = nx.community.louvain_communities(graph, resolution=resolution, seed=seed)

        groups = []
        indices = []
        for i, community in enumerate(communities):
            indices.append(list(community))
            groups.append([i] * len(community))

        indices = np.concatenate(indices)
        groups = np.concatenate(groups)
        return groups[np.argsort(indices)]

    def leiden(self, adj, resolution: float = 1.0, seed: int = 0):
        import igraph as ig
        import leidenalg as la

        row, col, data = adj.row, adj.col, adj.data
        mask = row < col
        edges = list(zip(row[mask], col[mask]))
        weights = data[mask].tolist()

        g = ig.Graph(n=adj.shape[0], edges=edges, directed=False)
        g.es["weight"] = weights

        partition = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=resolution,
            seed=seed,
        )

        return np.array(partition.membership)

    def __call__(self, seed: int = 0) -> np.ndarray:
        return self.solver(adj=self._adj, seed=seed)

import torch
import numpy as np
import scipy.sparse as sp


def deg_est(adj_mat: sp.dok_matrix):
    """ Compute a degree diagonal matrix from an adjacency matrix. """
    num_nodes = adj_mat.shape[0]
    diag_deg, _ = np.histogram(adj_mat.nonzero()[0], np.arange(num_nodes + 1))

    diag_mat = sp.coo_matrix((num_nodes, num_nodes))
    diag_mat.setdiag(diag_deg)

    # estimate sqrt and reciprocal for GCN message aggregation
    diag_mat = diag_mat.sqrt()
    np.reciprocal(diag_mat.data, where=diag_mat.data != 0, out=diag_mat.data)
    return diag_mat


def tensor_from_coo(coo_matrix: sp.coo_matrix):
    """ Build tensor from a scipy sparse matrix. """
    values = coo_matrix.data
    indices = np.vstack((coo_matrix.row, coo_matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo_matrix.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

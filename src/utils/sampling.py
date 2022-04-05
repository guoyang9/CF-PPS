import numpy as np


def sample(matrix, entity, start, end, neg_num):
    """ matrix and entity should correspond with each other.
    :param matrix: <user, item>, for example,
    :param entity: current user id, for example,
    :param start: where randint begins,
    :param end: where randint ends
    """
    negatives = []
    for _ in range(neg_num):
        j = np.random.randint(start, end)
        while (entity, j) in matrix or j in negatives:
            j = np.random.randint(start, end)
        negatives.append(j)
    return negatives

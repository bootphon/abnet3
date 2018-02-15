import torch
from torch import FloatTensor
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np


class SoftDTW(torch.autograd.Function):

    def __init__(self, distance='cos', gamma=0.1):

        self.gamma = gamma

        if distance == 'cos':
            self.distance = F.cosine_similarity
        elif callable(distance):
            self.distance = distance
        else:
            raise ValueError("This distance is not supported")


    @staticmethod
    def softmin(a, b, c, gamma):
        """
        Softmin function as described in the Soft DTW paper.
        There is a trick here to get better precision (remove max value from vector
        before doing the exponentiation).
        """
        max_value = max(a, b, c)
        sum = (np.exp(a - max_value) +
               np.exp(b - max_value) +
               np.exp(c - max_value))
        return - gamma * (np.log(sum) + max_value)



    def forward(self, matrix_a: FloatTensor, matrix_b: FloatTensor):
        """
        :param matrix_b: 
        :param matrix_a: 
        :param gamma: 
        :return: 
        """
        n = matrix_a.size()[0]
        m = matrix_b.size()[0]

        # distance matrix initialization
        distances = np.zeros((n, m))

        # compute distance matrix
        for i in range(n):
            for j in range(m):
                distances[i, j] = self.distance(matrix_a[i], matrix_b[j])

        # soft DTW
        R = np.full((n+1, m+1), np.inf)  # we index starting at 1 to facilitate the following loop
        R[0, 0] = 0
        # forward passs
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                R[i, j] = distances[i-1, j-1] + min(R[i-1, j], R[i-1, j-1], R[i, j-1])

        # after this, we have to save some variables for the backward pass (mostly D and R)
        self.save_for_backward([
            distances, R,
        ])
        return R



    def backward(self, grad_outputs):
        pass
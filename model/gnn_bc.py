import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter, Linear, Sequential, LogSoftmax


class GNNBC(Module):
    def __init__(self, n, nclass, nfeat, nlayer, lambda_1, lambda_2, dropout):
        super(GNNBC, self).__init__()
        self.n = n
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.nclass = nclass
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.dropout = dropout
        self.w1 = Parameter(torch.FloatTensor(nfeat, nfeat), requires_grad=True)
        self.w2 = Sequential(Linear(2 * nfeat, nclass), LogSoftmax(dim=1))
        self.params1 = [self.w1]
        self.params2 = list(self.w2.parameters())
        self.laplacian = None
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.uniform_(self.w1, 0, 1)

    def forward(self, feat, adj):
        if self.laplacian is None:
            n = adj.shape[0]
            indices = torch.Tensor([list(range(n)), list(range(n))])
            values = torch.FloatTensor([1.0] * n)
            eye = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n]).to(adj.device)
            self.laplacian = eye - adj
        lap = self.laplacian
        y: Tensor = torch.rand(self.n, self.nfeat).to(adj.device)
        z: Tensor = feat
        for i in range(self.nlayer):
            feat = F.dropout(feat, self.dropout, training=self.training)
            temp = torch.mm(self.w1, z.t())
            temp = torch.mm(z, temp)
            temp = torch.sigmoid(temp)
            temp = torch.mm(temp, y)
            temp = (self.lambda_2 / self.lambda_1) * temp
            y_n = torch.spmm(lap, y) - temp
            temp = torch.mm(y.t(), z)
            temp = torch.mm(y, temp)
            temp = torch.sigmoid(temp)
            z_n = feat - self.lambda_2 * temp
            y = y_n
            z = z_n
        y = F.normalize(y, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        p = torch.cat((y, z), dim=1)
        p = F.dropout(p, self.dropout, training=self.training)
        return self.w2(p)

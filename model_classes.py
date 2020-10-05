import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

import ipdb

USE_BREAKPOINTS = False

# DCPF SOLVER CLASS. FORWARD RETURNS THETA
class DCPFLayer(nn.Module):
    def __init__(self, mpc, B, gen, bus, br):
        super(DCPFLayer, self).__init__()
        self.B = B
        self.gen = gen
        self.bus = bus
        self.br  = br
        self.mpc = mpc

    def forward(self, Pnew):
        non_slack_indices = torch.LongTensor(np.where(self.bus[:, 1] != 3)[0])
        slack_index = np.where(self.bus[:, 1] == 3)[0][0]
        slack_gen_indices = np.where(self.bus[:, 1] != 1)[0]

        Pnet = -torch.Tensor(self.bus[:, 2])
        Pnet[slack_gen_indices] += Pnew

        # mapping = {}
        # mapping = {**mapping, **dict(zip(self.gen[:,0], Pnew))}
        
        # non_slack_indices = []
        # Pnet = torch.zeros((len(self.bus)))
        # for index in range(len(self.bus)):
        #     if self.bus[index, 1] != 3:
        #         non_slack_indices.append(index)
        #     else:
        #         slack_index = index
        #     if self.bus[index, 0] in mapping:
        #         Pnet[index] = mapping[self.bus[index, 0]] - self.bus[index, 2]
        #     else:
        #         Pnet[index] = - self.bus[index, 2]
        # if USE_BREAKPOINTS: 
        #     ipdb.set_trace()
        #     print("Pnet: {}".format(Pnet))
        
        # non_slack_indices = torch.LongTensor(non_slack_indices)

        B_reduced = self.B[:, non_slack_indices]
        B_reduced = B_reduced[non_slack_indices,:]

        theta = torch.zeros(len(self.bus))
        theta[non_slack_indices] = torch.inverse(B_reduced) @ (Pnet[non_slack_indices])
        if USE_BREAKPOINTS: 
            ipdb.set_trace()
            print("theta : {}".format(theta))

        Pnet[slack_index] = self.B[slack_index,:] @ theta

        Pslack = Pnet[slack_index] + self.bus[slack_index, 2]
        Pnew_postflow = torch.cat([Pslack.unsqueeze(0), Pnew[1:]])
        # ipdb.set_trace()
        # Pnew[slack_index] = Pnet[slack_index] + self.bus[slack_index, 2]
        return theta, non_slack_indices, Pnew_postflow

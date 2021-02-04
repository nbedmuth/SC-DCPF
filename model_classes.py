import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from pypower import idx_bus #, idx_gen

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

        self.non_slack_idxes = torch.LongTensor(np.where(self.bus[:, idx_bus.BUS_TYPE] != 3)[0])
        self.slack_idx = np.where(self.bus[:, idx_bus.BUS_TYPE] == 3)[0][0]
        self.slack_gen_idxes = np.where(self.bus[:, idx_bus.BUS_TYPE] != 1)[0]
        self.B_reduced = self.B[self.non_slack_idxes[:, None], self.non_slack_idxes]

    def forward(self, Pg):
        Pnet = -torch.Tensor(self.bus[:, idx_bus.PD])
        Pnet[self.slack_gen_idxes] += Pg

        theta = torch.zeros(len(self.bus))
        theta[self.non_slack_idxes] = torch.inverse(self.B_reduced) @ (Pnet[self.non_slack_idxes])
        # print("theta : {}".format(theta))

        Pnet[self.slack_idx] = self.B[self.slack_idx, :] @ theta 

        Pg_slack = Pnet[self.slack_idx] + self.bus[self.slack_idx, idx_bus.PD]
        Pg_postflow = torch.cat([Pg_slack.unsqueeze(0), Pg[1:]])

        return theta, Pg_postflow

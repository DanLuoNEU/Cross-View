# Copyright (c) 2021 Qualcomm Technologies, Inc.

# All Rights Reserved.

from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F
from modelZoo.sigmoid_module import Sigmoid

class Binarization(nn.Module):
    def __init__(
        self,
        in_channels: int,
    ):
        super(Binarization, self).__init__()

        print('******New Binarization******')
        self.gs = Sigmoid()
        self.in_channels = in_channels

        self.gating_layers = [
            nn.Linear(in_channels, in_channels),
        ]
        
        self.gate_network = nn.Sequential(*self.gating_layers)
        self.init_weights()

    def load_weights_standalone(self):

        model_path = '/home/balaji/Documents/code/RSL/binarization/models/exp13/106.pth'
        state_dict = torch.load(model_path, map_location=torch.device(1))#
        fc_weight = state_dict['gate_network.0.weight']
        fc_bias = state_dict['gate_network.0.bias']
        
        return fc_weight, fc_bias

    def init_weights(self, gate_bias_init: float = 0.0, pretrain=1) -> None:
        
        if pretrain:
            print('load pretrain binarization module')
            fc_weight, fc_bias = self.load_weights_standalone()
            fc = self.gate_network[0]
            fc.weight = torch.nn.Parameter(fc_weight)
            fc.bias = torch.nn.Parameter(fc_bias)
            
        else:
            for i in range(len(self.gating_layers)):

                fc = self.gate_network[i]
                torch.nn.init.xavier_uniform_(fc.weight)
                fc.bias.data.fill_(gate_bias_init)

    def forward(self, gate_inp: torch.Tensor) -> torch.Tensor:
        """Gumbel gates, Eq (8)"""

        gate_inp = torch.transpose(gate_inp, 2, 1)  ## change from (1, 161, 50 ) to (1, 50, 161)  
        
        gate_inp = torch.pow(gate_inp, 2)
        gate_inp = torch.div(gate_inp, 0.1)
        pi_log = self.gate_network(gate_inp)
        pi_log = torch.transpose(pi_log, 2, 1) ## change from (1, 50, 161) to (1, 161, 50)
        return self.gs(pi_log, force_hard=True)


if __name__ == '__main__':

    npoles = 161
    binarize = Binarization(npoles)
    logits = torch.randn(1, 161, 50)
    
    out = binarize(logits)

    print('check')
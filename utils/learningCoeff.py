import numpy as np
import torch.nn as nn
import torch

class generatingCoeff(nn.Module):
    def __init__(self, inShape, outShpae):
        super(generatingCoeff, self).__init__()
        self.inShae = inShape
        self.outShape = outShpae
        self.layer1 = nn.Linear(self.inShape,512 )
        self.layer2 = nn.Linear(512, 1024)
        self.layer3 = nn.Linear(1024, 512)
        self.layer4 = nn.Linear(512, self.outShape)


import torch
import torch.nn as nn

class HardSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):

        y_hard = input.clone()
        y_hard = y_hard.zero_()
        y_hard[input >= 0.6] = 1

        return y_hard

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Sigmoid(torch.nn.Module):
    def __init__(self):
        """
        Implementation of softmax for a binary case using sigmoid.
        """
        super(Sigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def custom_sigmoid(self, logits, hard=False):

        out = self.sigmoid(logits)

        if hard:
            out = HardSoftmax.apply(out)

        return out

    def forward(self, logits, force_hard=False):
        
        return self.custom_sigmoid(logits, hard=force_hard)
            

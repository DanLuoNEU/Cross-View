# Copyright (c) 2021 Qualcomm Technologies, Inc.

# All Rights Reserved.

#  ============================================================================
#  @@-COPYRIGHT-START-@@
#
# Adapted and modified from the code by Andreas Veit:
# https://github.com/andreasveit/convnet-aig/blob/master/gumbelmodule.py
# Gumbel Softmax Sampler
# Works for categorical and binary input
#
# BSD 3-Clause License
#
# Copyright (c) 2018, Andreas Veit
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#  @@-COPYRIGHT-END-@@
#  ============================================================================


import torch
import torch.nn as nn
import pdb

class HardSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        y_hard = input.clone()
        y_hard = y_hard.zero_()
        a = torch.mean(input).data.item()
        b = 1-a
        if a < b:
            y_hard[input > b] = 1
            y_hard[input <= a] = 1
        else:
            y_hard[input > a] = 1
            y_hard[input <= b] = 1

        # y_hard[input >= 0.6] =1

        return y_hard

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class GumbelSigmoid(torch.nn.Module):
    def __init__(self):
        """
        Implementation of gumbel softmax for a binary case using gumbel sigmoid.
        """
        super(GumbelSigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumbel_samples_tensor = -torch.log(
            eps - torch.log(uniform_samples_tensor + eps)
        )

        return gumbel_samples_tensor

    def gumbel_sigmoid_sample(self, logits, temperature, inference=False):
        """Adds noise to the logits and takes the sigmoid. No Gumbel noise during inference."""
        if not inference:
            gumbel_samples_tensor = self.sample_gumbel_like(logits.data)
            gumbel_trick_log_prob_samples = logits + gumbel_samples_tensor.data
        else:
            gumbel_trick_log_prob_samples = logits

        soft_samples = self.sigmoid(gumbel_trick_log_prob_samples / temperature)
        # pdb.set_trace()
        return soft_samples

    def gumbel_sigmoid(self, logits, temperature=2 / 3, hard=False, inference=False):
        out = self.gumbel_sigmoid_sample(logits, temperature, inference)
        if hard:
            out = HardSoftmax.apply(out)
        # pdb.set_trace()
        return out

    def forward(self, logits, force_hard=False, temperature=2 / 3, inference=False):
        # inference = not self.training
        inference = inference
        # if self.training and not force_hard:
        if not inference and not force_hard:
            return self.gumbel_sigmoid(
                logits, temperature=temperature, hard=False, inference=inference
            )
        else:
            return self.gumbel_sigmoid(
                logits, temperature=temperature, hard=True, inference=inference
            )


if __name__ == '__main__':
    gumbelSigmoid = GumbelSigmoid()

    # logits = torch.randn(1, 161)
    logits = torch.tensor([0.001, -0.001, 0.12, 0.04, 0.0001])
    out = gumbelSigmoid(logits, force_hard=True, temperature=0.01, inference=True)

    print('check')
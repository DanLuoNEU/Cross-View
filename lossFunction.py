import torch

def hashingLoss(b1, b2, y, m, alpha, gpu_id):
    mesLoss = torch.nn.MSELoss()
    l1Loss = torch.nn.L1Loss()

    # l1 = 0.5 * (1-y)*dis(b1,b2)
    l1 = (1-y)*mesLoss(b1,b2)

    # l2 = 0.5 * y * max(margin-dis(b1,b2), 0)
    zeros = torch.zeros_like(l1).cuda(gpu_id)
    margin = m * torch.ones_like(l1).cuda(gpu_id)
    l2 = y * torch.max(margin - mesLoss(b1, b2), zeros)

    # l3 = alpha * ((l1_dis(abs(b1),1) + l1_dis(abs(b2),1))
    ONE = torch.ones_like(b1).cuda(gpu_id)
    l3 = (l1Loss(torch.abs(b1), ONE) + l1Loss(torch.abs(b2), ONE))

    loss = 0.5*l1 + 0.5*l2 + alpha*l3

    return loss, l1, l2, l3

def binaryLoss(b,gpu_id):
    l1Loss = torch.nn.L1Loss()
    ONE = torch.ones_like(b).cuda(gpu_id)
    biLoss = (l1Loss(torch.abs(b), ONE))
    return biLoss


def CrossEntropyLoss(c, b, gpu_id):
    # c = c.squeeze(0).permute(1, 0)
    'exp expression'
    # output = (b + 1)/torch.sum(b+1, dim=1).unsqueeze(1)
    k, _ = torch.max(c, dim=1)
    kk = k.unsqueeze(1)
    # target = (torch.exp(torch.abs(c) - kk) - torch.exp(-kk))/\
    #     (torch.sum(torch.exp(torch.abs(c)-kk), dim=1).unsqueeze(1) - torch.exp(-kk))
    n = torch.exp(torch.abs(c) - kk) - torch.exp(-kk)
    target = n/torch.sum(n, dim=1) #p

    # target = (torch.exp(torch.abs(c)) - 1)/torch.sum(torch.exp(torch.abs(c)-1),dim=1)

    output = (b+1)/torch.sum(b+1, dim=1).unsqueeze(1) # q
    # target = torch.abs(c)/torch.sum(torch.abs(c), dim=1).unsqueeze(1)

    # print('target:', target[0])
    # pos_weight = torch.ones([output.shape[1]])
    # crossEntropy = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda(gpu_id)
    # crossEntropy = torch.nn.BCELoss().cuda(gpu_id)
    m = torch.nn.Sigmoid().cuda(gpu_id)
    crossEntropy = torch.nn.MultiLabelSoftMarginLoss().cuda(gpu_id)
    # CEloss = crossEntropy(output, target)

    # CEloss = crossEntropy(m(output), target)

    CEloss = - torch.sum(output * torch.log(target))

    return CEloss

def sparsityLoss(c, b):
    lam = 2
    sumB = torch.sum(b)
    sparseLoss = - torch.sum(torch.matmul(b, torch.log(torch.abs(c+1e-6)))) + lam*sumB

    return sparseLoss

if __name__ == '__main__':
    gpu_id = 1
    c = torch.randn((1,161)).cuda(gpu_id)
    b = torch.randn((1,161)).cuda(gpu_id)

    ce = CrossEntropyLoss(c, b, gpu_id)

    print('check')
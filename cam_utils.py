from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import *
import scipy.io
from modelZoo.networks import *
from torch.autograd import Variable
from scipy.spatial import distance
import torch.nn
from modelZoo.sparseCoding import sparseCodingGenerator
from modelZoo.actHeat import *
from dataset.NUCLA_ViewProjection_trans import *
from dataset.NUCLA_viewProjection_CS import *
from dataset.NTURGBDsubject import *
from modelZoo.DyanOF import *
from dataset.NTU_viewProjection import *
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import *
from lossFunction import binaryLoss

gpu_id = 0
num_workers = 4
fistaLam = 0.1
print('gpu_id: ',gpu_id)
print('num_workers: ',num_workers)
print('fistaLam: ',fistaLam)

PRE = 0
T = 36
dataset = 'NUCLA'

Alpha = 0.1
lam1 = 2
lam2 = 1
N = 80*2
Epoch = 150
num_class = 10
dataType = '2D'

clip = 'Multi'
fusion = False

def load_data():

    if dataset == 'NUCLA':
        num_class = 10
        path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
        
        'CS:'

        'CV:'
        testSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='1,2', T=T,
                                    target_view='view_2',
                                    project_view='view_1', test_view='view_3')
        testloader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=num_workers)

    elif dataset == 'NTU':
        num_class = 60
        if dataType == '3D':
            root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
        else:
            root_skeleton = "/data/NTU-RGBD/poses"
        nanList = list(np.load('./NTU_badList.npz')['x'])
        'CS:'

        'CV:'
        testSet = NTURGBD_viewProjection(root_skeleton, root_list="/data/Yuexi/NTU-RGBD/list/", nanList=nanList, dataType= dataType, clip=clip,
                                    phase='train', T=36, target_view='C002', project_view='C001', test_view='C003')
        testloader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=num_workers)

    return testloader

def load_model():

    modelRoot = '/home/balaji/Documents/code/RSL/CS_CV/Cross-View/models/'
    modelPath = modelRoot + dataset + '/0228/CV_dynamicsStream_fista01_reWeighted_sqrC_T72/'

    map_location = torch.device(gpu_id)
    stateDict = torch.load(os.path.join(modelPath, '15.pth'), map_location=map_location)['state_dict']

    return stateDict

def load_net(num_class, stateDict):

    Drr = stateDict['dynamicsClassifier.sparseCoding.rr']
    Dtheta = stateDict['dynamicsClassifier.sparseCoding.theta']

    kinetics_pretrain = './pretrained/i3d_kinetics.pth'
    net = twoStreamClassification(num_class=num_class, Npole=(2*N+1), num_binary=(2*N+1), Drr=Drr, Dtheta=Dtheta,
                                    dim=2, gpu_id=gpu_id, dataType=dataType, fistaLam=fistaLam,  kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)

    net.load_state_dict(stateDict)
    net.eval()

    return net

class GradCamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        
        stateDict = load_model()
        self.net = load_net(num_class, stateDict)

        for name, param in self.net.named_parameters():
            param.requires_grad = True
            #print(name, param.data.shape)

        self.layerhook.append(self.net.RGBClassifier.layer1.register_forward_hook(self.forward_hook()))

        
    def activations_hook(self,grad):
        self.gradients = grad

    def get_act_grads(self):
        
        return self.gradients

    def forward_hook(self):

        def hook(module, inp, out):    
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
            
        return hook

    def forward(self, input_clip, inputImg_clip, t, fusion=False):
        label_clip, b, outClip_v = self.net(input_clip, inputImg_clip, t, fusion)
        return label_clip, b, outClip_v, self.selected_out

if __name__ == '__main__':
    
    gcmodel = GradCamModel().to('cuda:0')
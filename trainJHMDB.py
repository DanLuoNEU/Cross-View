from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import *
from lossFunction import binaryLoss
from dataset.JHMDB_dloader import *
gpu_id = 1
num_workers = 8
PRE = 0

T = 36
dataset = 'NUCLA'
# dataset = 'NTU'
Alpha = 0.5
lam1 = 1
lam2 = 1
N = 40*4
Epoch = 100
dataType = '2D'
# clip = 'Single'
clip = 'Multi'
fusion = False
num_class = 12

P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'
# saveModel = os.path.join(modelRoot, dataset, '/BinarizeSparseCode_m32A1')
saveModel = modelRoot + dataset + '/2Stream/train_t36_JHMDB/'
if not os.path.exists(saveModel):
    os.mkdir(saveModel)

dataRoot = '/data/Yuexi/JHMDB'

trainAnnot, testAnnot, actionList = get_train_test_annotation(dataRoot)
trainSet = jhmdbDataset(trainAnnot, testAnnot, T=T, split='train', if_occ=False)
trainloader = DataLoader(trainSet, batch_size=1, shuffle=False, num_workers=8)

testSet = jhmdbDataset(trainAnnot, testAnnot, T=T, split='test', if_occ=False)
testloader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=8)

kinetics_pretrain = './pretrained/i3d_kinetics.pth'
net = twoStreamClassification(num_class=num_class, Npole=(N+1), num_binary=(N+1), Drr=Drr, Dtheta=Dtheta,
                                  PRE=0, dim=2, gpu_id=gpu_id, dataType=dataType, kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)

# net = RGBAction(num_class=num_class, kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)
net.train()

# optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3, weight_decay=0.001, momentum=0.9)

optimizer = torch.optim.SGD([{'params':filter(lambda x: x.requires_grad, net.dynamicsClassifier.parameters()), 'lr':1e-4},
{'params':filter(lambda x: x.requires_grad, net.RGBClassifier.parameters()), 'lr':1e-6}], weight_decay=0.001, momentum=0.9)


scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 130], gamma=0.1)
Criterion = torch.nn.CrossEntropyLoss()
mseLoss = torch.nn.MSELoss()

# Criterion = torch.nn.BCELoss()
LOSS = []
ACC = []
print('training dataset:JHMDB')

for epoch in range(0, Epoch+1):
    print('start training epoch:', epoch)
    lossVal = []
    lossCls = []
    lossBi = []
    lossMSE = []
    for i, sample in enumerate(trainloader):
        y = sample['actLabel'].cuda(gpu_id)
        testImgSequence = sample['testImgSequence']
        testSkeleton = sample['testSkeleton']

        input_img = testImgSequence.float().cuda(gpu_id)
        input_skel = testSkeleton.float().cuda(gpu_id)
        t = input_skel.shape[2]
        # print('input_img shape:', input_img.shape, 'input_skel shape:', input_skel.shape)
        label = torch.zeros(input_skel.shape[1], num_class).cuda(gpu_id)
        clipBI = torch.zeros(input_skel.shape[1]).cuda(gpu_id)
        clipMSE = torch.zeros(input_skel.shape[1]).cuda(gpu_id)


        # label_rgb = torch.zeros(input_img.shape[1], num_class).cuda(gpu_id)
        # label_dym = torch.zeros(input_skel.shape[1], num_class).cuda(gpu_id)


        for clip in range(0, input_img.shape[1]):
            'two stream model'
            skel_clip = input_skel[:, clip, :, :, :].reshape(1, t, -1)

            img_clip = input_img[:, clip, :, :, :]

            # print(skel_clip.shape, img_clip.shape)
            label_clip, b, outClip = net(skel_clip, img_clip, t, fusion)
            label[clip] = label_clip

            clipBI[clip] = binaryLoss(b, gpu_id)

            clipMSE[clip] = mseLoss(outClip, skel_clip)

            # print('bi loss:', clipBI[clip])

        label = torch.mean(label, 0, keepdim=True)
        # print('action label:', label, 'bi score:',torch.mean(clipBI) )
        loss = lam1 * Criterion(label, y) + Alpha * (torch.mean(clipBI)) + lam2 * (torch.mean(clipMSE))

        loss.backward()
        optimizer.step()
        lossVal.append(loss.data.item())
        lossCls.append(Criterion(label, y).data.item())
        lossBi.append((torch.mean(clipBI)).data.item())
        lossMSE.append((torch.mean(clipMSE)).data.item())
        # print('cls:', Criterion(label, y).data.item(),'|bi:',torch.mean(clipBI).data.item(),'|mse:',torch.mean(clipMSE).data.item())

    loss_val = np.mean(np.array(lossVal))

    print('epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|Bi:',
          np.mean(np.array(lossBi)),
          '|mse:', np.mean(np.array(lossMSE)))

    scheduler.step()
    # if epoch % 5 == 0:
    #     torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
    #                 'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')

    if epoch % 1 == 0:
        print('start validating:')
        count = 0
        pred_cnt = 0
        Acc = []
        with torch.no_grad():
            for i, sample in enumerate(testloader):
                inputSkeleton = sample['testSkeleton'].float().cuda(gpu_id)
                inputImage = sample['testImgSequence'].float().cuda(gpu_id)

                t = inputSkeleton.shape[2]
                y = sample['actLabel'].data.item()
                label = torch.zeros(inputSkeleton.shape[1], num_class)
                for i in range(0, inputSkeleton.shape[1]):

                    input_clip = inputSkeleton[:, i, :, :, :].reshape(1, t, -1)
                    inputImg_clip = inputImage[:, i, :, :, :]
                    # label_clip, _, _ = net(input_clip, t) # DY+BL+CL
                    # label_clip, _ = net(input_clip, t) # DY+CL


                    label_clip, _, _ = net(input_clip, inputImg_clip, t, fusion)
                    # label_clip = net(inputImg_clip)
                    label[i] = label_clip
                label = torch.mean(label, 0, keepdim=True)

                pred = torch.argmax(label).data.item()
                # print('sample:',i, 'pred:', pred, 'gt:', y)
                count += 1

                if pred == y:
                    pred_cnt += 1

                Acc = pred_cnt / count

                print('epoch:', epoch, 'Acc:%.4f' % Acc, 'count:', count, 'pred_cnt:', pred_cnt)
                ACC.append(Acc)

print('done')

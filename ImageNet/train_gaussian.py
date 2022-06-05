import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision
import torch.nn.functional as F
from model_clamp import densenet121, squeezenet1_1, mobilenet_v2

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--clp', type=float, default=1.0, help='clip value')
parser.add_argument('--model_type', type=int, default=0, help='type of trained model')
parser.add_argument('--den', type=int, default=1, help='density of clipping')
parser.add_argument('--dr', type=float, default=1.0)
args = parser.parse_args()


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        #self.mu = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        #self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.resnet = torchvision.models.resnet50(pretrained=True)
        #self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #self.fc = nn.Linear(1024, 1000)

    def register_hooks(self):
        def forward_hook_layer1(module, input, output):
            heat = torch.norm(output, dim=1)
            self.heat1.append(heat)
        def forward_hook_layer2(module, input, output):
            heat = torch.norm(output, dim=1)
            self.heat2.append(heat)
        def forward_hook_layer3(module, input, output):
            heat = torch.norm(output, dim=1)
            self.heat3.append(heat)
        def forward_hook_layer4(module, input, output):
            heat = torch.norm(output, dim=1)
            self.heat4.append(heat)
        self.resnet.layer1.register_forward_hook(forward_hook_layer1)
        self.resnet.layer2.register_forward_hook(forward_hook_layer2)
        self.resnet.layer3.register_forward_hook(forward_hook_layer3)
        self.resnet.layer4.register_forward_hook(forward_hook_layer4)

    def clamp(self, x, a=1.0, dr=1.0):
        norm = torch.norm(x, dim=1, keepdim=True)
        thre = torch.mean(torch.mean(a * norm, dim=2, keepdim=True), dim=3, keepdim=True)
        x = x / torch.clamp_min(norm, min=1e-7)
        mask = (norm > thre).float()
        normd = thre * torch.exp(-1 / thre / thre * (norm - thre) * (norm - thre) * math.log(dr))
        norm = norm * (1 - mask) + normd * mask
        x = x * norm
        return x

    def features(self, input):
        # self.heat1 = []
        # self.heat2 = []
        # self.heat3 = []:q
        # self.heat4 = []
        #x = (input - self.mu) / self.std
        x = self.resnet.conv1(input)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # x = self.resnet.layer1(x)
        # x = self.clamp(x)
        # x = self.resnet.layer2(x)
        # x = self.clamp(x)
        # x = self.resnet.layer3(x)
        # x = self.clamp(x)
        # x = self.resnet.layer4(x)
        # x = self.clamp(x)

        k = 1
        for i in range(len(self.resnet.layer1)):
            b = self.resnet.layer1[i]
            x = b(x)
            # x = self.clamp(x, 2.0 - i/(len(self.resnet.layer1)-1))
            if k % args.den == 0:
                x = self.clamp(x, args.clp, args.dr)
            k += 1
        for i in range(len(self.resnet.layer2)):
            b = self.resnet.layer2[i]
            x = b(x)
            # x = self.clamp(x, 2.0 - i/(len(self.resnet.layer2)-1))
            if k % args.den == 0:
                x = self.clamp(x, args.clp, args.dr)
            k += 1
        for i in range(len(self.resnet.layer3)):
            b = self.resnet.layer3[i]
            x = b(x)
            # x = self.clamp(x, 2.0 - i/(len(self.resnet.layer3)-1))
            if k % args.den == 0:
                x = self.clamp(x, args.clp, args.dr)
            k += 1
        for i in range(len(self.resnet.layer4)):
            b = self.resnet.layer4[i]
            x = b(x)
            # x = self.clamp(x, 2.0 - i/(len(self.resnet.layer4)-1))
            if k % args.den == 0:
                x = self.clamp(x, args.clp, args.dr)
            k += 1

        return x

    def logits(self, features):
        x = self.resnet.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        #x3 = self.logits3(x3)
        return x

class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        # self.mu = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(
        #     3)
        # self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(
        #     3)
        self.incep = torchvision.models.inception_v3(pretrained=True, aux_logits=False)

    def clamp(self, x, a=1.0, dr=1.0):
        norm = torch.norm(x, dim=1, keepdim=True)
        thre = torch.mean(torch.mean(a * norm, dim=2, keepdim=True), dim=3, keepdim=True)
        x = x / torch.clamp_min(norm, min=1e-7)
        mask = (norm > thre).float()
        normd = thre * torch.exp(-1 / thre / thre * (norm - thre) * (norm - thre) * math.log(dr))
        norm = norm * (1 - mask) + normd * mask
        x = x * norm
        return x

    def features(self, input):
        #x = (input - self.mu) / self.std
        x = self.incep._transform_input(input)
        # N x 3 x 299 x 299
        x = self.incep.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.incep.Conv2d_2a_3x3(x)
        k = 2
        if k % args.den == 0:
            x = self.clamp(x, args.clp, args.dr)
        k += 1
        # N x 32 x 147 x 147
        x = self.incep.Conv2d_2b_3x3(x)
        if k % args.den == 0:
            x = self.clamp(x, args.clp, args.dr)
        k += 1
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.incep.Conv2d_3b_1x1(x)
        if k % args.den == 0:
            x = self.clamp(x, args.clp, args.dr)
        k += 1
        # N x 80 x 73 x 73
        x = self.incep.Conv2d_4a_3x3(x)
        if k % args.den == 0:
            x = self.clamp(x, args.clp, args.dr)
        k += 1
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.incep.Mixed_5b(x)
        if k % args.den == 0:
            x = self.clamp(x, args.clp, args.dr)
        k += 1
        # N x 256 x 35 x 35
        x = self.incep.Mixed_5c(x)
        if k % args.den == 0:
            x = self.clamp(x, args.clp, args.dr)
        k += 1
        # N x 288 x 35 x 35
        x = self.incep.Mixed_5d(x)
        if k % args.den == 0:
            x = self.clamp(x, args.clp, args.dr)
        k += 1
        # N x 288 x 35 x 35
        x = self.incep.Mixed_6a(x)
        if k % args.den == 0:
            x = self.clamp(x, args.clp, args.dr)
        k += 1
        # N x 768 x 17 x 17
        x = self.incep.Mixed_6b(x)
        if k % args.den == 0:
            x = self.clamp(x, args.clp, args.dr)
        k += 1
        # N x 768 x 17 x 17
        x = self.incep.Mixed_6c(x)
        if k % args.den == 0:
            x = self.clamp(x, args.clp, args.dr)
        k += 1
        # N x 768 x 17 x 17
        x = self.incep.Mixed_6d(x)
        if k % args.den == 0:
            x = self.clamp(x, args.clp, args.dr)
        k += 1
        # N x 768 x 17 x 17
        x = self.incep.Mixed_6e(x)
        if k % args.den == 0:
            x = self.clamp(x, args.clp, args.dr)
        k += 1
        # N x 768 x 17 x 17
        aux_defined = self.incep.training and self.incep.aux_logits
        if aux_defined:
            aux = self.incep.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.incep.Mixed_7a(x)
        if k % args.den == 0:
            x = self.clamp(x, args.clp, args.dr)
        k += 1
        # N x 1280 x 8 x 8
        x = self.incep.Mixed_7b(x)
        if k % args.den == 0:
            x = self.clamp(x, args.clp, args.dr)
        k += 1
        # N x 2048 x 8 x 8
        x = self.incep.Mixed_7c(x)
        if k % args.den == 0:
            x = self.clamp(x, args.clp, args.dr)
        # N x 2048 x 8 x 8
        return x, aux

    def logits(self, features):
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(features, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.incep.fc(x)
        return x

    def forward(self, x):
        x, aux = self.features(x)
        x = self.logits(x)
        return x

class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        # self.mu = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(
        #     3)
        # self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(
        #     3)
        self.dense = densenet121(pretrained=True, clamp=1.0)
        # sd0 = torch.load('./incep_best_fc.pth.tar')['state_dict']
        # sd0_new = {}
        # for key, value in sd0.items():
        #     new_key = key.replace('module.incep.', '')
        #     sd0_new[new_key] = value
        # self.incep.load_state_dict(sd0_new, strict=True)

    def features(self, input):
        #x = (input - self.mu) / self.std
        x = self.dense.features(input)
        return x

    def logits(self, features):
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.dense.classifier(out)
        return out

    def forward(self, x):
        features = self.features(x)
        out = self.logits(features)
        return out

class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeeze = squeezenet1_1(pretrained=True, clamp=1.0)

    def features(self, input):
        # x = (input - self.mu) / self.std
        x = self.squeeze.features(input)
        return x

    def logits(self, features):
        out = self.squeeze.classifier(features)
        return out

    def forward(self, x):
        features = self.features(x)
        out = self.logits(features)
        if len(out.shape) == 4:
            out = out.squeeze(-1).squeeze(-1)
        return out

class Mobile(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobile = mobilenet_v2(pretrained=True, clamp=args.clp, dr=args.dr, gaussian=True)

    def features(self, input):
        # x = (input - self.mu) / self.std
        x = self.mobile.features(input)
        x = self.mobile.clamp(x, args.clp, args.dr)
        return x

    def logits(self, features):
        x = nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)
        out = self.mobile.classifier(x)
        return out

    def forward(self, x):
        features = self.features(x)
        out = self.logits(features)
        return out

class Clamp(nn.Module):
    def __init__(self, thre=1.0):
        super().__init__()
        self.thre = thre

    def forward(self, x, w=None, meth=1):
        if meth > 0:
            cam = torch.sum(x * w, dim=1, keepdim=True)
            cam = cam - torch.clamp_max(torch.min(torch.min(cam, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0], max=0)
            thre = torch.mean(torch.mean(self.thre * cam, dim=2, keepdim=True), dim=3, keepdim=True)
            x = x / torch.clamp_min(cam, min=1e-2)
            if meth == 2:
                x = x * thre
            else:
                mask = (cam > thre).float()
                norm = cam * (1 - mask) + thre * mask
                x = x * norm
        return x

class ResNet50_cam(nn.Module):
    def __init__(self, clp):
        super(ResNet50_cam, self).__init__()
        self.mu = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.clamps1 = []
        self.clamps2 = []
        self.clamps3 = []
        self.clamps4 = []
        self.grads1 = []
        self.grads2 = []
        self.grads3 = []
        self.grads4 = []
        self.resnet = torchvision.models.resnet50(pretrained=True)

        for i in range(len(self.resnet.layer1)):
            self.clamps1.append(Clamp(clp))
        for i in range(len(self.resnet.layer2)):
            self.clamps2.append(Clamp(clp))
        for i in range(len(self.resnet.layer3)):
            self.clamps3.append(Clamp(clp))
        for i in range(len(self.resnet.layer4)):
            self.clamps4.append(Clamp(clp))
        self.register_hooks()

    def register_hooks(self):

        def backward_hook1(module, grad_in, grad_out):
            self.grads1.append(grad_out[0].clone().detach())
        def backward_hook2(module, grad_in, grad_out):
            self.grads2.append(grad_out[0].clone().detach())
        def backward_hook3(module, grad_in, grad_out):
            self.grads3.append(grad_out[0].clone().detach())
        def backward_hook4(module, grad_in, grad_out):
            self.grads4.append(grad_out[0].clone().detach())

        for i in range(len(self.resnet.layer1)):
            self.clamps1[i].register_backward_hook(backward_hook1)
        for i in range(len(self.resnet.layer2)):
            self.clamps2[i].register_backward_hook(backward_hook2)
        for i in range(len(self.resnet.layer3)):
            self.clamps3[i].register_backward_hook(backward_hook3)
        for i in range(len(self.resnet.layer4)):
            self.clamps4[i].register_backward_hook(backward_hook4)


    def features(self, input):
        self.grads1 = []
        self.grads2 = []
        self.grads3 = []
        self.grads4 = []
        # self.heat1 = []
        # self.heat2 = []
        # self.heat3 = []
        # self.heat4 = []
        #x0 = (input - self.mu) / self.std
        x0 = input
        x = self.resnet.conv1(x0)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # x = self.resnet.layer1(x)
        # x = self.clamp(x)
        # x = self.resnet.layer2(x)
        # x = self.clamp(x)
        # x = self.resnet.layer3(x)
        # x = self.clamp(x)
        # x = self.resnet.layer4(x)
        # x = self.clamp(x)

        for i in range(len(self.resnet.layer1)):
            b = self.resnet.layer1[i]
            x = b(x)
            #x = self.clamp(x, 2.0 - i/(len(self.resnet.layer1)-1))
            c = self.clamps1[i]
            x = c(x, meth=0)
        for i in range(len(self.resnet.layer2)):
            b = self.resnet.layer2[i]
            x = b(x)
            #x = self.clamp(x, 2.0 - i/(len(self.resnet.layer2)-1))
            c = self.clamps2[i]
            x = c(x, meth=0)
        for i in range(len(self.resnet.layer3)):
            b = self.resnet.layer3[i]
            x = b(x)
            #x = self.clamp(x, 2.0 - i/(len(self.resnet.layer3)-1))
            c = self.clamps3[i]
            x = c(x, meth=0)
        for i in range(len(self.resnet.layer4)):
            b = self.resnet.layer4[i]
            x = b(x)
            #x = self.clamp(x, 2.0 - i/(len(self.resnet.layer4)-1))
            c = self.clamps4[i]
            x = c(x, meth=0)

        x = self.logits(x)
        pre = torch.argmax(x, dim=1, keepdim=True)
        o = torch.sum(torch.gather(x, dim=1, index=pre))
        o.backward(retain_graph=True)
        self.zero_grad()
        x = self.resnet.conv1(x0)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        #print(len(self.clamps1), len(self.clamps2), len(self.clamps3), len(self.clamps4))
        #print(len(self.grads1), len(self.grads2), len(self.grads3), len(self.grads4))

        # x = self.resnet.layer1(x)
        # x = self.clamp(x)
        # x = self.resnet.layer2(x)
        # x = self.clamp(x)
        # x = self.resnet.layer3(x)
        # x = self.clamp(x)
        # x = self.resnet.layer4(x)
        # x = self.clamp(x)

        for i in range(len(self.resnet.layer1)):
            b = self.resnet.layer1[i]
            x = b(x)
            # x = self.clamp(x, 2.0 - i/(len(self.resnet.layer1)-1))
            w = torch.mean(torch.mean(self.grads1[-i-1], dim=2, keepdim=True), dim=3, keepdim=True)
            c = self.clamps1[i]
            x = c(x, w.detach(), meth=1)
        for i in range(len(self.resnet.layer2)):
            b = self.resnet.layer2[i]
            x = b(x)
            # x = self.clamp(x, 2.0 - i/(len(self.resnet.layer2)-1))
            w = torch.mean(torch.mean(self.grads2[-i - 1], dim=2, keepdim=True), dim=3, keepdim=True)
            c = self.clamps2[i]
            x = c(x, w.detach(), meth=1)
        for i in range(len(self.resnet.layer3)):
            b = self.resnet.layer3[i]
            x = b(x)
            # x = self.clamp(x, 2.0 - i/(len(self.resnet.layer3)-1))
            w = torch.mean(torch.mean(self.grads3[-i - 1], dim=2, keepdim=True), dim=3, keepdim=True)
            c = self.clamps3[i]
            x = c(x, w.detach(), meth=1)
        for i in range(len(self.resnet.layer4)):
            b = self.resnet.layer4[i]
            x = b(x)
            # x = self.clamp(x, 2.0 - i/(len(self.resnet.layer4)-1))
            w = torch.mean(torch.mean(self.grads4[-i - 1], dim=2, keepdim=True), dim=3, keepdim=True)
            c = self.clamps4[i]
            x = c(x, w.detach(), meth=1)

        return x

    def logits(self, features):
        x = self.resnet.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        #x3 = self.logits3(x3)
        return x

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()



best_acc1 = 0


def main():


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()
    # model = ResNet50()
    # model = Inception()
    # model = Dense()
    # model = Squeeze()
    #model = Mobile()
    im_size = 224
    if args.model_type == 0:
        model = ResNet50()
    elif args.model_type == 1:
        model = Inception()
        im_size = 299
    else:
        model = Mobile()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.model_type == 0:
        optimizer = torch.optim.SGD(model.module.resnet.fc.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.model_type == 1:
        optimizer = torch.optim.SGD(model.module.incep.fc.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.module.mobile.classifier.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    '''
    optimizer = torch.optim.SGD(model.module.resnet.fc.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    '''


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(im_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(im_size * 256 / 224)),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=int(args.batch_size/2), shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.start_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()
    model.apply(fix_bn)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    #with torch.no_grad():
    if 1:
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        if args.model_type == 0:
            shutil.copyfile(filename, 'resnet50_gaussian_clp{}_dr{}.pth.tar'.format(args.clp, args.dr))
        elif args.model_type == 1:
            shutil.copyfile(filename, 'incep_gaussian_clp{}_dr{}.pth.tar'.format(args.clp, args.dr))
        else:
            shutil.copyfile(filename, 'mobile_gaussian_clp{}_dr{}.pth.tar'.format(args.clp, args.dr))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()


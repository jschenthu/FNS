import os
import time
import argparse
import math
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from model_clamp import densenet121, squeezenet1_1, mobilenet_v2


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--bs', type=int, default=16, help='number of batch size')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--nonrob', action='store_true', default=False, help='if non-robust model')
parser.add_argument('--target', type=int, default=859, help='The target class')
parser.add_argument('--classes', type=int, default=1000, help='number of classes to attack')
parser.add_argument('--p_size', type=float, default=0.05, help='patch size. E.g. 0.05 ~= 5% of image')
parser.add_argument('--im_size', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--step', type=float, default=2 / 255, help='patch update step')
parser.add_argument('--data', default='ImageNet', help='folder of images to attack')
parser.add_argument('--save', default='patches_robust_exp', help='folder to output images and model checkpoints')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--clp', type=float, default=1.0, help='clamp ratio')
parser.add_argument('--advp', type=int, default=0, help='if using the AdvPatch attack')
parser.add_argument('--hwrat', type=float, default=1.0, help='the ratio of height to width')
parser.add_argument('--den', type=int, default=1, help='density of clipping')
parser.add_argument('--dr', type=float, default=1.0)
parser.add_argument('--model', type=int, default=0)
args = parser.parse_args()
if args.nonrob:
    args.save = args.save + '_model{}_clp{}_dr{}_nonrob_advp{}'.format(args.model, args.clp, args.dr, args.advp)
else:
    args.save = args.save + '_model{}_clp{}_dr{}_rob_advp{}'.format(args.model, args.clp, args.dr, args.advp)
if args.model == 1:
    args.im_size = 299
print(args)



class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.mu = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        if args.nonrob:
            self.resnet = torchvision.models.resnet50(pretrained=True)
        else:
            self.resnet = torchvision.models.resnet50()
            sd0 = torch.load('resnet50_exp_clp{}_dr{}.pth.tar'.format(args.clp, args.dr))['state_dict']
            sd = {}
            for k, v in sd0.items():
                if k[0:len('module.resnet.')] == 'module.resnet.':
                    sd[k[len('module.resnet.'):]] = v
            self.resnet.load_state_dict(sd, strict=True)

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
        x = x / torch.clamp_min(norm, min=1e-7)
        thre = torch.mean(torch.mean(a * norm, dim=2, keepdim=True), dim=3, keepdim=True)
        mask = (norm > thre).float()
        normd = thre * torch.exp(-1 / thre * (norm - thre) * math.log(dr))
        norm = norm * (1 - mask) + normd * mask
        x = x * norm
        return x

    def features(self, input):
        x = (input - self.mu) / self.std
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)


        k = 1
        for i in range(len(self.resnet.layer1)):
            b = self.resnet.layer1[i]
            x = b(x)
            if k % args.den == 0:
                x = self.clamp(x, args.clp, args.dr)
            k += 1
        for i in range(len(self.resnet.layer2)):
            b = self.resnet.layer2[i]
            x = b(x)
            if k % args.den == 0:
                x = self.clamp(x, args.clp, args.dr)
            k += 1
        for i in range(len(self.resnet.layer3)):
            b = self.resnet.layer3[i]
            x = b(x)
            if k % args.den == 0:
                x = self.clamp(x, args.clp, args.dr)
            k += 1
        for i in range(len(self.resnet.layer4)):
            b = self.resnet.layer4[i]
            x = b(x)
            if k % args.den == 0:
                x = self.clamp(x, args.clp, args.dr)
            k += 1

        return x

    def logits(self, features):
        x = self.resnet.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)
        return x

    def forward(self, x):
        if args.clp > 0:
            x= self.features(x)
            x = self.logits(x)
        else:
            x = (x - self.mu) / self.std
            x = self.resnet(x)
        return x


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.mu = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(
            3)
        self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(
            3)
        if args.nonrob:
            self.incep = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        else:
            self.incep = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
            sd0 = torch.load('incep_exp_clp{}_dr{}.pth.tar'.format(args.clp, args.dr))['state_dict']
            sd = {}
            for k, v in sd0.items():
                if k[0:len('module.incep.')] == 'module.incep.':
                    sd[k[len('module.incep.'):]] = v
            self.incep.load_state_dict(sd, strict=True)

    def clamp(self, x, a=1.0, dr=1.0):
        norm = torch.norm(x, dim=1, keepdim=True)
        thre = torch.mean(torch.mean(a * norm, dim=2, keepdim=True), dim=3, keepdim=True)
        x = x / torch.clamp_min(norm, min=1e-7)
        mask = (norm > thre).float()
        normd = thre * torch.exp(-1 / thre * (norm - thre) * math.log(dr))
        norm = norm * (1 - mask) + normd * mask
        x = x * norm
        return x

    def features(self, input):
        x = (input - self.mu) / self.std
        x = self.incep._transform_input(x)
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
        return x

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
        if args.clp > 0:
            x= self.features(x)
            x = self.logits(x)
        else:
            x = (x - self.mu) / self.std
            x = self.incep(x)
        return x


class Mobile(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(
            3)
        self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(
            3)

        if args.nonrob:
            self.mobile = mobilenet_v2(pretrained=True, clamp=args.clp, dr=args.dr, gaussian=False)
        else:
            self.mobile = mobilenet_v2(clamp=args.clp, dr=args.dr, gaussian=False)
            sd0 = torch.load('mobile_exp_clp{}_dr{}.pth.tar'.format(args.clp, args.dr))['state_dict']
            sd = {}
            for k, v in sd0.items():
                if k[0:len('module.mobile.')] == 'module.mobile.':
                    sd[k[len('module.mobile.'):]] = v
            self.mobile.load_state_dict(sd, strict=True)
        if args.clp == 0:
            self.mobile = torchvision.models.mobilenet_v2(pretrained=True)



    def features(self, input):
        x = (input - self.mu) / self.std
        # x = (input - self.mu) / self.std
        x = self.mobile.features(x)
        x = self.mobile.clamp(x, args.clp, args.dr)
        return x

    def logits(self, features):
        x = nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)
        out = self.mobile.classifier(x)
        return out

    def forward(self, x):
        if args.clp > 0:
            x= self.features(x)
            x = self.logits(x)
        else:
            x = (x - self.mu) / self.std
            x = self.mobile(x)
        return x



if not os.path.exists(args.save + '/' + str(args.target)):
    os.makedirs(args.save + '/' + str(args.target))

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

if args.model == 0:
    net = ResNet50()
elif args.model == 1:
    net = Inception()
else:
    net = Mobile()


if args.cuda:
    net.cuda()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

idx = np.arange(50)
training_idx = np.array([])
for i in range(1):
    training_idx = np.append(training_idx, [idx[i * 50:i * 50 + 10]])
training_idx = training_idx.astype(np.int32)

train_loader = torch.utils.data.DataLoader(
    dset.ImageFolder(args.data, transforms.Compose([
        transforms.Resize(round(args.im_size * 1.050)),
        transforms.CenterCrop(args.im_size),
        transforms.ToTensor(),
    ])),
    batch_size=args.bs, shuffle=False, sampler=None,
    num_workers=2, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    dset.ImageFolder(args.data, transforms.Compose([
        transforms.Resize(round(args.im_size * 1.000)),
        transforms.CenterCrop(args.im_size),
        transforms.ToTensor(),
    ])),
    batch_size=args.bs, shuffle=False, sampler=None,
    num_workers=2, pin_memory=True)



# Single patch-------------------------------------------------------------------
def InitPatch(im_size, p_size):
    mask = torch.ones(im_size)
    b, c, h, w = im_size
    side = int(math.sqrt(h * w * p_size))
    rx = np.random.randint(0, h - side)
    ry = np.random.randint(0, w - side)
    mask[:, :, rx:rx + side, ry:ry + side] = 0
    noise = np.zeros(im_size)
    # generate from image----------------------------------------------
    pic = np.zeros((h, w, 3)).astype(np.uint8)
    picp_bgr = cv2.resize(pic, (side, side))
    picp = cv2.transpose(picp_bgr[:, :, ::-1])
    # generate randomly------------------------------------------------
    # noise[:,:, rx:rx+side, ry:ry+side] = np.random.randint(0,255,[b,c,side,side])/255.
    noise[:, :, rx:rx + side, ry:ry + side] = np.transpose(picp) / 255
    noise = noise.astype(np.float32)
    noise = torch.from_numpy(noise)
    if args.cuda:
        mask, noise = mask.cuda(), noise.cuda()
    return mask, noise, rx, ry, side

def InitPatchB(im_size, p_size, rxy=None):
    b, c, h, w = im_size
    side = math.sqrt(h * w * p_size)
    sidex = int(side * math.sqrt(args.hwrat))
    sidey = int(side / math.sqrt(args.hwrat))
    if rxy is None:
        rx = np.random.randint(0, w - sidex, size=(b, ))
        ry = np.random.randint(0, h - sidey, size=(b, ))
    else:
        rx = rxy[0] * np.ones((b,))
        ry = rxy[1] * np.ones((b,))
    patch = np.zeros([1,c,sidey,sidex])
    rx = torch.from_numpy(rx).type(torch.long)
    ry = torch.from_numpy(ry).type(torch.long)
    patch = torch.from_numpy(patch).type(torch.float32)
    if args.cuda:
        rx, ry, patch = rx.cuda(), ry.cuda(), patch.cuda()
    return patch, rx, ry



def Inherit(mask_old, noise_old, rx_old, ry_old, side):
    mask = torch.ones(mask_old.shape)
    b, c, h, w = mask_old.shape
    rx = np.random.randint(0, h - side)
    ry = np.random.randint(0, w - side)
    mask[:, :, rx:rx + side, ry:ry + side] = 0
    noise = torch.zeros(mask_old.shape)
    noise[:, :, rx:rx + side, ry:ry + side] = noise_old[:, :, rx_old:rx_old + side, ry_old:ry_old + side]
    if args.cuda:
        mask, noise = mask.cuda(), noise.cuda()
    return mask, noise, rx, ry, side


def InheritB(im_size, patch, rxy=None):
    b, c, h, w = im_size
    _, c, sidey, sidex = patch.shape
    if rxy is None:
        rx = torch.randint(0, w - sidex, size=(b,), device=torch.device('cuda'))
        ry = torch.randint(0, h - sidey, size=(b,), device=torch.device('cuda'))
    else:
        rx = rxy[0] * torch.ones(b, device=torch.device('cuda'))
        ry = rxy[1] * torch.ones(b, device=torch.device('cuda'))
    return rx, ry

# Double patches-----------------------------------------------------------------
def InitPatch2(im_size, p_size):
    mask = torch.ones(im_size)
    b, c, h, w = im_size
    side = int(math.sqrt(h * w * p_size))
    rx1 = np.random.randint(0, h - side)
    ry1 = np.random.randint(0, w - side)
    rx2 = np.random.randint(0, h - side)
    while rx2 > (rx1 - side) and rx2 < (rx1 + side):
        rx2 = np.random.randint(0, w - side)
    ry2 = np.random.randint(0, w - side)
    while ry2 > (ry1 - side) and ry2 < (ry1 + side):
        ry2 = np.random.randint(0, h - side)
    mask[:, :, rx1:rx1 + side, ry1:ry1 + side] = 0
    mask[:, :, rx2:rx2 + side, ry2:ry2 + side] = 0
    noise = np.zeros(im_size)
    noise[:, :, rx1:rx1 + side, ry1:ry1 + side] = np.random.randint(0, 255, [b, c, side, side]) / 255.
    noise[:, :, rx2:rx2 + side, ry2:ry2 + side] = np.random.randint(0, 255, [b, c, side, side]) / 255.
    noise = noise.astype(np.float32)
    noise = torch.from_numpy(noise)
    if args.cuda:
        mask, noise = mask.cuda(), noise.cuda()
    return mask, noise, rx1, ry1, rx2, ry2, side


def Inherit2(mask_old, noise_old, rx1_old, ry1_old, rx2_old, ry2_old, side):
    mask = torch.ones(mask_old.shape)
    b, c, h, w = mask_old.shape
    rx1 = np.random.randint(0, h - side)
    ry1 = np.random.randint(0, w - side)
    rx2 = np.random.randint(0, h - side)
    while rx2 > (rx1 - side) and rx2 < (rx1 + side):
        rx2 = np.random.randint(0, w - side)
    ry2 = np.random.randint(0, w - side)
    while ry2 > (ry1 - side) and ry2 < (ry1 + side):
        ry2 = np.random.randint(0, h - side)
    mask[:, :, rx1:rx1 + side, ry1:ry1 + side] = 0
    mask[:, :, rx2:rx2 + side, ry2:ry2 + side] = 0
    noise = torch.zeros(mask_old.shape)
    noise[:, :, rx1:rx1 + side, ry1:ry1 + side] = noise_old[:, :, rx1_old:rx1_old + side, ry1_old:ry1_old + side]
    noise[:, :, rx2:rx2 + side, ry2:ry2 + side] = noise_old[:, :, rx2_old:rx2_old + side, ry2_old:ry2_old + side]
    if args.cuda:
        mask, noise = mask.cuda(), noise.cuda()
    return mask, noise, rx1, ry1, rx2, ry2, side


def get_noise(patch, im_size, rx, ry):
    b, c, h, w = im_size
    _, c, h_, w_ = patch.shape
    idx_x0 = torch.arange(0, w_, device=torch.device('cuda')).view(1, 1, -1)
    idx_y0 = torch.arange(0, h_, device=torch.device('cuda')).view(1, -1, 1)
    zim = torch.zeros(1, c, args.im_size * args.im_size, device=torch.device('cuda'))
    shift_x = rx.view(-1, 1, 1)
    shift_y = ry.view(-1, 1, 1)
    idx_x = idx_x0 + shift_x
    idx_y = idx_y0 + shift_y
    idx = (idx_y * w + idx_x).unsqueeze(1).repeat(1, c, 1, 1).view(b, c, -1)
    mask = zim.repeat(b, 1, 1).scatter_(2, idx, 1.0).view(b, c, h, w)
    mask = 1.0 - mask
    noise = zim.repeat(b, 1, 1).scatter_(2, idx, patch.repeat(b, 1, 1, 1).view(b, c, -1)).view(b, c, h, w)
    return noise, mask



def attack(epoch, patch, rxy=None):
    net.eval()
    total = 0
    accu = 0.
    succ = 0.
    for batch_idx, (data, labels) in enumerate(train_loader):
        if args.cuda:
            data, labels = data.cuda(), labels.cuda()
        if epoch == 0 and batch_idx == 0:
            patch, rx, ry = InitPatchB(data.shape, args.p_size, rxy=rxy)
        else:
            rx, ry = InheritB(data.shape, patch, rxy=rxy)

        total += 1
        pre = labels
        for count in range(1):
            patch.requires_grad = True
            noise, mask = get_noise(patch, data.shape, rx, ry)
            adv = torch.mul(data, mask) + torch.mul(noise, 1 - mask)
            feat = net.features(adv)

            adv_out = net(adv)

            if args.advp == 1:
                ys = torch.zeros_like(labels, device=torch.device('cuda')) + args.target
                losst = F.cross_entropy(adv_out, ys)
            else:
                if args.target < 1000:
                    target = adv_out[:, args.target]
                    gndtru = torch.gather(adv_out, dim=1, index=pre.unsqueeze(1)).squeeze(1)
                    losst = torch.mean(gndtru - target)
                else:
                    losst = F.cross_entropy(adv_out, labels)
            loss = losst
            loss.backward()
            grad = patch.grad.clone()
            patch.grad.data.zero_()
            patch.requires_grad = False
            patch = torch.clamp(patch - lr * args.step * torch.sign(grad), 0, 1)
        pre = torch.argmax(adv_out, dim=1)
        succ += (pre == args.target).type(torch.float32).sum()
        accu += (pre == labels).type(torch.float32).sum()

    if 1:
        vutils.save_image(noise,
                          args.save + '/' + str(args.target) + '/' + str(epoch) + '_rob{}_noise.png'.format(
                              args.nonrob),
                          normalize=False)
        vutils.save_image(patch,
                          args.save + '/' + str(args.target) + '/' + str(epoch) + '_rob{}_patch.png'.format(
                              args.nonrob),
                          normalize=False)
        vutils.save_image(mask,
                          args.save + '/' + str(args.target) + '/' + str(epoch) + '_rob{}_mask.png'.format(
                              args.nonrob),
                          normalize=False)
    evaluate(epoch, patch)
    return patch

def evaluate(epoch, patch, rxy=None):
    net.eval()
    total = 0
    accu = 0.
    succ = 0.

    for batch_idx, (data, labels) in enumerate(train_loader):
        if args.cuda:
            data, labels = data.cuda(), labels.cuda()
        rx, ry = InheritB(data.shape, patch, rxy=rxy)
        total += data.shape[0]
        for count in range(1):
            patch.requires_grad = False
            noise, mask = get_noise(patch, data.shape, rx, ry)
            adv = torch.mul(data, mask) + torch.mul(noise, 1 - mask)
            adv_out = F.softmax(net(adv), dim=1)
        pre = torch.argmax(adv_out, dim=1)
        succ += (pre == args.target).type(torch.float32).sum()
        accu += (pre == labels).type(torch.float32).sum()
    print('Epoch:{:3d}, Classify Accuracy:{:.2f}, Target Success:{:.2f}'.format(epoch, accu / total * 100, succ / total * 100))



def evaluate_clean():
    net.eval()
    total = 0
    accu = 0.
    succ = 0.
    for batch_idx, (data, labels) in enumerate(test_loader):
        if args.cuda:
            data, labels = data.cuda(), labels.cuda()

        total += data.shape[0]
        for count in range(1):

            adv_out = F.softmax(net(data), dim=1)
        pre = torch.argmax(adv_out, dim=1).detach().cpu().numpy()
        accu += np.sum(pre == labels.detach().cpu().numpy())
        succ += np.sum(pre == 859)

    print('Clean Examples: Classify Accuracy:{:.2f}, Target Success:{:.2f}'.format(accu / total * 100, succ / total * 100))



if __name__ == '__main__':
    lr = 1.0
    evaluate_clean()
    patch, _, _ = InitPatchB([1, 3, args.im_size, args.im_size], args.p_size)

    for i in range(args.epoch):
        if i == 30:
            lr = lr / 2.
        patch = attack(i, patch)


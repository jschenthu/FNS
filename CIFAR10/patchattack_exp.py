import os
import argparse
import math
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from cifar_resnet import resnet110



parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--bs', type=int, default=80, help='number of batch size')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--nonrob', action='store_true', default=False, help='if non-robust model')
parser.add_argument('--target', type=int, default=5, help='The target class')
parser.add_argument('--classes', type=int, default=10, help='number of classes to attack')
parser.add_argument('--p_size', type=float, default=0.1, help='patch size. E.g. 0.05 ~= 5% of image')
parser.add_argument('--im_size', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--step', type=float, default=2 / 255, help='patch update step')
parser.add_argument('--save', default='patches_robust', help='folder to output images and model checkpoints')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--clp', type=float, default=1.0, help='clamp ratio')
parser.add_argument('--den', type=int, default=1, help='density of norm clipping')
parser.add_argument('--advp', type=int, default=0, help='if using the adversarial patch method proposed by Brown et al.')
parser.add_argument('--dr', type=float, default=1.0, help='decay rate')
args = parser.parse_args()
print(args)



class ResNet110(nn.Module):
    def __init__(self):
        super(ResNet110, self).__init__()
        self.mu = torch.from_numpy(np.array([0.4914, 0.4822, 0.4465])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.std = torch.from_numpy(np.array([0.2023, 0.1994, 0.2010])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        if args.nonrob:
            self.resnet = resnet110()
            sd0 = torch.load('ori_new/checkpoint.pth.tar')['state_dict']
            self.resnet.load_state_dict(sd0, strict=True)
        else:
            self.resnet = resnet110()
            sd0 = torch.load('roa9by9fine/checkpoint.pth.tar')['state_dict']
            sd = {}
            for k, v in sd0.items():
                if k[0:len('1.')] == '1.':
                    sd[k[len('1.'):]] = v
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
        self.resnet.layer1.register_forward_hook(forward_hook_layer1)
        self.resnet.layer2.register_forward_hook(forward_hook_layer2)
        self.resnet.layer3.register_forward_hook(forward_hook_layer3)

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


        for i in range(len(self.resnet.layer1)):
            b = self.resnet.layer1[i]
            x = b(x)
            if (len(self.resnet.layer1) - 1 - i) % args.den == 0:
                x = self.clamp(x, args.clp, args.dr)
        for i in range(len(self.resnet.layer2)):
            b = self.resnet.layer2[i]
            x = b(x)
            if (len(self.resnet.layer2) - 1 - i) % args.den == 0:
                x = self.clamp(x, args.clp, args.dr)
        for i in range(len(self.resnet.layer3)):
            b = self.resnet.layer3[i]
            x = b(x)
            if (len(self.resnet.layer3) - 1 - i) % args.den == 0:
                x = self.clamp(x, args.clp, args.dr)

        return x

    def logits(self, features):
        x = self.resnet.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


if not os.path.exists(args.save + '/' + str(args.target)):
    os.makedirs(args.save + '/' + str(args.target))

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)


net = ResNet110()



if args.cuda:
    net.cuda()


idx = np.arange(50)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

train_loader = torch.utils.data.DataLoader(
    dset.CIFAR10(root='../data', train=False, download=True, transform=transform_train),
    batch_size=args.bs, shuffle=False, sampler=None,
    num_workers=2, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    dset.CIFAR10(root='../data', train=False, download=True, transform=transform_test),
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
    side = int(math.sqrt(h * w * p_size))
    if rxy is None:
        rx = np.random.randint(0, w - side, size=(b, ))
        ry = np.random.randint(0, h - side, size=(b, ))
    else:
        rx = rxy[0] * np.ones((b,))
        ry = rxy[1] * np.ones((b,))
    patch = np.zeros([1,c,side,side])
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

def InheritB(im_size, p_size, rxy=None):
    b, c, h, w = im_size
    side = int(math.sqrt(h * w * p_size))
    if rxy is None:
        rx = np.random.randint(0, w - side, size=(b,))
        ry = np.random.randint(0, h - side, size=(b,))
    else:
        rx = rxy[0] * np.ones((b,))
        ry = rxy[1] * np.ones((b,))
    rx = torch.from_numpy(rx).type(torch.long)
    ry = torch.from_numpy(ry).type(torch.long)
    if args.cuda:
        rx, ry = rx.cuda(), ry.cuda()
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
    shift_x = rx.view(-1,1,1,1)
    shift_y = ry.view(-1,1,1,1)
    shift = torch.cat([shift_x, shift_y], dim=-1).repeat(1,h,w,1)
    idx_x = torch.arange(0, w).type(torch.long).view(1,-1).repeat(h,1)
    idx_y = torch.arange(0, h).type(torch.long).view(-1, 1).repeat(1, w)
    idx = torch.stack([idx_x, idx_y], dim=-1).view(1, h, w, 2).repeat(b,1,1,1)
    idx = idx.type(torch.float32)
    shift = shift.type(torch.float32)
    if args.cuda:
        idx = idx.cuda()
    idx = idx - shift #(N,h,w,2)
    mask_x = (idx[:, :, :, 0] >= 0).type(torch.float32) * (idx[:, :, :, 0] < w_).type(torch.float32)
    mask_y = (idx[:, :, :, 1] >= 0).type(torch.float32) * (idx[:, :, :, 1] < h_).type(torch.float32)
    mask = mask_x * mask_y #(N,h,w)
    mask = mask.view(b, 1, h, w).repeat(1, c, 1, 1)
    mask = mask.type(torch.float32)
    mask = 1.0 - mask
    idx[:, :, :, 0] = torch.clamp_max(torch.clamp_min(idx[:, :, :, 0], min=0), max=w_-1)
    idx[:, :, :, 0] = 2.0 * (idx[:, :, :, 0] / (w_ - 1.0)) - 1.0
    idx[:, :, :, 1] = torch.clamp_max(torch.clamp_min(idx[:, :, :, 1], min=0), max=h_-1)
    idx[:, :, :, 1] = 2.0 * (idx[:, :, :, 1] / (h_ - 1.0)) - 1.0
    patch = patch.repeat(b, 1, 1, 1)
    noise = F.grid_sample(patch, idx)
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
            rx, ry = InheritB(data.shape, args.p_size, rxy=rxy)

        total += 1
        pre = labels
        for count in range(1):
            patch.requires_grad = True
            noise, mask = get_noise(patch, data.shape, rx, ry)
            adv = torch.mul(data, mask) + torch.mul(noise, 1 - mask)
            adv_in = adv
            adv_out = net(adv_in)

            if args.advp == 0:
                target = adv_out[:, args.target]
                gndtru = torch.gather(adv_out, dim=1, index=pre.unsqueeze(1)).squeeze(1)
                losst = torch.mean(gndtru - target)
            else:
                lt = torch.zeros_like(labels).cuda() + args.target
                losst = F.cross_entropy(adv_out, lt)


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
    for batch_idx, (data, labels) in enumerate(test_loader):
        if args.cuda:
            data, labels = data.cuda(), labels.cuda()
        rx, ry = InheritB(data.shape, args.p_size, rxy=rxy)
        total += (labels != args.target).type(torch.float32).sum()
        for count in range(1):
            patch.requires_grad = False
            noise, mask = get_noise(patch, data.shape, rx, ry)
            adv = torch.mul(data, mask) + torch.mul(noise, 1 - mask)


            adv_in = adv


            adv_out = net(adv_in)

        pre = torch.argmax(adv_out, dim=1)
        succ += ((pre == args.target) * (labels != args.target)).type(torch.float32).sum()
        accu += ((pre == labels) * (labels != args.target)).type(torch.float32).sum()
    print('Epoch:{:3d}, Classify Accuracy:{:.2f}, Target Success:{:.2f}'.format(epoch, accu / total * 100, succ / total * 100))

def evaluate_clean():
    net.eval()
    total = 0
    accu = 0.
    succ = 0.
    for batch_idx, (data, labels) in enumerate(test_loader):
        if args.cuda:
            data, labels = data.cuda(), labels.cuda()


        total += (labels != args.target).type(torch.float32).sum()
        for count in range(1):
            adv = data
            adv_in = adv
            adv_out = net(adv_in)

        pre = torch.argmax(adv_out, dim=1)
        succ += ((pre == args.target) * (labels != args.target)).type(torch.float32).sum()
        accu += ((pre == labels) * (labels != args.target)).type(torch.float32).sum()

    print('Clean Examples: Classify Accuracy:{:.2f}, Target Success:{:.2f}'.format(accu / total * 100, succ / total * 100))

if __name__ == '__main__':
    lr = 1.0
    evaluate_clean()
    patch, _, _ = InitPatchB([1, 3, args.im_size, args.im_size], args.p_size)
    for i in range(args.epoch):
        if i == 15:
            lr = lr / 2.
        patch = attack(i, patch)

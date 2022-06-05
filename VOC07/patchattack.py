from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
import cv2
import math

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=10, help="size of each image batch")
parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
parser.add_argument("--val_path", type=str, default="clean/voc_test", help="path to validate data")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="coco.names", help="path to class label file")
parser.add_argument("--iou_thres", type=float, default=0.001, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument('--p_size', type=float, default=0.08, help='size of the patch')
parser.add_argument('--hwrat', type=float, default=1.0, help='the ratio of the height and the width of the patch')
parser.add_argument('--step', type=float, default=2.0 / 255, help='step of optimization')
parser.add_argument('--save', type=str, default='patches_baseline', help='save path')
parser.add_argument('--clp', type=float, default=1000, help='ratio of FNC')
parser.add_argument('--target', type=int, default=-1, help='target of attack')
parser.add_argument('--dr', type=float, default=1.0, help='decay rate')
parser.add_argument('--gaussian',  action='store_true', default=False, help='if using gaussian decay')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.system('mkdir '+opt.save)

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

valid_path = opt.val_path
class_names = load_classes('coco.names')

# Initiate model
model = Darknet(opt.model_def, opt.clp, opt.dr, opt.gaussian).to(device)
if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
elif opt.weights_path.endswith(".pt"):
    model.load_state_dict(torch.load(opt.weights_path, map_location=device)['model'], strict=True)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path, map_location=device), strict=True)

dataset = ListDataset(valid_path, img_size=opt.img_size, augment=False, multiscale=False)
testloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn
    )
trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn
    )

colors = plt.cm.hsv(np.linspace(0, 1, 81)).tolist()
norm = matplotlib.colors.Normalize(0, 1, clip=True)

def InitPatchB(im_size, p_size, rxy=None):
    b, c, h, w = im_size
    side = math.sqrt(h * w * p_size)
    sidex = int(side * math.sqrt(opt.hwrat))
    sidey = int(side / math.sqrt(opt.hwrat))
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
    rx, ry, patch = rx.cuda(), ry.cuda(), patch.cuda()
    return patch, rx, ry

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

def get_noise(patch, im_size, rx, ry):
    b, c, h, w = im_size
    _, c, h_, w_ = patch.shape
    idx_x0 = torch.arange(0, w_, device=torch.device('cuda')).view(1, 1, -1)
    idx_y0 = torch.arange(0, h_, device=torch.device('cuda')).view(1, -1, 1)
    zim = torch.zeros(1, c, opt.img_size * opt.img_size, device=torch.device('cuda'))
    shift_x = rx.view(-1, 1, 1)
    shift_y = ry.view(-1, 1, 1)
    idx_x = idx_x0 + shift_x
    idx_y = idx_y0 + shift_y
    idx = (idx_y * w + idx_x).unsqueeze(1).repeat(1, c, 1, 1).view(b, c, -1)
    mask = zim.repeat(b, 1, 1).scatter_(2, idx, 1.0).view(b, c, h, w)
    mask = 1.0 - mask
    noise = zim.repeat(b, 1, 1).scatter_(2, idx, patch.repeat(b, 1, 1, 1).view(b, c, -1)).view(b, c, h, w)
    return noise, mask

def attack(epoch, model, patch, lr):
    model.eval()

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(trainloader, desc="Detecting objects")):
        imgs = imgs.cuda()
        patch.requires_grad = True
        rx, ry = InheritB(imgs.shape, patch, rxy=None)
        noise, mask = get_noise(patch, imgs.shape, rx, ry)
        adv = torch.mul(imgs, mask) + torch.mul(noise, 1 - mask)
        outputs = model(adv)
        loss = 0
        for i in range(len(outputs)):
            if outputs[i] is not None:
                if opt.target < 0:
                    loss += torch.max(outputs[i][:,4])
                else:
                    assert opt.target < 80
                    loss += -torch.sum(torch.clamp(outputs[i][:,4] - 0.5, min=0).detach() * outputs[i][:,5 + opt.target])
        loss.backward()
        grad = patch.grad.clone()
        patch.requires_grad = False
        patch = torch.clamp(patch - lr * opt.step * torch.sign(grad), 0, 1)

    vutils.save_image(noise,
                      opt.save + '/'  + str(epoch) + '_noise.png',
                      normalize=False)
    vutils.save_image(patch,
                      opt.save + '/'  + str(epoch) + '_patch.png',
                      normalize=False)
    vutils.save_image(mask,
                      opt.save + '/'  + str(epoch) + '_mask.png',
                      normalize=False)

    precision, recall, AP, f1, ap_class, detratio = evaluate(model, patch, opt.iou_thres, opt.conf_thres, opt.nms_thres, opt.img_size)
    print(f"epoch: {epoch}, mAP: {AP.mean()}, nboxes: {detratio}")

    return patch




def evaluate(model, patch, iou_thres, conf_thres, nms_thres, img_size):
    model.eval()
    patch.requires_grad = False

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    num_detect = 0
    num_gt = 0
    succ_list = []
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(testloader, desc="Detecting objects")):
        # Extract labels
        labels += targets[:, 1].tolist()
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        num_gt += imgs.shape[0]

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        targets = targets.type(Tensor)

        rx, ry = InheritB(imgs.shape, patch, rxy=None)
        noise, mask = get_noise(patch, imgs.shape, rx, ry)
        adv = torch.mul(imgs, mask) + torch.mul(noise, 1 - mask)


        with torch.no_grad():
            outputs = model(adv)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            for i in range(len(outputs)):
                if outputs[i] is not None:
                    num_detect += outputs[i].shape[0]
                    succ_list.append(0)
                else:
                    succ_list.append(1)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    detratio = num_detect / num_gt


    return precision, recall, AP, f1, ap_class, detratio

def evaluate_clean(model, iou_thres, conf_thres, nms_thres, img_size):
    model.eval()
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    num_detect = 0
    num_gt = 0
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(testloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        num_gt += imgs.shape[0]

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        targets = targets.type(Tensor)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            for i in range(len(outputs)):
                if outputs[i] is not None:
                    num_detect += outputs[i].shape[0]



        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    detratio = num_detect / num_gt

    return precision, recall, AP, f1, ap_class, detratio




if __name__ == "__main__":
    print("Compute mAP...")

    precision, recall, AP, f1, ap_class, detratio = evaluate_clean(
        model,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(
            f"+ Class '{c}' ({class_names[c]}) - AP: {'%.5f' % AP[i]} - P: {'%.5f' % precision[i]} - R: {'%.5f' % recall[i]}")

    print(f"mAP: {AP.mean()}, nboxes: {detratio}")

    print('Start attacking...')

    patch, _, _ = InitPatchB([1, 3, opt.img_size, opt.img_size], opt.p_size)
    lr = 1.0

    for i in range(60):
        if i == 30:
            lr = lr / 2
        patch = attack(i, model, patch, lr)






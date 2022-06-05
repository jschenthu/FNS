import glob
import random
import os
import sys
import numpy as np
from PIL import Image, ExifTags
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import xml.etree.ElementTree as ET

def load_classes(path):
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

classes = load_classes('coco.names')

def fixXml(xml_path):
    tree=ET.parse(xml_path)   
    root = tree.getroot()
    size = root.find("size")
    width = int(size.find("width").text)  #读取宽
    height = int(size.find("height").text) #读取高
    big = max(width, height)
    bbox_info = []
    for obj in root.findall('object'):
        tmp = []
        cls = obj.find('name').text
        cls = classes.index(cls)
        bbox = obj.find('bndbox')
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)
        xc = (x1 + x2 + big - width) / (2. * big)
        yc = (y1 + y2 + big - height) / (2. * big)
        ww = max((x2 - x1) / (1.0 * big), 0.)
        hh = max((y2 - y1) / (1.0 * big), 0.)
        bbox_info.append([cls, xc, yc, ww, hh])  
    bbox_info = np.array(bbox_info)
    bbox_info = torch.from_numpy(bbox_info)
    return bbox_info

def fixXml1(xml_path):
    tree=ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    width = int(size.find("width").text)  #读取宽
    height = int(size.find("height").text) #读取高
    small = min(width, height)
    big = max(width, height)
    bbox_info = []
    for obj in root.findall('object'):
        tmp = []
        cls = obj.find('name').text
        cls = classes.index(cls)
        bbox = obj.find('bndbox')
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)
        xc = (x1 + x2 + small - width) / (2. * small)
        yc = (y1 + y2 + small - height) / (2. * small)
        ww = max((x2 - x1) / (1.0 * small), 0.)
        hh = max((y2 - y1) / (1.0 * small), 0.)
        bbox_info.append([cls, xc, yc, ww, hh])
    bbox_info = np.array(bbox_info)
    bbox_info = torch.from_numpy(bbox_info)
    return bbox_info, big, small

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def crop_to_square(img):
    c, h, w = img.shape
    crp_size = min(h, w)
    img = transforms.CenterCrop(crp_size)(img)

    return img


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = Image.open(img_path)
        
        try:
            for orientation in ExifTags.TAGS.keys() : 
                if ExifTags.TAGS[orientation]=='Orientation' : break 
            exif=dict(img._getexif().items())
            if   exif[orientation] == 3 : 
                img=img.rotate(180)
            elif exif[orientation] == 6 : 
                img=img.rotate(270)
            elif exif[orientation] == 8 : 
                img=img.rotate(90)
        except:
            pass
        
        img = transforms.ToTensor()(img)
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)

## for voc dataset
class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=False, multiscale=True, normalized_labels=False):
        self.img_files = os.listdir(list_path)
        self.img_files.sort()
        self.label_files = os.listdir(os.path.join('RawImage', 'voc_test', 'Annotations'))
        self.label_files.sort()
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.list_path = list_path

    def __getitem__(self, index):
        ##  Image
        img_path = self.img_files[index % len(self.img_files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(os.path.join(self.list_path, img_path)).convert('RGB'))
        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        ##  Label
        label_path = os.path.join('RawImage/voc_test/Annotations', self.label_files[index % len(self.label_files)])
        targets = None
        if os.path.exists(label_path):
            boxes = fixXml(label_path)
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)




## for coco dataset
#class ListDataset(Dataset):
    #def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        #with open(list_path, "r") as file:
            #self.img_files = file.readlines()

        #self.label_files = [
            #path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            #for path in self.img_files
        #]
        #self.img_size = img_size
        #self.max_objects = 100
        #self.augment = augment
        #self.multiscale = multiscale
        #self.normalized_labels = normalized_labels
        #self.min_size = self.img_size - 3 * 32
        #self.max_size = self.img_size + 3 * 32
        #self.batch_count = 0

    #def __getitem__(self, index):

        ## ---------
        ##  Image
        ## ---------

        #img_path = self.img_files[index % len(self.img_files)].rstrip()

        ## Extract image as PyTorch tensor
        #img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        ## Handle images with less than three channels
        #if len(img.shape) != 3:
            #img = img.unsqueeze(0)
            #img = img.expand((3, img.shape[1:]))

        #_, h, w = img.shape
        #h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        ## Pad to square resolution
        #img, pad = pad_to_square(img, 0)
        #_, padded_h, padded_w = img.shape

        ## ---------
        ##  Label
        ## ---------

        #label_path = self.label_files[index % len(self.img_files)].rstrip()

        #targets = None
        #if os.path.exists(label_path):
            #boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            ## Extract coordinates for unpadded + unscaled image
            #x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            #y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            #x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            #y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            ## Adjust for added padding
            #x1 += pad[0]
            #y1 += pad[2]
            #x2 += pad[1]
            #y2 += pad[3]
            ## Returns (x, y, w, h)
            #boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            #boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            #boxes[:, 3] *= w_factor / padded_w
            #boxes[:, 4] *= h_factor / padded_h

            #targets = torch.zeros((len(boxes), 6))
            #targets[:, 1:] = boxes

        ## Apply augmentations
        #if self.augment:
            #if np.random.random() < 0.5:
                #img, targets = horisontal_flip(img, targets)

        #return img_path, img, targets

    #def collate_fn(self, batch):
        #paths, imgs, targets = list(zip(*batch))
        ## Remove empty placeholder targets
        #targets = [boxes for boxes in targets if boxes is not None]
        ## Add sample index to targets
        #for i, boxes in enumerate(targets):
            #boxes[:, 0] = i
        #targets = torch.cat(targets, 0)
        ## Selects new image size every tenth batch
        #if self.multiscale and self.batch_count % 10 == 0:
            #self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        ## Resize images to input shape
        #imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        #self.batch_count += 1
        #return paths, imgs, targets

    #def __len__(self):
        #return len(self.img_files)

class ListDataset_crp(Dataset):
    def __init__(self, list_path, img_size=416, augment=False, multiscale=True, normalized_labels=False):
        self.img_files = os.listdir(list_path)
        self.img_files.sort()
        self.label_files = os.listdir(os.path.join('RawImage', 'voc_test', 'Annotations'))
        self.label_files.sort()
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.list_path = list_path

    def __getitem__(self, index):
        ##  Image
        img_path = self.img_files[index % len(self.img_files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(os.path.join(self.list_path, img_path)).convert('RGB'))
        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        #img = crop_to_square(img)
        #_, padded_h, padded_w = img.shape

        ##  Label
        label_path = os.path.join('RawImage/voc_test/Annotations', self.label_files[index % len(self.label_files)])
        targets = None
        if os.path.exists(label_path):
            boxes, big, small = fixXml1(label_path)
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
            crp_size = int(small / big * 416)
            img = transforms.CenterCrop(crp_size)(img)



        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
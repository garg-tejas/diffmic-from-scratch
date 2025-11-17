import os, torch, cv2, random
import numpy as np
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
from scipy.ndimage.morphology import binary_erosion
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import filters
import numpy as np
import imageio
import dataloader.transforms as trans
import json, numbers
from glob import glob
import pickle

class BUDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.trainsize = (224,224)
        self.train = train
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        img = Image.open(img_path).convert('RGB')
        img_torch = self.transform_center(img)
        label = int(data_pac['label'])
        return img_torch, label

    def __len__(self):
        return self.size


class APTOSDataset(Dataset):
    def __init__(self, data_list, train=True, preprocess_ben=False, ben_sigma=200):
        self.trainsize = (512,512)
        self.train = train
        self.preprocess_ben = preprocess_ben
        self.ben_sigma = ben_sigma
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)
        if train:
            self.transform_center = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random brightness/contrast
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),  # Random sharpness
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def ben_preprocess(self, img_np):
        try:
            img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            # Contrast enhancement
            img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), self.ben_sigma), -4, 128)

            # Circle mask to remove borders (set outside to black)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                contour = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)

                # Create and apply mask
                mask = np.zeros_like(img)
                cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)  # Filled white circle
                img = cv2.bitwise_and(img, mask)

            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return img_np  # Fallback to original

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        if self.preprocess_ben:
            img_np = self.ben_preprocess(img_np)
        img = Image.fromarray(img_np)

        img_torch = self.transform_center(img)
        label = int(data_pac['label'])
        return img_torch, label

    def __len__(self):
        return self.size



class ISICDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.trainsize = (224,224)
        self.train = train
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)

        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        img = Image.open(img_path).convert('RGB')
        img_torch = self.transform_center(img)
        label = int(data_pac['label'])
        return img_torch, label


    def __len__(self):
        return self.size


class ChestXrayDataSet(Dataset):
    def __init__(self, image_list_file, train=True):
        data_dir = "dataset/chest/all/images/images"
        self.trainsize = (256, 256)
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                label.append(1) if (np.array(label)==0).all() else label.append(0)
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
        if train:
            self.transform_center = transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        trans.RandomHorizontalFlip(),
                                        trans.RandomRotation(20),
                                        transforms.ToTensor(),
                                        normalize
                                    ])
        else:
            self.transform_center = transforms.Compose([
                                        transforms.Resize(224),
                                        transforms.ToTensor(),
                                        normalize
                                    ])

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        image = self.transform_center(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

import torch
import timm
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import pandas as pd
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import copy
from PIL import Image
import torchvision.transforms as transforms
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomBrightnessContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import *
from utils import *

# setting up the directory

data_dir="/scratch/sgarg75/Practice1"
files_path=os.path.join(data_dir,"dataset").replace("\\","/")
images_path=os.path.join(data_dir,"images").replace("\\","/")
device = torch.device("cuda")


#define a transformer

def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []

    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence



#Making a custom dataset

class ChestXray14Dataset(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14):

    self.img_list = []
    self.img_label = []
    #self.transform = transform
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))

#    if annotation_percent < 100:
 #     random.Random(99).shuffle(indexes)
  #    num_data = int(indexes.shape[0] * annotation_percent / 100.0)
   #   indexes = indexes[:num_data]

    _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
    self.img_list = []
    self.img_label = []

    for i in indexes:
      self.img_list.append(_img_list[i])
      self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)
   
    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

#define the datasets


dataset_train = ChestXray14Dataset(images_path, os.path.join(files_path,"Xray14_train_official.txt"),
                                           augment=build_transform_classification(normalize="chestx-ray", mode="train"))

dataset_val = ChestXray14Dataset(images_path, os.path.join(files_path,"Xray14_val_official.txt"),
                                         augment=build_transform_classification(normalize="chestx-ray", mode="valid"))
dataset_test = ChestXray14Dataset(images_path, os.path.join(files_path,"Xray14_test_official.txt"),
                                          augment=build_transform_classification(normalize="chestx-ray", mode="test"))



#define the dataloaders

data_loader_train = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True,
                                   num_workers=8, pin_memory=True)
data_loader_val = DataLoader(dataset=dataset_val, batch_size=64, shuffle=False,
                                 num_workers=8, pin_memory=True)


#define diseases
diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


#load the model with pretrained weights

model=convnext_base(pretrained=False,in_22k=False)
#model=models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
#model.classifier = nn.Sequential(nn.LayerNorm((1024,), eps=1e-06, elementwise_affine=True), nn.Flatten(start_dim=1, end_dim=-1), nn.Linear(in_features=1024, out_features=14, bias=True))
model.to(device)


#train one epoch

def train_one_epoch(data_loader_train, device,model, criterion, optimizer, epoch):

  model.train()

  for i, (samples, targets) in enumerate(data_loader_train):
    samples, targets = samples.float().to(device), targets.float().to(device)

    outputs = model(samples)

    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
#evaluate on the validation dataset

def evaluate(data_loader_val, device, model, criterion):
  losses=0
  with torch.no_grad():
        losses = MetricLogger('Loss', ':.4e')
        model.eval()
        for i, (samples, targets) in enumerate(data_loader_val):
            
            samples, targets = samples.float().to(device), targets.float().to(device)
        
            outputs = model(samples)
        
            loss = criterion(outputs, targets)
            
            losses.update(loss.item(), samples.size(0))
            losses.update(loss.item(), samples.size(0))
            

  return losses.avg




#train the model

#log_file = os.path.join(files_path, "models.log")
num_epoch=35

start_epoch = 0
init_loss = 1000000
#experiment = args.exp_name + "_run_" + str(i)
best_val_loss = init_loss
patience_counter = 0
#save_model_path = os.path.join(files_path, experiment)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, mode='min',
                                       threshold=0.0001, min_lr=0, verbose=True)
#model = build_classification_model(args)
#print(model)


for epoch in range(0, num_epoch):
    train_one_epoch(data_loader_train,device, model, criterion, optimizer, epoch)

    val_loss = evaluate(data_loader_val, device,model, criterion)

    lr_scheduler.step(val_loss)
    if val_loss < best_val_loss:
          print(
            "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}".format(epoch, best_val_loss, val_loss))

          best_val_loss = val_loss
          patience_counter = 0

          

    else:
          print("Epoch {:04d}: val_loss did not improve from {:.5f} ".format(epoch, best_val_loss ))
          patience_counter += 1

    



#test classification model

def test_classification(model, data_loader_test, device):
 
  model.eval()

  y_test = torch.FloatTensor().cuda()
  p_test = torch.FloatTensor().cuda()

  with torch.no_grad():
    for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
      targets = targets.cuda()
      y_test = torch.cat((y_test, targets), 0)

      if len(samples.size()) == 4:
        bs, c, h, w = samples.size()
        n_crops = 1
      elif len(samples.size()) == 5:
        bs, n_crops, c, h, w = samples.size()

      varInput = torch.autograd.Variable(samples.view(-1, c, h, w).cuda())

      out = model(varInput)
      outMean = out.view(bs, n_crops, -1).mean(1)
      p_test = torch.cat((p_test, outMean.data), 0)

  return y_test, p_test   

        
    
    
    
# test the model

print ("start testing.....")
output_file = os.path.join(os.path.join(data_dir,"Attempt2/Output1/scratch").replace("\\","/"),"results9.txt").replace("\\","/")

data_loader_test = DataLoader(dataset=dataset_test, batch_size=64, shuffle=False,
                              num_workers=8, pin_memory=True)

#log_file = os.path.join(files_path, "models.log")
#if not os.path.isfile(log_file):
#  print("log_file ({}) not exists!".format(log_file))
#else:
mean_auc = []
with open(output_file, 'a') as writer:
    print(">> Disease = {}".format(diseases))
    writer.write("Disease = {}\n".format(diseases))
    
    y_test, p_test = test_classification(model, data_loader_test, device)
    '''
    if test_diseases is not None:
      y_test = copy.deepcopy(y_test[:,test_diseases])
      p_test = copy.deepcopy(p_test[:, test_diseases])
      individual_results = metric_AUROC(y_test, p_test)
    else:
    '''
    individual_results = metric_AUROC(y_test, p_test)
    print(">>AUC = {}".format(np.array2string(np.array(individual_results), precision=4, separator=',')))
    writer.write(
      "AUC = {}\n".format(np.array2string(np.array(individual_results), precision=4, separator='\t')))
    
    mean_over_all_classes = np.array(individual_results).mean()
    print(">> Mean AUC = {:.4f}".format(mean_over_all_classes))
    writer.write("Mean AUC = {:.4f}\n".format( mean_over_all_classes))
    
    
    









    


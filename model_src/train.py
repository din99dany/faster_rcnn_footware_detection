import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils
import transforms as T
from engine import train_one_epoch, evaluate
from torch.utils import data
import argparse
import os
from azureml.core import Dataset

################ CONSTANTS ################
###########################################
IMG_MAX_DIM     = 1024
MIN_BOX_DIM     = 20
NUM_CLASSES     = 2
LABEL           = '/m/09j5n'
NUM_EPOCHS      = 20

world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])

distributed = world_size > 1

print(world_size)
print(rank)
print(local_rank)
print(distributed)

################ PARSE ARGS ###############
###########################################
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str)
args = parser.parse_args()
print("PARSE-DONE")

# set device
if distributed:
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialize distributed process group using default env:// method
if distributed:
    torch.distributed.init_process_group(backend="nccl")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES)
model.to(device)

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(dtype=torch.float32))
    #if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        #transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class DatasetCNNR(data.Dataset):
  def __init__(self, list_IDs, labels, location = '', transform=None, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    self.labels = labels
    self.list_IDs = list_IDs
    self.transform = transform
    self.location = args.data_dir
    self.mean = mean
    self.std = std

  def __len__(self):
    return len(self.list_IDs)
  
  def __getitem__(self, index):
    'Generates one sample of data'
    
    ID          = self.list_IDs[index]
    num_objs    = len(self.labels[ID])
    img_path    = self.location + '/' + ID + '.jpg'
    img         = Image.open(img_path).convert("RGB")
    
    boxes = []
    w, h = img.size
    for ground_truth in self.labels[ID]:
        x_min = int(w * ground_truth[0])
        x_max = int(w * ground_truth[1])
        y_min = int(h * ground_truth[2])
        y_max = int(h * ground_truth[3])
        if(((x_max - x_min) >= MIN_BOX_DIM) and ((y_max - y_min) >= MIN_BOX_DIM)):
            boxes.append([x_min, y_min, x_max, y_max])
        
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
    target = {}
    target["boxes"]     = boxes
    target["labels"]    = torch.ones((num_objs,), dtype=torch.int64)
    target["image_id"]  = torch.tensor([index])
    target["area"]      = (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0]) if boxes.numel() else torch.zeros((1,), dtype=torch.float32)
    target["iscrowd"]   = torch.zeros((num_objs,), dtype=torch.int64)
        
    if self.transform is not None:
        img, target = self.transform(img, target)
    
    return img, target

def isImageValid(row): 
    w, h = Image.open(args.data_dir + '/' + row[0] + ".jpg").size
    x_min = int(w * row[1])
    x_max = int(w * row[2])
    y_min = int(h * row[3])
    y_max = int(h * row[4])

    if((w > IMG_MAX_DIM) or (h > IMG_MAX_DIM)):
        return False

    if(((x_max - x_min) >= MIN_BOX_DIM) and ((y_max - y_min) >= MIN_BOX_DIM)):
        return True

    return False

def extract_data_label(d):
  label_dict = {}  
  idx_list = d.filter(items=['ImageID','XMin','XMax','YMin','YMax']).to_numpy()

  for id in idx_list:
    if ( isImageValid(id) ):
        if id[0] not in label_dict:
            label_dict[id[0]] = [id[1:].tolist()]
        else:
            label_dict[id[0]].append(id[1:].tolist())

  return label_dict

print("START-DATA-CONSTRUCTION")
unsplit_train_label_dict = extract_data_label( pd.read_csv(args.data_dir + '/train_set_train_annotations-bbox.csv'))
list_id_train = list(unsplit_train_label_dict.keys())

# Create the dataset
training_set = DatasetCNNR(list_id_train, unsplit_train_label_dict, transform = get_transform(train=True))

# Wrap the training set in a distribuited sampler to work with the distribuited model
if distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_set)
else:
    train_sampler = None

training_generator = torch.utils.data.DataLoader(
    training_set,
    batch_size = 2,
    shuffle = (train_sampler is None),
    num_workers = 2,
    sampler = train_sampler,
    collate_fn = utils.collate_fn,
    drop_last = True
)

# Wrap the model in a distribuited model wrapper
print("START-TRAIN")
if distributed:
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

# construct an optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0005)

# construct scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Train the model
for epoch in range(NUM_EPOCHS):
    if distributed:
        train_sampler.set_epoch(epoch)

    train_one_epoch(model, optimizer, training_generator, device, epoch, print_freq=30)
    lr_scheduler.step()


if not distributed or rank == 0:
    torch.save(model.module, "./outputs/model_chad.pt")

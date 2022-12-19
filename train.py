
import os
import numpy as np 
import pandas as pd
import torch
import torchvision
import super_gradients
from pathlib import Path, PurePath

from PIL import Image
import pprint
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

from super_gradients import init_trainer, Trainer
from super_gradients.common import MultiGPUMode
from super_gradients.training.utils.distributed_training_utils import setup_gpu_mode
from super_gradients.training import Trainer
from super_gradients.training import training_hyperparams

from super_gradients.training import models
from super_gradients import Trainer

import onnx

class config:
    EXPERIMENT_NAME = 'kitchenware_classification'
    MODEL_NAME = 'vit_large'
    CHECKPOINT_DIR = 'checkpoints'
    WEIGHTS = "imagenet"
    TRAINING_PARAMS = "training_hyperparams/imagenet_vit_train_params"
    NUM_CLASSES = 6
    BATCH_SIZE = 16

    # specify the paths to training and validation set 
    IMAGE_PATH = './data/images'
    TRAIN_DATA = './data/train.csv'
    TEST_DATA = './data/test.csv'

    

    # set the input height and width
    INPUT_HEIGHT = 224
    INPUT_WIDTH = 224

    # set the input height and width
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    NUM_WORKERS = os.cpu_count()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'




# Instantiate, SG Trainer, model, and training params

trainer = Trainer(experiment_name=config.EXPERIMENT_NAME, ckpt_root_dir=config.CHECKPOINT_DIR)
model = models.get(model_name=config.MODEL_NAME, num_classes=config.NUM_CLASSES, pretrained_weights=config.WEIGHTS)
training_params =  training_hyperparams.get(config.TRAINING_PARAMS)


training_params["max_epochs"] = 5
training_params["zero_weight_decay_on_bias_and_bn"] = True
training_params['train_metrics_list'] = ['Accuracy']
training_params['valid_metrics_list'] = ['Accuracy']
training_params['ema'] = True
training_params["criterion_params"] = {'smooth_eps': 0.1} 
training_params['average_best_models'] = True
training_params["sg_logger_params"]["launch_tensorboard"] = False

# Split training data into train and validation sets

def add_image_col(df):
    df['image'] = df['Id'].apply(lambda x: x +'.jpg')
    
# read labels into pandas df all cols as string
labels_df = pd.read_csv(config.TRAIN_DATA, dtype='str')
test_df = pd.read_csv(config.TEST_DATA, dtype='str')

# create col (xxxx.jpg), the image filename
add_image_col(labels_df)
add_image_col(test_df)

# map labels to integer
le = LabelEncoder()
labels_df['targets'] = le.fit_transform(labels_df['label'])

#split into train and validation sets
train_df, val_df = train_test_split(labels_df,  stratify= labels_df['targets'], test_size=.10, shuffle=True, random_state=42)




class KitchwareDataset(Dataset):
    def __init__(self, dataframe , img_dir, split, transform = None):
        self.img_labels = dataframe 
        self.img_dir = img_dir
        self.split = split
        self.transform = transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self , idx):
        if self.split in ['train', 'val']:
            img_path = os.path.join(self.img_dir , self.img_labels.iloc[idx, 2])
            label = self.img_labels.iloc[idx, 3]
        else:
            img_path = os.path.join(self.img_dir , self.img_labels.iloc[idx, 1])
            
        original_image = Image.open(img_path)
        image = np.array(original_image)
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        if self.split in ['train', 'val']: 
            return image, label 
        else:
            return image

# Transformations

# initialize our data augmentation functions
make_tensor = ToTensorV2()
normalize = A.Normalize(mean=config.IMAGENET_MEAN, 
                        std=config.IMAGENET_STD)
resize = A.Resize(height=config.INPUT_HEIGHT,
                  width=config.INPUT_WIDTH)
horizontal_flip = A.HorizontalFlip(p=0.50)
flip = A.Flip(p=0.50)
random_ninety = A.RandomRotate90()
random_crop = A.RandomCrop(height=config.INPUT_HEIGHT,
                           width=config.INPUT_WIDTH,
                           p=0.75)
hue_saturation = A.HueSaturationValue(p=.5)
iso_noise = A.ISONoise(p=.5)
color_jitter = A.ColorJitter(p=.5)
emboss = A.Emboss(p=.5)
channel_shuffle = A.ChannelShuffle(p=.5)
randomly_choose_one = A.OneOf([flip, 
                               random_ninety, 
                               iso_noise,
                               color_jitter,
                               emboss,
                               hue_saturation,
                               channel_shuffle], p=.50)


# initialize our training and validation set data augmentation pipeline
train_transforms = A.Compose([
  resize, 
  horizontal_flip, 
  random_crop,
  randomly_choose_one,
  normalize,
  make_tensor
])

val_transforms = A.Compose([resize, normalize, make_tensor])

train_data = KitchwareDataset(train_df , config.IMAGE_PATH , 'train', transform = train_transforms)
val_data = KitchwareDataset(val_df , config.IMAGE_PATH , 'val', transform = val_transforms)
test_data = KitchwareDataset(test_df, config.IMAGE_PATH, 'test',transform = val_transforms)

train_dataloader = DataLoader(train_data, batch_size = config.BATCH_SIZE , shuffle = True)
val_dataloader = DataLoader(val_data, batch_size = config.BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = config.BATCH_SIZE, shuffle = False)



# Training model

trainer.train(model=model, 
              training_params=training_params, 
              train_loader=train_dataloader,
              valid_loader=val_dataloader)

# Load the best model that we trained

best_model = models.get(config.MODEL_NAME,
                        num_classes=config.NUM_CLASSES,
                        checkpoint_path=os.path.join(trainer.checkpoints_dir_path, "ckpt_best.pth"))

# Export model as onnx model

best_model.eval()
best_model.prep_model_for_conversion(input_size=[1, 3, 224, 224])
dummy_input = torch.randn([1, 3, 224, 224], device=next(best_model.parameters()).device)

torch.onnx.export(
    best_model, 
    dummy_input, 
    "kitchenware_model.onnx",
    export_params=True,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
)

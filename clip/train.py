import re, random
from pathlib import Path
from typing import List
import pickle
import json
from pprint import pprint
from icecream import ic
import os, sys, shutil
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils import data
import clip
from PIL import Image

# Latest Update : 22 June 2021, 09:55 GMT+7

# TO ADD :
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used


BATCH_SIZE = 32
EPOCH = 10


#BATCH_SIZE must larger than 1
train_dataloader = data.DataLoader(...,batch_size = BATCH_SIZE) #Define your own dataloader

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32", device=device, jit=False) #Must set jit=False for training
if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

for epoch in range(EPOCH):
  for batch in train_dataloader :
      optimizer.zero_grad()

      list_image, list_txt = batch #list_images is list of image in numpy array(np.uint8), or list of PIL images
    
      images= torch.stack([preprocess(Image.fromarray(img)) for img in list_image],dim=0).to(device) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
      texts = clip.tokenize(list_txt).to(device)
    
      logits_per_image, logits_per_text = model(images, texts)

      ground_truth = torch.arange(BATCH_SIZE,dtype=torch.long,device=device)

      total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
      total_loss.backward()
      if device == "cpu":
         optimizer.step()
      else : 
        convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model)

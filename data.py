import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import utils

# numpy shape change
def load_img(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image,(64, 64))

    image = np.transpose(image.astype(np.float32),
                          (2, 0, 1))

    return image


# image_dir = dir of all images
# train_split_file = parts of dir"s text file
class DogCatDataset(Dataset):
    def __init__(self, image_dir, train_split_file):
        self.images = [x for x in utils.find_all_files_over(image_dir,train_split_file)]
        self.labels = [x for x in utils.find_all_files_label(image_dir,train_split_file)]
        # for x in utils.find_all_files_over(image_dir,train_split_file):
        #     if "cat" in x:
        #         self.labels.append(1)
        #     elif "dog" in x:
        #         self.labels.append(0)
        #         print("dog")         

    def __getitem__(self, index):
        image = load_img(self.images[index])
        # label = np.array([0,0])
        # label[self.labels[index]] = 1
        label = np.array(self.labels[index])
        

        
        return image, label

    def __len__(self):
        return len(self.images)

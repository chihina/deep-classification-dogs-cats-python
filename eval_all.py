import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
import copy
import sys
import numpy as np
import torch.nn.functional as F

from data import DogCatDataset
import utils

import matplotlib.pyplot as plt

best_accuracy = 0
path_list = os.listdir("../dogs-vs-cats/model/resnet/")

def decide_path(path):

    # モデル保存領域のpath
    PATH = "../dogs-vs-cats/model/resnet/" + path

    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(PATH))

    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
        # print('cuda is available!')

    else:
        print('cuda is not available')

    # make dataset                                  
    eval_set = DogCatDataset("../dogs-vs-cats/datasets", train_split_file="../dogs-vs-cats/image_dir_path_eval.txt")

    # remake datasets
    evaluation_data_loader = DataLoader(dataset=eval_set,
                                        batch_size=128,
                                        # shuffle=True
                                        shuffle=False)


    epoch_loss = 0

    model.train(False)

    # for result dictionary
    result_dic = {}
    result_dic["dog_true"] = 0
    result_dic["cat_true"] = 0
    result_dic["dog_false"] = 0
    result_dic["cat_false"] = 0

    data_length = len(evaluation_data_loader)
    

    for iteration, batch in enumerate(evaluation_data_loader,1):
         
        image = batch[0]
        
        # is this correct?
        label = batch[1]

        minibatch = image.size()[0]
        for j in range(minibatch):
            image[j] = utils.norm(image[j]/255.0, vgg=True)
        

        if cuda:
            image = Variable(image).cuda()
            label = Variable(label).cuda()
        # print("{}/{}".format(iteration, data_length))
        pred = model(image)
        pred_labels_list = torch.argmax(pred, dim=1)
        for pred_label, correct_label in zip(pred_labels_list, label):
            if pred_label == correct_label:
                if correct_label == 0:
                    result_dic["dog_true"] += 1
                elif correct_label == 1:
                    result_dic["cat_true"] += 1
            else:
                if correct_label == 0:
                    result_dic["dog_false"] += 1
                elif correct_label == 1:
                    result_dic["cat_false"] += 1

    data_length = len(evaluation_data_loader)
    data_sum = 0
    for value in result_dic.values():
        data_sum += int(value)
    
    accuracy = (result_dic["dog_true"] + result_dic["cat_true"]) / data_sum * 100
    return accuracy
    # print("test {} sets".format(data_length))
    # print("accuracy: {}%".format(accuracy))
    # print("dog_true:{}, cat_true:{}, dog_false:{}, cat_false:{}".format(result_dic["dog_true"], result_dic["cat_true"], result_dic["dog_false"], result_dic["cat_false"]))

best_accuracy_dic = {}   
for i,path in enumerate(path_list):
    print("{}/{}".format(i+1, len(path_list)))
    accuracy = decide_path(path)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_accuracy_dic[path] = best_accuracy
        print("best accuracy: {} {}".format(accuracy, path))

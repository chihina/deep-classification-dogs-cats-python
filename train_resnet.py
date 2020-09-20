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


# for test
all_epoch = 300

# loss確認リスト定義
loss_list_idx = []
loss_list_num = []


# save weight
def checkpoint(epoch, pretrained_flag=False):
    model_out_path = "../dogs-vs-cats/model/resnet"+ "/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)

    print("Checkpoint saved to {}".format(opt.save_folder))

# save weight
def best_checkpoint(epoch, pretrained_flag=False):
    
    model_out_path = "../dogs-vs-cats/model/resnet"+ "/best_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)

    print("Best Checkpoint saved to {}".format("../dogs-vs-cats/model"))

# train method 
def train(epoch):
    epoch_loss = 0
    data_length = len(training_data_loader)

    for iteration, batch in enumerate(training_data_loader,1):
         
        image = batch[0]
        
        # is this correct?
        label = batch[1]

        # print(len(batch[0]))
        optimizer.zero_grad()
        
        minibatch = image.size()[0]
        for j in range(minibatch):
            image[j] = utils.norm(image[j]/255.0, vgg=True)
        

        if cuda:
            image = Variable(image).cuda()
            label = Variable(label).cuda()

        # for result dictionary
        result_dic = {}
        result_dic["dog_true"] = 0
        result_dic["cat_true"] = 0
        result_dic["dog_false"] = 0
        result_dic["cat_false"] = 0

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

        data_sum = 0
        for value in result_dic.values():
            data_sum += int(value)

        accuracy = (result_dic["dog_true"] + result_dic["cat_true"]) / data_sum * 100

        loss = loss_fn(pred, label)
        # if iteration % 100 == 0:
        # print("train pred : {}".format(pred))
        # print("label : {}".format(label))
        print("train loss : {}".format(loss))
    
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        print("Epoch: [{}/{}] [{}/{}]  train_loss: {}"
              .format(epoch, all_epoch, iteration,
                      data_length, loss))

    average_loss = epoch_loss / data_length
    
    return  average_loss, accuracy             
                     
                      

    print("===> Epoch {} Complete: Avg train_Loss: {}"
          .format(epoch, epoch_loss / data_length))

def validation(epoch):
    epoch_loss = 0
    epoch_IoU_loss = 0
    data_length = len(validation_data_loader)
    model.eval()
    
    
    for iteration, batch in enumerate(validation_data_loader,1):
        
        image = batch[0]

         # is this correct?
        label = batch[1]

        with torch.no_grad():
            optimizer.zero_grad()
 
            minibatch = image.size()[0]
            for j in range(minibatch):
                image[j] = utils.norm(image[j]/255.0, vgg=True)
                
            
            if cuda:
                image = Variable(image).cuda()
                label = Variable(label).cuda()
                
            # print(overlay.size())         
            pred = model(image)
            
            loss = loss_fn(pred, label)
            
            if iteration % 100 == 0:
                # print("image : {}".format(image))
                # print("image id : {}".format(id(image)))
                # print("val pred : {}".format(pred))
                # print("pred id: {}".format(id(pred)))
                # print("label : {}".format(label))
                print("val loss : {}".format(loss))
            
            optimizer.step()
            
            epoch_loss += loss.item()
            # epoch_IoU_loss += IoU_loss

            print("Epoch: [{}/{}] [{}/{}]  val_loss: {}"
                .format(epoch, all_epoch, iteration,
                        data_length, loss))
            
            

    average_loss = epoch_loss / data_length
    loss_list_idx.append(epoch)
    loss_list_num.append(average_loss)

    # average_IoU_loss = epoch_IoU_loss / data_length
    
    print("===> Epoch {} Complete: Avg val_Loss: {}"
          .format(epoch, epoch_loss / data_length))

    
    return average_loss



print("===> Loading datasets")
# make datasets
train_set = DogCatDataset("../dogs-vs-cats/datasets", train_split_file="../dogs-vs-cats/image_dir_path_train.txt")

# remake datasets
training_data_loader = DataLoader(dataset=train_set,
                                  batch_size=256,
                                  shuffle=True)


# make dataset                                  
val_set = DogCatDataset("../dogs-vs-cats/datasets", train_split_file="../dogs-vs-cats/image_dir_path_val.txt")

# remake datasets
validation_data_loader = DataLoader(dataset=val_set,
                                  batch_size=256,
                                #   shuffle=True
                                  shuffle=False)

print("===> Building model")


# モデル保存領域のpath
PATH = "../dogs-vs-cats/model/resnet/best_epoch_292.pth" 

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load(PATH))

# model = models.resnet18(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, 2)

print('{} Train samples found'.format(len(train_set)))
print('{} Test samples found'.format(len(val_set)))
torch.manual_seed(1)

# model = CNN()

cuda = torch.cuda.is_available()
if cuda:
    model.cuda()
    print('cuda is available!')

else:
    print('cuda is not available')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.SGD(model.parameters(), lr=1.0873867414847836e-05,
#                      momentum=0.9, weight_decay=6.262828765291544e-06)

optimizer = optim.SGD(model.parameters(), lr=1.0873867414847836e-05,
                     momentum=0.9, weight_decay=6.262828765291544e-06)
# Training
best_loss = 100000.0

for epoch in range(1 + 580, all_epoch +1 + 580):
    model.train(True)
    current_train_loss, accuracy = train(epoch)
    model.train(False)
    current_val_loss = validation(epoch)

    # loss書き出し
    with open("../dogs-vs-cats/result_loss/resnet/train_data.txt", mode='a') as f:
        f.write("train_loss epoch: " + str(epoch) + " : " + str(current_train_loss) + "\n")

    with open("../dogs-vs-cats/result_loss/resnet/accuracy_data.txt", mode='a') as f:
        f.write("accuracy epoch: " + str(epoch) + " : " + str(accuracy) + "\n")        

    with open("../dogs-vs-cats/result_loss/resnet/val_data.txt", mode='a') as f:
        f.write("val_loss epoch: " + str(epoch) + " : " + str(current_val_loss) + "\n")

    if current_val_loss < best_loss:
        best_loss = current_val_loss
        best_checkpoint(epoch)
        print("epoch : {} Best Loss : {}".format(epoch,best_loss))
    

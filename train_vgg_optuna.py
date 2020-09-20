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
import optuna
# optuna.logging.disable_default_handler()
import time

# start time
start_time = time.time()

# for define all epoch num
all_epoch = 5

# loss確認リスト定義
loss_list_idx = []
loss_list_num = []

# select pretrained model
pretrained_model_name = "vgg"

# save weight
def checkpoint(epoch, pretrained_flag=False):
    model_out_path = "../dogs-vs-cats/model/"+ pretrained_model_name + "/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)

    print("Checkpoint saved to {}".format(opt.save_folder))

# save weight
def best_checkpoint(epoch, pretrained_flag=False):
    
    model_out_path = "../dogs-vs-cats/model/resnet"+ pretrained_model_name + "/best_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)

    print("Best Checkpoint saved to {}".format("../dogs-vs-cats/model"))

# train method 
def train(epoch, optimizer):
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

        pred = model(image)
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
                     
                      

    print("===> Epoch {} Complete: Avg train_Loss: {}"
          .format(epoch, epoch_loss / data_length))

def validation(epoch, optimizer):
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
                                  batch_size=128,
                                  shuffle=True)


# make dataset                                  
val_set = DogCatDataset("../dogs-vs-cats/datasets", train_split_file="../dogs-vs-cats/image_dir_path_val.txt")

# remake datasets
validation_data_loader = DataLoader(dataset=val_set,
                                  batch_size=128,
                                #   shuffle=True
                                  shuffle=False)

print("===> Building model")

model = models.vgg16(pretrained=True)
print(model)

model.classifier[6] = nn.Linear(4096,2)
print(model)

print('{} Train samples found'.format(len(train_set)))
print('{} Test samples found'.format(len(val_set)))
torch.manual_seed(1)

cuda = torch.cuda.is_available()
if cuda:
    model.cuda()
    print('cuda is available!')

else:
    print('cuda is not available')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define loss function
loss_fn = nn.CrossEntropyLoss()

# Training
best_loss = 100000.0

def adam(model, trial, weight_decay):
    adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
    return optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)

def momentum(model, trial, weight_decay):
    momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
    return optim.SGD(model.parameters(), lr=momentum_sgd_lr,
                     momentum=0.9, weight_decay=weight_decay)

def get_optimizer(trial, model):
    optimizer = trial.suggest_categorical('optimizer', [adam, momentum])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    return optimizer(model, trial, weight_decay)

# optuna
def objective(trial):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    optimizer = get_optimizer(trial, model)
    global best_loss

    for epoch in range(1, all_epoch+1):
        model.train(True)
        train(epoch, optimizer)
        model.train(False)
        current_val_loss = validation(epoch, optimizer)
        # loss書き出し
        # with open("../dogs-vs-cats/result_loss/resnet/test.txt", mode='a') as f:
        #     f.write("val_loss epoch: " + str(epoch) + " : " + str(current_val_loss) + "\n")
        if current_val_loss < best_loss:
            best_loss = current_val_loss
            # best_checkpoint(epoch)
            print("epoch : {} Best Loss : {}".format(epoch,best_loss))
    return current_val_loss    

# optuna exe
study = optuna.create_study()
study.optimize(objective, n_trials=10) 
with open("../dogs-vs-cats/result_loss/" + pretrained_model_name +  "/best_prams.txt", mode='w') as f:
    f.write(str(study.best_params) + "\n") 
    f.write("best loss score" + str(study.best_value) + "\n")      

# finish time
finish_time = time.time()

# write cost time
with open("../dogs-vs-cats/result_loss/" + pretrained_model_name + "/time.txt", mode='w') as f:
    f.write(str(finish_time - start_time) + "s takes")

# # 折れ線グラフを出力
# left = np.array(loss_list_idx)
# height = np.array(loss_list_num)
# plt.plot(left, height/3, linestyle="dashdot")
# plt.show()
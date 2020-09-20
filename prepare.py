# coding: UTF-8


import torch
from torch import nn
import os, cv2, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.autograd import Variable

# for figure
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

import time

# start time
start_time = time.time()

#データセットのディレクトリ
TRAIN_DIR = "./train/train/"

# データセットのサイズ
ROWS = 64
COLS = 64
CHANNELS = 3

# 犬のトレーニングデータセット
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

# 猫のトレーニングセット
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

# 犬、猫1000枚ずつに分割
train_images = train_dogs[:1000] + train_cats[:1000]

# データの順番をシャッフルする
# (しないと、犬の画像1000枚連続で学習されてしまう)
# random.shuffle(train_images)

# 画像読み込み
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
  
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

# データ作成
def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)
    for i, image_file in enumerate(images):
        image = read_image(image_file)

        # 犬の場合
        if i < 1000:
            label = 0
            data[i] = (torch.from_numpy(image.T), label)
            print(data[i])
        # 猫の場合    
        else:
            label = 1
            data[i] = (image.T, label)

        if i%250 == 0: print('Processed {} of {}'.format(i, count))
        if i == 5:
            break
    
    return data

# trainデータの前処理
train = prep_data(train_images)

# trainデータの形状確認
print("Train shape: {}".format(train.shape))

# データの確認
labels = []
for i in train_images:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)


# データの確認(個数)
zero_num = labels.count(0)
one_num = labels.count(1)
x = [1,2]
y = []
labels = ["dog","cat"]
y.append(zero_num)
y.append(one_num)
plt.bar(x,y,width=0.5, color='#0096c8', edgecolor='b', linewidth=2, tick_label= labels, align="center")
# plt.show()

# 画像データ確認(可視化)
def show_cats_and_dogs(idx):
    cat = read_image(train_cats[idx])
    dog = read_image(train_dogs[idx])
    pair = np.concatenate((cat, dog), axis=1)
    plt.figure(figsize=(10,5))
    plt.imshow(pair)
    plt.show()
# show_cats_and_dogs(0)    

# トレーニング、テスト用にデータを分割
train_data, test_data = train_test_split(train, test_size=0.2)

# バッチサイズを決定し、データを整形
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=True)

# CNNネットワーク
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 100)
        self.fc2 = nn.Linear(100, 34)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# modelのインスタンスを生成
model = CNN()

# optimizerのパラメータ設定
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 誤差の関数の設定
criterion = nn.CrossEntropyLoss()

# trainのメソッド
def train(epoch):
    model.train()
    for batch_idx, (image,label) in enumerate(train_loader):
        image, label = Variable(image), Variable(label)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(image), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()

    for (image, label) in test_loader:
        image, label = Variable(image.float(), volatile=True), Variable(label)
        output = model(image)
        test_loss += criterion(output, label).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#学習
for epoch in range(1, 5):
    train(epoch)
    test()

finish_time = time.time()
print("it takes {} minutes".format(finish_time-start_time))
import numpy as np
import matplotlib.pyplot as plt


print("===> draw figure........")
with open("../dogs-vs-cats/result_loss/resnet/train_data.txt", mode='r') as f:
    train_lines = f.readlines()

with open("../dogs-vs-cats/result_loss/resnet/val_data.txt", mode='r') as f:
    val_lines = f.readlines()

loss_list_idx_train = []
loss_list_num_train = []

loss_list_idx_val = []
loss_list_num_val = []

for line in train_lines:
    elements = line.strip().split(":")
    loss_list_idx_train.append(int(elements[1]))
    loss_list_num_train.append(float(elements[2]))

for line in val_lines:
    elements = line.strip().split(":")
    loss_list_idx_val.append(int(elements[1]))
    loss_list_num_val.append(float(elements[2]))

# 折れ線グラフを出力
left_train = np.array(loss_list_idx_train)
height_train = np.array(loss_list_num_train)
plt.plot(left_train, height_train, linestyle="dashdot", color="red", label="train")

left_val = np.array(loss_list_idx_val)
height_val = np.array(loss_list_num_val)
plt.plot(left_val, height_val, linestyle="dashdot", color="blue", label="validation")

plt.title("Result of training")
plt.xlabel('Epoch num')
plt.ylabel('Loss')
plt.legend()
plt.show()
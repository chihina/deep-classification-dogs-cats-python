import numpy as np
import matplotlib.pyplot as plt


print("===> draw figure........")
with open("../dogs-vs-cats/result_loss/resnet/accuracy_data.txt", mode='r') as f:
    lines = f.readlines()

loss_list_idx = []
loss_list_num = []

for line in lines:
    elements = line.strip().split(":")
    loss_list_idx.append(int(elements[1]))
    loss_list_num.append(float(elements[2]))
 
# 折れ線グラフを出力
left = np.array(loss_list_idx)
height = np.array(loss_list_num)
plt.plot(left, height, linestyle="dashdot")
plt.title("Result of training")
plt.xlabel('Epoch num')
plt.ylabel('Accuracy')
plt.show()
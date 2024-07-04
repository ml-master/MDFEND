import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
# 读取存储为txt文件的数据
def data_read(dir_path):
    loss_list=[]
    with open(dir_path, "r",encoding='utf-8') as f:
        data = json.load(f)
        print(data)
        for epoch in data:
            loss_list.append(epoch['loss'])
    f.close()
    print(loss_list)
    return loss_list
    # return np.asfarray(data, float)


if __name__ == "__main__":
    train_loss_path = "train_loss_and_val.json"  # 存储文件路径

    y_train_loss = data_read(train_loss_path)  # loss值，即y轴
    x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')  # x轴标签
    plt.ylabel('loss')  # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss",color='red')
    plt.legend()
    plt.title('Loss curve in new data')
    plt.show()
    plt.savefig("loss.png")

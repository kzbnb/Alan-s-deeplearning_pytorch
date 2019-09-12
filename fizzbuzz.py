# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]

#定义正确的结果函数
def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0

#由0123可输出正确单词
def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

#导入
import numpy as np
import torch

#二进制数的位数
NUM_DIGITS = 10

#十进制数转换为二进制，矩阵每一行为一个二进制数
def binary_encode(i, num_digits):

    return np.array([i >> d & 1 for d in range(num_digits)])

#读取数据，做训练集
trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])


#隐藏层为100
NUM_HID=100

#定义训练模型
#线性&ReLU
model=torch.nn.Sequential(

    #输入单元
    torch.nn.Linear(10,NUM_HID),

    #中间的隐藏单元
    #激活函数
    torch.nn.ReLU(),

    #输出单元
    torch.nn.Linear(NUM_HID,4)
)

#代价函数
loss_fn=torch.nn.CrossEntropyLoss()

#定义优化器，优化全体parameters,lr是学习率
optimizer=torch.optim.SGD(model.parameters(),lr=0.05)

#batch,大小为128
BATCH_SIZE=128

#开始训练

#
for epoch in range(10000):

    #从0到trX的长度，其中步长为batch_size
    for start in range(0,len(trX),BATCH_SIZE):

        #取一个batch的数据
        end=start+BATCH_SIZE
        batchX=trX[start:end]
        batchY=trY[start:end]

        #正向传播，得到预测
        y_pred=model(batchX)

        #得到代价
        loss=loss_fn(y_pred,batchY)

        #梯度清零
        optimizer.zero_grad()

        #反向传播
        loss.backward()

        #更新模型
        optimizer.step()

        #获取代价函数的数值
        loss=loss_fn(model(trX),trY).item()

        #输出
        print('Epoch:', epoch, 'Loss:', loss)




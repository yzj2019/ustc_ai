import torch
import math
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

random.seed()

class my_npfunc:
    '''自定义接受numpy向量的函数和函数导数，返回numpy向量'''
    def softmax(x):
        '''softmax函数，输入x为np向量，即shape形式为(i,)'''
        res = np.zeros([x.shape[0],])
        for i in range(0, x.shape[0]):
            res[i] = math.exp(x[i])
        res = res / res.sum()
        return res
    
    def softmax_deriv(x):
        '''softmax函数输出向量对输入向量求偏导的Jacobi矩阵'''
        y = my_npfunc.softmax(x)
        res = np.zeros([x.shape[0], x.shape[0]])
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[0]):
                if i==j:
                    res[i][j] = y[i] - y[i]*y[i]
                else:
                    res[i][j] = - y[i]*y[j]
        return res
        
    def sigmoid(x):
        '''sigmoid函数'''
        res = np.zeros([x.shape[0],])
        for i in range(0, x.shape[0]):
            res[i] = 1/(1+math.exp(-x[i]))
        return res

    def sigmoid_deriv(x):
        '''sigmoid函数输出向量对输入向量求偏导的Jacobi矩阵'''
        y = my_npfunc.sigmoid(x)
        res = np.eye(x.shape[0])
        for i in range(0, x.shape[0]):
            res[i][i] = y[i] - y[i]*y[i]
        return res


class MLP(object):
    '''自定义多层感知机类'''
    def __init__(self, lossfunc, lr, epochs):
        '''初始化'''
        self.lossfunc = lossfunc    # 损失函数名称
        self.lr = lr                # 学习率
        self.epochs = epochs        # 迭代次数
        self.layers = []            # 存储各层网络
        self.loss = []              # 存储每次训练的loss变化过程

    def add_layer(self, innum, outnum, func, weight=None, bias=None):
        '''添加一层全连接层'''
        layer = {}
        layer['type'] = 'all_connect'
        # 初始化为均值为0，标准差为输入数量的平方根倒数的正态分布的随机采样
        if type(weight)==np.ndarray:
            layer['weight'] = weight.reshape(outnum, innum)
        else:
            layer['weight'] = np.random.normal(loc=0.0, scale=pow(innum, -0.5), size=(outnum, innum))
        if type(bias)==np.ndarray:
            layer['bias'] = bias.reshape(outnum, 1)
        else:
            layer['bias'] = np.zeros([outnum, 1])
        layer['func'] = func
        self.layers.append(layer)

    def loss_func(self, pred, label):
        '''损失函数，接受np向量pred(预测值)、label(训练值)，返回loss标量值'''
        if self.lossfunc=='crossentropy':
            # 此处pred为各个类别的概率值，label为真实类别的onehot编码
            return -1 * np.multiply(np.log(pred), label).sum()

    def loss_func_deriv(self, pred, label):
        '''损失函数对输入向量的偏导，接受np向量pred(预测值)、label(训练值)，返回np向量'''
        if self.lossfunc=='crossentropy':
            # 此处pred为各个类别的概率值，label为真实类别的onehot编码
            return -1 * np.multiply(np.reciprocal(pred.astype(float)), label)

    def forward(self, input_data):
        '''值的正向传播过程，input_data为np向量'''
        layer_num = len(self.layers)
        self.layers[0]['input'] = input_data.reshape([input_data.shape[0], 1])
        for i in range(0, layer_num):
            # 逐层正向传播，当前层的input已经设置好了，是np矩阵形式而不是np向量形式
            temp = np.dot(self.layers[i]['weight'], self.layers[i]['input']) + self.layers[i]['bias']
            self.layers[i]['u'] = temp
            if self.layers[i]['func']=='sigmoid':
                # 注意传入激活函数时，应reshape为向量
                temp = my_npfunc.sigmoid(temp.reshape(temp.shape[0],))
                self.layers[i]['output'] = temp.reshape(temp.shape[0], 1)
            elif self.layers[i]['func']=='softmax':
                temp = my_npfunc.softmax(temp.reshape(temp.shape[0],))
                self.layers[i]['output'] = temp.reshape(temp.shape[0], 1)
            if i<layer_num-1:
                # 下一层的输入为上一层的输出
                self.layers[i+1]['input'] = self.layers[i]['output']
        return self.layers[layer_num-1]['output']

    def backward(self, label):
        '''做误差的反向传播，计算各层的delta，传入的label(训练值)为np向量'''
        layer_num = len(self.layers)
        # 输出层delta
        pred = self.layers[layer_num-1]['output']
        pred = pred.reshape(pred.shape[0],)         # 转为向量
        self.loss.append(self.loss_func(pred, label))
        temp1 = self.loss_func_deriv(pred, label)   # part{L}/part{y}
        temp1 = temp1.reshape(1, temp1.shape[0])    # 转为行向量
        temp2 = self.layers[layer_num-1]['u']       # 线性映射后的中间变量，在forward时存了避免重复计算
        if self.layers[layer_num-1]['func']=='sigmoid':
            # 注意传入激活函数时，应reshape为向量
            temp2 = my_npfunc.sigmoid_deriv(temp2.reshape(temp2.shape[0],))
            self.layers[layer_num-1]['delta'] = np.dot(temp1, temp2)
        elif self.layers[layer_num-1]['func']=='softmax':
            temp2 = my_npfunc.softmax_deriv(temp2.reshape(temp2.shape[0],))
            self.layers[layer_num-1]['delta'] = np.dot(temp1, temp2)
        for i in range(layer_num-2, -1, -1):
            # 逐层反向传播delta
            # temp1 = delta_{l+1} * weight_{l+1}, temp2 = f^'(u_l)
            temp1 = self.layers[i+1]['delta']
            temp1 = np.dot(temp1, self.layers[i+1]['weight'])
            temp2 = self.layers[i]['u']
            if self.layers[i]['func']=='sigmoid':
                # 注意传入激活函数时，应reshape为向量
                temp2 = my_npfunc.sigmoid_deriv(temp2.reshape(temp2.shape[0],))
                self.layers[i]['delta'] = np.dot(temp1, temp2)
            elif self.layers[i]['func']=='softmax':
                temp2 = my_npfunc.softmax_deriv(temp2.reshape(temp2.shape[0],))
                self.layers[i]['delta'] = np.dot(temp1, temp2)

    def train(self, train_data, train_label):
        '''MLP训练函数，输入train_data.shape=(num, attr_num)，train_label.shape=(num, 1)'''
        layer_num = len(self.layers)
        data_num = train_data.shape[0]
        # 对train_label预处理，成onehot编码矩阵(label_type_num, num)
        label_type_num = np.unique(train_label).shape[0]
        labels = np.zeros([data_num, label_type_num])
        for i in range(0, data_num):
            labels[i, train_label[i]] = 1
        # 逐次训练
        for i in range(0, self.epochs):
            # 逐个样本训练
            for j in range(0, data_num):
                input_data = train_data[i]
                label = labels[i]
                self.forward(input_data)
                self.backward(label)
                if i==0 and j==0:
                    print("手写MLP第一轮梯度：")
                # 逐层更新weight和bias
                for k in range(0, layer_num):
                    # print(self.layers[k]['delta'].shape)
                    bias_delta = self.layers[k]['delta'].T
                    weight_delta = np.dot(bias_delta, self.layers[k]['input'].T)
                    self.layers[k]['weight'] = self.layers[k]['weight'] - self.lr * weight_delta
                    self.layers[k]['bias'] = self.layers[k]['bias'] - self.lr * bias_delta
                    if i==0 and j==0:
                        print("Layer {0}: weight_delta=".format(k))
                        print(-1 * self.lr * weight_delta)
                        print("Layer {0}: bias_delta=".format(k))
                        print(-1 * self.lr * bias_delta)
            if (i+1)%(self.epochs//20)==0:
                print('epoch {0}/{1}: loss={2}'.format(i+1, self.epochs, self.loss[i]))


class PYTORCH_MLP(torch.nn.Module):  
    def __init__(self):
        super(PYTORCH_MLP,self).__init__()  
        # 初始化
        self.fc1 = torch.nn.Linear(5,4) # 隐层1 
        self.fc2 = torch.nn.Linear(4,4) # 隐层2
        self.fc3 = torch.nn.Linear(4,3) # 输出层
     
    def forward(self,din):
        # 前向传播
        dout = torch.sigmoid(self.fc1(din))  
        dout = torch.sigmoid(self.fc2(dout))
        dout = torch.softmax(self.fc3(dout), dim=1) # 输出层使用 softmax 激活函数
        return dout


if __name__ == '__main__':
    epochs=100
    learing_rate=0.05
    train_data = random.random(size=(100, 5))
    train_label = random.randint(0,3,(100, 1))
    # pytorch实现的MLP
    pytorch_mlp = PYTORCH_MLP()
    # 交叉熵
    lossfunc = torch.nn.CrossEntropyLoss()
    # 随机梯度下降
    optimizer = torch.optim.SGD(pytorch_mlp.parameters(), lr=learing_rate)
    loss_dict = []
    train_num = train_label.shape[0]
    print("Pytorch MLP第一轮参数，同样也用作手写MLP的初始化参数：")
    L_W1 = pytorch_mlp.fc1.weight.detach().numpy()
    L_W2 = pytorch_mlp.fc2.weight.detach().numpy()
    L_W3 = pytorch_mlp.fc3.weight.detach().numpy()
    L_b1 = pytorch_mlp.fc1.bias.detach().numpy()
    L_b2 = pytorch_mlp.fc2.bias.detach().numpy()
    L_b3 = pytorch_mlp.fc3.bias.detach().numpy()
    print("L_W1=",pytorch_mlp.fc1.weight.detach().numpy())
    print("L_W2=",pytorch_mlp.fc2.weight.detach().numpy())
    print("L_W3=",pytorch_mlp.fc3.weight.detach().numpy())
    print("L_b1=",pytorch_mlp.fc1.bias.detach().numpy())
    print("L_b2=",pytorch_mlp.fc2.bias.detach().numpy())
    print("L_b3=",pytorch_mlp.fc3.bias.detach().numpy())
    for i in range(epochs):
        inputs = torch.from_numpy(train_data)
        inputs = inputs.to(torch.float32)
        labels_dup = train_label.flatten()
        targets = torch.from_numpy(labels_dup)
        targets = targets.to(torch.long)
        outputs = pytorch_mlp.forward(inputs)
        loss = lossfunc(outputs,targets)
        loss_dict.append(loss)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i==0:
            print("Pytorch MLP第一轮梯度：")
            print("L_W1=",pytorch_mlp.fc1.weight.grad)
            print("L_W2=",pytorch_mlp.fc2.weight.grad)
            print("L_W3=",pytorch_mlp.fc3.weight.grad)
            print("L_b1=",pytorch_mlp.fc1.bias.grad)
            print("L_b2=",pytorch_mlp.fc2.bias.grad)
            print("L_b3=",pytorch_mlp.fc3.bias.grad)
    # 自己实现的MLP
    model1 = MLP('crossentropy', 0.05, 100)
    model1.add_layer(5, 4, 'sigmoid', weight=L_W1, bias=L_b1)
    model1.add_layer(4, 4, 'sigmoid', weight=L_W2, bias=L_b2)
    model1.add_layer(4, 3, 'softmax', weight=L_W3, bias=L_b3)
    model1.train(train_data, train_label)
    plt.plot(range(0, 100), model1.loss[0:100])
    plt.savefig('./fig.png',dpi=520)
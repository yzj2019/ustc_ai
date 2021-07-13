from process_data import load_and_process_data
from evaluation import get_macro_F1,get_micro_F1,get_acc
import numpy as np


# 实现线性回归的类
class LinearClassification:

    '''参数初始化 
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''
    def __init__(self,lr=0.001,Lambda= 0.001,epochs = 1000):
        self.lr=lr
        self.Lambda=Lambda
        self.epochs =epochs

    '''根据训练数据train_features,train_labels计算梯度更新参数W'''
    def fit(self,train_features,train_labels):
        ''''
        需要你实现的部分
        '''
        # 计算初始omega
        datanum = train_features.shape[0]           # 数据个数
        featnum = train_features.shape[1] + 1       # 标签个数+1
        e = np.ones([datanum,1], dtype=float)
        X = np.c_[train_features,e]
        i = np.eye(featnum, dtype=float)
        # print(X.shape)
        # print(X[0])
        a = np.dot(X.T, train_labels)               # X^TY
        b = np.add(np.dot(X.T, X), self.Lambda*i)   # X^TX + Lambda*I
        b = np.linalg.inv(b)                        # (X^TX + Lambda*I)^T
        self.omega = np.dot(b,a)
        # 开始迭代
        j = 0
        while j<self.epochs:
            # 每轮迭代
            k = 0
            if j%100==0:
                print("round {0}".format(j))
            j = j + 1
            #print(self.omega)
            while k<datanum:
                # 逐个数据迭代
                x = X[k].reshape(1,featnum)
                y = train_labels[k].reshape(1,1)
                b = np.add(np.dot(x.T, x), self.Lambda*i)                  # x^Tx + Lambda*I
                delta = np.subtract(2 * np.dot(b, self.omega), 2*x.T*y)    # 梯度
                self.omega = np.subtract(self.omega, self.lr * delta)
                k = k + 1


    '''根据训练好的参数对测试数据test_features进行预测，返回预测结果
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目'''
    def predict(self,test_features):
        ''''
        需要你实现的部分
        '''
        datanum = test_features.shape[0]
        e = np.ones([datanum,1], dtype=float)
        X = np.c_[test_features,e]
        return np.round(np.dot(X, self.omega))  # 简单舍入，作为分类的值



def main():
    # 加载训练集和测试集
    train_data,train_label,test_data,test_label=load_and_process_data()
    lR=LinearClassification()
    lR.fit(train_data,train_label) # 训练模型
    pred=lR.predict(test_data) # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))


main()

import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1,get_macro_F1,get_acc

class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''
    def __init__(self):
        self.Pc={}
        self.Pxc={}

    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''
    def fit(self,traindata,trainlabel,featuretype):
        '''
        需要你实现的部分
        '''
        # 计算先验概率
        y_lables = np.unique(trainlabel)    # 所有标签去重
        y_counts = len(trainlabel)          # 所有数据个数
        N = len(y_lables)                   # 标签种类数
        y_lable_data = {}                   # 标签对应的数据dict
        for y_lable in y_lables:
            # 遍历每种分类
            self.Pc[y_lable] = (trainlabel[trainlabel==y_lable].shape[0] + 1) / (y_counts + N)
            y_lable_data[y_lable] = traindata[(trainlabel==y_lable).reshape(trainlabel.shape[0],)]       # 标签为y_lable的数据

        # 计算后验概率
        for i in range(0,traindata.shape[1]):
            # 遍历每个attribute
            x_p = {}
            x_p['type'] = featuretype[i]    # 记录attribute种类
            x_p['data'] = {}                # 记录lable->后验概率/概率分布
            for y_lable in y_lables:
                # 遍历每种分类
                x_p['data'][y_lable] = {}                       # 记录后验概率/概率分布
                x_i_datas = y_lable_data[y_lable][:,i]          # 标签为y_lable的数据中，attribute i的所有数据
                if featuretype[i]:
                    # 连续型attribute，记录gauss分布的mu和sigma
                    # print(np.mean(x_i_datas))
                    x_p['data'][y_lable]['mu']    = np.mean(x_i_datas)
                    # print(np.std(x_i_datas))
                    x_p['data'][y_lable]['sigma'] = np.std(x_i_datas)
                else:
                    # 离散型attribute，则记录值->Laplace平滑后验概率
                    x_i_types = np.unique(traindata[:,i])
                    for x_i_type in x_i_types:
                        # 对该attribute的每个取值
                        x_p['data'][y_lable][x_i_type] = (x_i_datas[x_i_datas==x_i_type].shape[0] + 1) / (y_lable_data[y_lable].shape[0] + len(x_i_types))
            self.Pxc[i] = x_p

    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''
    def predict(self,features,featuretype):
        '''
        需要你实现的部分
        '''       
        lables = np.empty([features.shape[0],1], dtype=int)     # 保存分类结果
        attr_num = len(featuretype)
        for i in range(0,features.shape[0]):
            # 逐条数据处理
            x = features[i]
            max_p = 0
            for y_lable in self.Pc.keys():
                # 遍历每种标签
                p = 1
                for j in range(0,attr_num):
                    # 遍历每种attribute
                    x_p = self.Pxc[j]
                    x_j = x[j]
                    if x_p['type']:
                        # 连续型，用mu和sigma计算gauss分布概率密度
                        mu = x_p['data'][y_lable]['mu']
                        sigma = x_p['data'][y_lable]['sigma']
                        p = p * np.exp(-(x_j - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
                    else:
                        # 离散型
                        p = p * x_p['data'][y_lable][x_j]
                p = p * self.Pc[y_lable]
                if p > max_p:
                    y = y_lable
                    max_p = p
            lables[i,0] = y
        return lables




def main():
    # 加载训练集和测试集
    train_data,train_label,test_data,test_label=load_and_process_data()
    feature_type=[0,1,1,1,1,1,1,1] #表示特征的数据类型，0表示离散型，1表示连续型

    Nayes=NaiveBayes()
    Nayes.fit(train_data,train_label,feature_type) # 在训练集上计算先验概率和条件概率

    pred=Nayes.predict(test_data,feature_type)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))

main()
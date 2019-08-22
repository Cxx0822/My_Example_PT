import torch
from torch.autograd import Variable

import numpy as np
import os
import os.path

# 基本信息配置
batch_n = 32
hidden_dim = 100
input_dim = 320
output_dim = 2

max_epochs = 50
learning_rate = 0.05

filename = "data.txt"


def load_data(filename, input_dim):
    # 读取数据集
    with open(filename, "r") as fr:
        arrayOfLines = fr.readlines()
        numberOfLines = len(arrayOfLines)

        X = np.zeros((numberOfLines, input_dim))
        Y = []
        index = 0  
        for line in arrayOfLines:
            line = line.strip()
            listFromLine = line.split(' ')
            X[index, :] = listFromLine[0:input_dim]      # 读取特征列

            if listFromLine[-1] == '0':                  # 读取标签列
                Y.append(0)
            elif listFromLine[-1] == '1':
                Y.append(1)     

            index += 1  

    return X, Y


class Model(torch.nn.Module):
    # 构建网络结构
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.hidden = torch.nn.Linear(input_dim, hidden_dim)   
        self.out = torch.nn.Linear(hidden_dim, output_dim)   

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden(x))      
        x = self.out(x)
        return x


def train():
    # 训练模型
    X, Y = load_data(filename, input_dim)    # 加载数据集

    # 生成测试集和训练集
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

    # 保存测试集和训练集以便后续预测分析
    with open("X_test.txt", "w") as fr:
        for i in range(len(X_test)):
            for j in range(len(X_test[i])):
                fr.write(str(X_test[i][j]) + '\t')    
            fr.write('\n')    

    with open("Y_test.txt", "w") as fr:
        for i in range(len(Y_test)):
            fr.write(str(Y_test[i]) + '\t')    
        fr.write('\n')

    input = torch.FloatTensor(X_train)
    label = torch.LongTensor(Y_train)

    # 模型输出
    model = Model(input_dim, hidden_dim, output_dim)
    # 选择损失函数和优化器
    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练数据
    for epoch in range(50):
        out = model(input)                
        loss = loss_fun(out, label)    
        optimizer.zero_grad()   
        loss.backward()         
        optimizer.step()   

        prediction = torch.max(out, 1)[1] 
        pred_y = prediction.data.numpy()
        target_y = label.data.numpy()
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        print("Epoch: ", epoch, "Accuracy: ", accuracy, "Loss: ", loss.item())

    torch.save(model, 'model.pt')


def predict(X_test): 
    # 预测模型  
    model = torch.load('model.pt')

    test_input = torch.FloatTensor(X_test)
    out = model(test_input) 
    prediction = torch.max(out, 1)[1] 
    pred_y = prediction.item()
    print("pred_y: ", pred_y)


if __name__ == "__main__":    
    train()

    '''
    X_tests = []
    with open("X_test.txt", "r") as fr:
        for line in fr.readlines():    
            curLine = line.strip().split("\t")    
            floatLine = np.matrix(list(map(float, curLine)), dtype="float32")  
            X_tests.append(floatLine) 

    for X_test in X_tests:
        predict(X_test)
    '''

# My_Example_PT

# 平台
&emsp;&emsp;Windows 10    
&emsp;&emsp;PyTorch 1.2.0   
&emsp;&emsp;python 3.5.4 

## Simple Network
&emsp;&emsp;利用神经网络对文本数据集分类。

### 文件
&emsp;&emsp;data.txt      
&emsp;&emsp;simple_network.py 

### 数据集
&emsp;&emsp;首先确保数据集的格式和`data.txt`中的格式一致，即特征+标签。如果不一致，需要更改`simple_network.py`中的`load_data()`函数。其最终返回值为特征和标签。    

### 训练&预测
&emsp;&emsp;在`main`中依次选择训练和测试部分即可。  

## CNN
&emsp;&emsp;利用CNN对图片数据集分类。 

### 文件  
&emsp;&emsp;dataSet   
&emsp;&emsp;load_data.py         
&emsp;&emsp;info.yml      
&emsp;&emsp;model.py    
&emsp;&emsp;train.py   
&emsp;&emsp;predict.py     

### 配置
&emsp;&emsp;首先打开`info.yml`，更改里面的配置信息。 

### 数据集
&emsp;&emsp;数据集加载采用`DataLoader`模式，返回为可迭代的数据集和标签序列。   

### 模型
&emsp;&emsp;采用普通的卷积，池化和全连接层。    

### 训练&预测
&emsp;&emsp;在`main`中依次选择训练和测试部分即可。     

[详细说明文档](https://cxx0822.github.io/2019/08/18/PyTorch%E7%9A%84%E7%AE%80%E5%8D%95%E4%BD%BF%E7%94%A8/#more)
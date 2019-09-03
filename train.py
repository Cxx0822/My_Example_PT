# -*-coding:utf-8-*-
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from model import Models
from load_data import * 
import yaml
import time
import datetime

with open("info.yml") as stream:
    my_data = yaml.load(stream, Loader=yaml.FullLoader)
    # python3.6 可能需要去掉 Loader=yaml.FullLoader

data_dir_train = my_data['data_dir_train']
data_dir_valid = my_data['data_dir_valid']
image_dim = my_data['image_dim'] 
n_classes = my_data['n_classes'] 

image_size = my_data['image_size']
batch_size = my_data['batch_size']
learning_rate = my_data['learning_rate']
epochs = my_data['epochs']

writer = SummaryWriter()


def my_train():
    # 是否有GPU加速
    use_gpu = torch.cuda.is_available()

    dataloader = {"train": load_data(data_dir_train, image_size, batch_size), 
                  "valid": load_data(data_dir_valid, image_size, batch_size)}

    # 加载模型、损失、优化器等
    model = Models(image_dim, n_classes)
    loss_f = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if use_gpu:
        model = model.cuda()

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 10)

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        # 计时
        start_time = time.time()

        for i, data in enumerate(dataloader["train"], 0):
            train_images, train_labels = data
            if use_gpu:
                train_images, labels = Variable(train_images.cuda()), Variable(train_labels.cuda())
            else:
                train_images, labels = Variable(train_images), Variable(train_labels)

            optimizer.zero_grad()                              # 梯度置零
            outputs = model(train_images)                      # 模型输出
            _, train_predicted = torch.max(outputs.data, 1)    # 预测结果

            train_correct += (train_predicted == labels.data).sum()

            loss = loss_f(outputs, labels)    # 计算损失
            loss.backward()                   # 反向传播
            optimizer.step()                  # 单步优化（调整学习率）

            running_loss += loss.item()
            train_total += train_labels.size(0)

        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        runtime = time.time() - start_time
        print('now_time: %s, , time/epoch:%.2fs, time left: %.2fhours' % (str(now_time), runtime, (epochs - epoch) * runtime / 3600))
        print('train %d epoch loss: %.3f  acc: %.3f ' % (epoch + 1, running_loss / train_total, 100 * train_correct / train_total))

        writer.add_scalar('Train/Loss', running_loss / train_total, epoch + 1)
        writer.add_scalar('Train/Acc', 100 * train_correct / train_total, epoch + 1) 

    torch.save(model, 'model.pt')


if __name__ == "__main__":
    my_train()

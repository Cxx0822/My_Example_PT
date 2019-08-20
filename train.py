# -*-coding:utf-8-*-
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from model import Models
from load_data import * 
import yaml

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
    use_gpu = torch.cuda.is_available()

    dataloader = {"train": load_data(data_dir_train, image_size, batch_size), 
                  "valid": load_data(data_dir_valid, image_size, batch_size)}

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
        for i, data in enumerate(dataloader["train"], 0):
            inputs, train_labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(train_labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct += (train_predicted == labels.data).sum()
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_total += train_labels.size(0)

        print('train %d epoch loss: %.3f  acc: %.3f ' % (epoch + 1, running_loss / train_total, 100 * train_correct / train_total))
        writer.add_scalar('Train/Loss', running_loss / train_total, epoch + 1)
        writer.add_scalar('Train/Acc', 100 * train_correct / train_total, epoch + 1) 

        # 模型测试
        correct = 0
        test_loss = 0.0
        test_total = 0
        test_total = 0
        model.eval()
        for data in dataloader["valid"]:
            images, labels = data
            if use_gpu:
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = loss_f(outputs, labels)
            test_loss += loss.item()
            test_total += labels.size(0)
            correct += (predicted == labels.data).sum()

        print('test  %d epoch loss: %.3f  acc: %.3f ' % (epoch + 1, test_loss / test_total, 100 * correct / test_total))
        writer.add_scalar('Test/Loss', test_loss / test_total, epoch + 1)
        writer.add_scalar('Test/Acc', 100 * correct / test_total, epoch + 1)
    writer.close()

    torch.save(model, 'model.pt')


if __name__ == "__main__":
    my_train()

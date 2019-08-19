import torch
from torch.autograd import Variable
from model import Models
from load_data import * 
import matplotlib.pyplot as plt
import yaml

with open("info.yml") as stream:
    my_data = yaml.load(stream, Loader=yaml.FullLoader)

data_dir_test = my_data['data_dir_test']
data_dir_valid = my_data['data_dir_valid']
image_dim = my_data['image_dim'] 
n_classes = my_data['n_classes'] 

image_size = my_data['image_size']
batch_size = 4


def my_predict():
    use_gpu = torch.cuda.is_available()

    test_data = load_data(data_dir_test, image_size=image_size, batch_size=batch_size)
    X_test, y_test = next(iter(test_data))

    model = torch.load('model.pt')

    if use_gpu:
        model = model.cuda()

    if use_gpu:
        images = Variable(X_test.cuda())
    else:
        images = Variable(X_test)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    print("Predict Label is: ", predicted.data)
    print("Real Label is :", y_test.data)

    img = torchvision.utils.make_grid(X_test)
    img = img.numpy().transpose([1, 2, 0])   # 转成numpy在转置
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    my_predict()

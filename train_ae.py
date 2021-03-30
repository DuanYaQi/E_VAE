import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn, optim

from ae import AE

import visdom

def train():
    mnist_train = datasets.MNIST('../data/mnist',
                                    train = True, 
                                    transform=transforms.Compose([transforms.ToTensor()]),
                                    download=True)
    mnist_train = DataLoader(mnist_train, batch_size = 32, shuffle = True)
    mnist_test = datasets.MNIST('../data/mnist',
                                    train = False, 
                                    transform=transforms.Compose([transforms.ToTensor()]),
                                    download=True)
    mnist_test = DataLoader(mnist_test, batch_size = 32, shuffle = True)

    #不需要label，因为是无监督学习
    x, _ = iter(mnist_train).next()  
    print('x:', x.shape)

    device = torch.device('cuda')

    model = AE().to(device)
    criteon = nn.MSELoss() # loss function
    optimzer = optim.Adam(model.parameters(), lr = 1e-3)
    print(model)    

    vis = visdom.Visdom()

    for epoch in range(1000):
        
        # 训练过程
        for batchIdx, (x, _) in enumerate(mnist_train):
            #forwardp [b, 1, 28, 28]
            x = x.to(device)
            x_hat = model(x)
            loss = criteon(x_hat, x)

            #backward
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

        # 打印loss
        print('epoch:', epoch, '  loss:', loss.item())

        # 测试过程
        x, _ = iter(mnist_test).next()
        x = x.to(device)
        with torch.no_grad():               #测试不用梯度
            x_hat = model(x)

        vis.images(x, nrow=8, win='x', opts=dict(title='x'))   #画输入
        vis.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))   #画输出

if __name__ == "__main__":
    train()
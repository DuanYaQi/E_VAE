import torch
from torch import nn

class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()
        # [b,784] => [b, 20]
        self.encoder = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,20),
            nn.ReLU(),
        )
        # [b, 20] => [b, 764] 因为是二进制图像最后sigmoid归到【0，1】区间
        self.decoder = nn.Sequential(
            nn.Linear(20,64),
            nn.ReLU(),
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
            param         x [b, 1, 28, 28]  第2个 1 表示为 1个通道  b是batch
            return 
        """
        batch_size = x.size(0)
        # flatten 拉为1维
        x = x.view(batch_size, 784)
        # encoder
        x = self.encoder(x)
        # decoder
        x = self.decoder(x)
        # reshape
        x = x.view(batch_size, 1, 28, 28)

        return x
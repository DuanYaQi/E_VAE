import torch
from torch import nn
from torch.nn.modules.activation import ReLU

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        # [b,784] => [b, 20]
        # u: [b, 10]  相当于潜在向量为10维 每个潜在单元有独自的分布
        # sigma: [b, 10]
        self.encoder = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,20),
            nn.ReLU(),
        )


        # [b, 10] => [b, 764] 因为是二进制图像最后sigmoid归到[0, 1]区间 10是因为20有两个参数 20/2
        self.decoder = nn.Sequential(
            nn.Linear(10,64),
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
        # [b, 20], including mean and sigma
        h_ = self.encoder(x)

        # [b, 20] =>> [b, 10] and [b, 10]
        mu, sigma = h_.chunk(2, dim=1)
        
        # 重参数 epison~N(0, 1)
        h = mu + sigma * torch.randn_like(sigma)

        # decoder [b, 10] =>> [b, 784]
        x_hat = self.decoder(h)
        # reshape [b, 784] =>> [b, 28, 28]
        x_hat = x_hat.view(batch_size, 1, 28, 28)

        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma, 2) - 
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1  # x接近于0时 log趋于负无穷 +1e-8使log内总是大于0 
        ) / (batch_size * 28 * 28) # 因为MSE是pixel级别的 所以kld也应该是pixel级别的

        return x_hat, kld
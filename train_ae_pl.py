import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl


from argparse import ArgumentParser
import visdom

# --------------------------------------------------------------------------------------
class AE(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.network = EncoderDecoder()

    def forward(self, x):
        """
            param         x [b, 1, 28, 28]  第2个 1 表示为 1个通道  b是batch
            return 
        """
        x_hat = self.network(x)

        return x_hat

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x = batch[0]
        x_hat = self.network(x)
        vis.images(x, nrow=8, win='x', opts=dict(title='x'))   #画输入
        vis.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))   #画输入
        loss = self.loss_func(x_hat, x)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def loss_func(self, x_hat, x):  
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# --------------------------------------------------------------------------------------
class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
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

# --------------------------------------------------------------------------------------
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--top_in_dir', type=str, help='Top-dir of where datasets are stored', default = '../data/mnist')
    parser.add_argument('--bneck_size', type=int, help='Bottleneck-AE size', default = 20)                 #TODO: Adapt haparms
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--max_epochs', type=int, default = 1000)
    return parser.parse_args()


# --------------------------------------------------------------------------------------
def train():
    args = parse_arguments()
    trainer_config = {
        'gpus'                   : 1,  # Set this to None for CPU training
        'max_epochs'             : args.max_epochs,
        'automatic_optimization' : True,
    }

    mnist_train = datasets.MNIST(args.top_in_dir,
                                    train = True, 
                                    transform=transforms.Compose([transforms.ToTensor()]),
                                    download=True)
    mnist_train = DataLoader(mnist_train, batch_size = args.batch_size, shuffle = True)
    mnist_test = datasets.MNIST(args.top_in_dir,
                                    train = False, 
                                    transform=transforms.Compose([transforms.ToTensor()]),
                                    download=True)
    mnist_test = DataLoader(mnist_test, batch_size = args.batch_size, shuffle = True)
  

    

    # network
    autoencoder = AE()
    trainer = pl.Trainer(**trainer_config)
    trainer.fit(autoencoder, mnist_train)

    #vis.images(x, nrow=8, win='x', opts=dict(title='x'))   #画输入
    #vis.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))   #画输出

if __name__ == "__main__":
    vis = visdom.Visdom()
    train()
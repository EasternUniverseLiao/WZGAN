import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class NetG(nn.Module):
    def __init__(self, opt):
        super(NetG, self).__init__()
        ngf = opt.ngf  # 生成器的feature map数
        # 输入是nz维度的噪声，nz*1*1的feature map
        # 1*1 -> 4*4
        self.layer1 = nn.Sequential(
            #  (H_input-1)*stride-2*padding+kernel_size = H_out
            nn.ConvTranspose2d(in_channels=opt.nz, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU()
        )
        # 4*4 -> 8*8
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU()
        )
        # 8*8 -> 16*16
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU()
        )
        # 16*16 -> 32*32
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU()
        )
        # 32*32 -> 96*96
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf, out_channels=3, kernel_size=5, stride=3, padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.layer5(X)
        return X


class NetD(nn.Module):
    def __init__(self, opt):
        super(NetD, self).__init__()
        ndf = opt.ndf
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=ndf, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        # 96*96 -> 32*32
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2,inplace=True))
        # 32*32 -> 16*16
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2,inplace=True))
        # 16*16 -> 8*8
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2,inplace=True))
        # 8*8 -> 4*4
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # 输出一个概率
        )

    def forward(self, X):
        # print(X.shape)
        X = self.layer1(X)
        # print(X.shape)
        X = self.layer2(X)
        # print(X.shape)
        X = self.layer3(X)
        # print(X.shape)
        X = self.layer4(X)
        # print(X.shape)
        X = self.layer5(X)
        return X.view(-1)


class Opt:
    nz = 100
    ngf = 64
    ndf = 64


def test():
    X = torch.randn(size=(64, 100, 1, 1))
    opt = Opt()
    G = NetG(opt)
    imgs = G(X)
    print(imgs.shape)
    copy_imgs = imgs
    imgs = imgs.detach().numpy()
    imgs = imgs.swapaxes(1, 3)
    print(imgs[0].shape)
    plt.imshow(imgs[0])  # 一坨马塞克
    plt.show()

    D = NetD(opt)
    p = D(copy_imgs)
    print(p.shape, p)


if __name__ == '__main__':
    test()

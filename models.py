'''
  Anime face generation based on DCGAN, Generator and Discriminator definition
'''

import torch.nn as nn

# generator define
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        ngf = opt.ngf   #number of generator feature
        self.batch_size = opt.batch_size
        self.linear = nn.Sequential(
            nn.Linear(opt.nz, 16*16*ngf),   #Project and reshape input: nz-dim vector, output: (4,4,16*ngf) feature map
            nn.BatchNorm1d(16*16*ngf),
            nn.ReLU(inplace=True))
        self.model = nn.Sequential(
            nn.ConvTranspose2d(16*ngf, 8*ngf, kernel_size=4, stride=2, padding=1, bias=True),   #out(8,8,8*ngf)
            nn.BatchNorm2d(8*ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(8 * ngf, 4 * ngf, kernel_size=4, stride=2, padding=1, bias=True),  # out(16,16,4*ngf)
            nn.BatchNorm2d(4 * ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(4 * ngf, 2 * ngf, kernel_size=4, stride=2, padding=1, bias=True),  # out(32,32,2*ngf)
            nn.BatchNorm2d(2 * ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(2 * ngf, 3, kernel_size=4, stride=2, padding=1, bias=True),  # out(64,64,3)
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape((self.batch_size,4,4,-1))
        x = x.permute(0,3,1,2)
        x = self.model(x)

        return x


# Discriminator definition
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        ndf = opt.ndf

        self.model = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=True),  #Conv1 input(n,3,64,64)
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  #(n,ndf,32,32)

            nn.Conv2d(ndf, 2*ndf, kernel_size=4, stride=2, padding=1, bias=True), #Conv2
            nn.BatchNorm2d(2*ndf),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  #(n,2*ndf,16,16)

            nn.Conv2d(2*ndf, 4*ndf, kernel_size=4, stride=2, padding=1, bias=True), #Conv3
            nn.BatchNorm2d(4*ndf),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  #(n,4*ndf,8,8)

            nn.Conv2d(4*ndf, 8*ndf, kernel_size=4, stride=2, padding=1, bias=True), #Conv4
            nn.BatchNorm2d(8*ndf),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  #(n,8*ndf,4,4)

            nn.Conv2d(8*ndf, 1, kernel_size=4, stride=1, padding=0, bias=True),  # Conv5
            nn.Sigmoid()     # classification
        )

    def forward(self, x):
        return self.model(x)




from models import *
from datasets import AnimeFaceDatasets
import os
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from config import Config


opt = Config()

def train():
    ''' define training process'''
    device = torch.device('cuda') if opt.gpu else torch.device('cpu')

    # load datasets
    train_dataset = AnimeFaceDatasets(root = opt.root)
    # dataloader
    train_loader = dataloader.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle =True, num_workers=opt.num_workers)

    #instance models
    Gnet = Generator(opt).to(device)
    Dnet = Discriminator(opt).to(device)

    #define optimizer for two both net
    G_optimizer = optim.Adam(params=Gnet.parameters(), lr=opt.lr1, betas=(opt.beta, 0.999))
    D_optimizer = optim.Adam(params=Dnet.parameters(), lr=opt.lr2, betas=(opt.beta, 0.999))
    criterions = nn.BCELoss().to(device)

    #define real_label and fake_label
    real_label = torch.ones(opt.batch_size).to(device)
    fake_label = torch.zeros(opt.batch_size).to(device)

    #define a fixed noise
    fix_noise = torch.randn(opt.batch_size, opt.nz)

    #training...
    print('training...')
    for epoch in range(opt.epoch):
        for idx, imgs in enumerate(train_loader):
            gen_loss = 0.0
            dis_loss = 0.0
            imgs = imgs.to(device)
            noise = torch.randn_like(fix_noise)
            # update discriminator, fix generator
            if idx % opt.d_every == 0:
                #for real sample
                real_out = Dnet(imgs)
                #for fake smaple
                generate_img = Gnet(noise)
                fake_out = Dnet(generate_img)
                d_loss_real = criterions(real_out, real_label)
                d_loss_fake = criterions(fake_out, fake_label)
                d_loss = d_loss_fake.item() + d_loss_real.item()
                if idx % opt.verbose == 0:
                    print('Epoch: {},iteration: {},d_loss: {}'.format(epoch, idx, d_loss))
                d_loss_real.backward()
                d_loss_fake.backward()
                D_optimizer.step()

            #update generator, fix discriminator
            if idx % opt.g_every == 0:
                generate_img = Gnet(noise)
                fake_out = Dnet(generate_img)
                g_loss = criterions(fake_out, real_label)
                if idx % opt.verbose == 0:
                    print('Epoch: {}, iteration: {}, g_loss {}'.format(epoch, idx, g_loss.item()))
            if idx % 100 == 0:
                # save generation
                fix_fake_image = Gnet(fix_noise)
                img_save_path = os.path.join(opt.save_path, 'result')
                if os.path.exists(img_save_path) == False:
                    os.mkdir(img_save_path)
                torchvision.utils.save_image(fix_fake_image[:16], '%s/%s_%s.png' % (img_save_path, epoch,idx), nrow=4)

                g_loss.backward()
                G_optimizer.step()
        if epoch % 10 == 0:
            # save model
            checkpoint_path = os.path.join(opt.save_path, 'checkpoints')
            if os.path.exists(checkpoint_path) == False:
                os.mkdir(checkpoint_path)
            torch.save(Gnet.state_dict(), '%s/Gnet_%s.pth' % (checkpoint_path, epoch))
            torch.save(Dnet.state_dict(), '%s/Dnet_%s.pth' % (checkpoint_path, epoch))
            print('end models saving....')
    print('end training....')




if __name__ == '__main__':
    train()





import torch
from models import *
from config import Config
import os
from torchvision.utils import save_image

opt = Config()

def test():
    #define Generator
    Gen = Generator(opt)

    # load pretrained models
    print('Loading state dict....')
    gen_state_dict = torch.load(os.path.join(opt.state_path, 'Gnet_{}.pth'.format(opt.state_num)))
    Gen.load_state_dict(gen_state_dict)
    print('End loading state dict....')

    # generate uniform noise (opt.batch_size, opt.nz)
    noise_input = torch.randn(opt.batch_size, opt.nz)
    gen_images = Gen(noise_input)

    #save as grids
    if not os.path.exists(opt.test_out):
        os.mkdir(opt.test_out)
    #save front 16 generate
    save_image(gen_images[:64], os.path.join(opt.test_out, 'test_out.png'), nrow = 8)
    print('Generate images save as grid done...')
    print('End testing.....')


if __name__ == '__main__':
    test()



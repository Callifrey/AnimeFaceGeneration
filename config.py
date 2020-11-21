'''
   training congfig
'''

class Config(object):
    def __init__(self):
        self.ngf = 32
        self.ndf = 32
        self.lr1 = 1e-3
        self.lr2 = 1e-3
        self.gpu = False
        self.root = '/kaggle/input/anime-faces/data/'
        self.batch_size = 32
        self.num_workers = 4
        self.epoch = 10
        self.beta = 0.9
        self.nz = 128

        self.d_every = 1
        self.g_every = 1

        self.verbose = 10
        self.save_path = '/kaggle/working/AnimeFaceGeneration/output/'

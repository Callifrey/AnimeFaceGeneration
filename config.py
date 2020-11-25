'''
   training congfig
'''

class Config(object):
    def __init__(self):
        self.ngf = 64
        self.ndf = 64
        self.lr1 = 1e-4
        self.lr2 = 1e-4
        self.gpu = True
        self.root = './data/data'
        self.batch_size = 32
        self.num_workers = 4
        self.epoch = 400
        self.beta = 0.9
        self.nz = 100

        self.d_every = 2
        self.g_every = 1

        self.verbose = 10
        self.save_path = './data/out'
        self.load_state = False
        self.state_path = './data/out/checkpoints'
        self.state_num = 390
        self.test_out = './data/out/test'



import h5py
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim
from torch.optim.lr_scheduler import *
from tensorboardX import SummaryWriter
import os

import configs
import utils.feature_loader as feat_loader
from utils.data_utils import SimpleDataManager
from methods.simple import Simple
from utils.io_utils import model_dict, parse_args, get_resume_file, get_best_file


class Experiment():
    def __init__(self, params):
        np.random.seed(10)

        image_size = 224

        real_base_file = configs.data_dir['real'] + 'base.json'
        real_val_file = configs.data_dir['real'] + 'val.json'
        rumor_base_file = configs.data_dir['rumor'] + 'base.json'
        rumor_val_file = configs.data_dir['rumor'] + 'val.json'

        real_base_datamgr = SimpleDataManager(image_size, configs.real_normalize_param, batch_size=32)
        real_base_loader = real_base_datamgr.get_data_loader(real_base_file, aug=params.train_aug)
        rumor_base_datamgr = SimpleDataManager(image_size, configs.rumor_normalize_param, batch_size=32)
        rumor_base_loader = rumor_base_datamgr.get_data_loader(rumor_base_file, aug=params.train_aug)
        real_val_datamgr = SimpleDataManager(image_size, configs.real_normalize_param, batch_size=32)
        real_val_loader = real_val_datamgr.get_data_loader(real_val_file, aug=params.train_aug)
        rumor_val_datamgr = SimpleDataManager(image_size, configs.rumor_normalize_param, batch_size=32)
        rumor_val_loader = rumor_val_datamgr.get_data_loader(rumor_val_file, aug=params.train_aug)

        novel_file = configs.data_dir['test'] + 'novel.json'

        novel_datamgr = SimpleDataManager(image_size, batch_size=64)
        novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)

        optimizer = params.optimizer

        shake_config = {'shake_forward': params.shake_forward, 'shake_backward': params.shake_backward,
                        'shake_picture': params.shake_picture}

        model = Simple(model_dict[params.model], 10, params.shake, shake_config)
        model = model.cuda()

        key = params.tag
        writer = SummaryWriter(log_dir=os.path.join(params.vis_log, key))

        params.checkpoint_dir = '%s/checkpoints/%s' % (configs.save_dir, params.checkpoint_dir)

        if not os.path.isdir(params.vis_log):
            os.makedirs(params.vis_log)

        outfile_template = os.path.join(params.checkpoint_dir.replace("checkpoints", "features"), "%s.hdf5")

        if params.mode == 'train' and not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if params.resume or params.mode == 'test':
            if params.mode == 'test':
                resume_file = get_best_file(params.checkpoint_dir)
            else:
                resume_file = get_resume_file(params.checkpoint_dir)

            if resume_file is not None:
                tmp = torch.load(resume_file)
                params.start_epoch = tmp['epoch'] + 1
                model.load_state_dict(tmp['state'])

        self.params = params
        self.real_val_file = real_val_file
        self.real_base_file = real_base_file
        self.rumor_val_file = rumor_val_file
        self.rumor_base_file = rumor_base_file
        self.image_size = image_size
        self.optimizer = optimizer
        self.outfile_template = outfile_template
        self.novel_loader = novel_loader
        self.real_base_loader = real_base_loader
        self.real_val_loader = real_val_loader
        self.rumor_base_loader = rumor_base_loader
        self.rumor_val_loader = rumor_val_loader
        self.writer = writer
        self.model = model
        self.key = key

    def train(self):
        if self.optimizer == 'Adam':
            train_optimizer = torch.optim.Adam(self.model.parameters())
            train_scheduler = None
        elif self.optimizer == 'SGD':
            train_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.train_lr, momentum=0.9,
                                              weight_decay=0.001)
            train_scheduler = StepLR(train_optimizer, step_size=50, gamma=0.1)
        else:
            raise ValueError('Unknown optimizer, please define by yourself')

        max_acc = 0
        start_epoch = self.params.start_epoch
        stop_epoch = self.params.stop_epoch
        for epoch in range(start_epoch, stop_epoch):
            self.model.train()
            self.model.train_loop(epoch, self.real_base_loader, self.rumor_base_loader, train_optimizer, train_scheduler, self.writer)

            self.model.eval()
            acc = self.test('val', epoch)

            if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
                print("best model! save...")
                max_acc = acc
                outfile = os.path.join(self.params.checkpoint_dir, 'best_model.tar')
                torch.save({'epoch': epoch, 'state': self.model.state_dict()}, outfile)

            if (epoch % self.params.save_freq == 0) or (epoch == stop_epoch - 1):
                outfile = os.path.join(self.params.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch': epoch, 'state': self.model.state_dict()}, outfile)

    def test(self, split='novel', epoch=0):
        self.outfile = self.outfile_template % split
        if split == 'novel':
            acc = self.model.test_loop(self.novel_loader)
        else:
            acc = self.model.val_loop(epoch, self.real_val_loader, self.rumor_val_loader, self.writer)

        print('Test Acc = %4.2f%%' % acc)
        return acc

    def run(self):
        if self.params.mode == 'train':
            self.train()
        elif self.params.mode == 'test':
            self.test()


if __name__ == '__main__':
    params = parse_args()
    exp = Experiment(params)
    exp.run()

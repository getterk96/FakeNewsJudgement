import numpy as np
import os
import glob
import argparse
import backbone
from torchvision import models

model_dict = dict(
    ResNet10=backbone.ResNet10,
    ResNet18=backbone.ResNet18,
    ResNet34=backbone.ResNet34,
    ResNet50=backbone.ResNet50,
    ResNet101=backbone.ResNet101)


def parse_args():
    parser = argparse.ArgumentParser(description='few-shot script')
    parser.add_argument('--mode', default='train', help='model: train/test')
    parser.add_argument('--model', default='ResNet18',
                        help='model: Conv{4|6} / ResNet{10|18|34|50|101}')
    parser.add_argument('--optimizer', default='SGD',
                        help='the optimizer of the training process')
    parser.add_argument('--train_lr', default=0.01, type=float, help='training learning rate')
    parser.add_argument('--test_lr', default=0.01, type=float, help='testing learning rate')
    parser.add_argument('--train_aug', action='store_true',
                        help='perform data augmentation or not during training')
    parser.add_argument('--save_freq', default=50, type=int, help='save frequency')
    parser.add_argument('--start_epoch', default=0, type=int, help='starting epoch')
    parser.add_argument('--stop_epoch', default=-1, type=int,
                        help='stopping epoch')
    parser.add_argument('--resume', action='store_true',
                        help='continue from previous trained model with largest epoch')
    parser.add_argument('--warmup', action='store_true',
                        help='continue from baseline, neglected if resume is true')  # never used in the paper
    parser.add_argument('--vis_log', default='/home/gaojinghan/closer/vis_log', help='the tensorboard log storage dir')
    parser.add_argument('--tag', default='', help='the tag of the experiment')
    parser.add_argument('--checkpoint_dir', default='', help='the place to store checkpoints')

    return parser.parse_args()


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
import json

import numpy as np
import cv2
from PIL import Image
import random

# calculate means and std
train_txt_path = '/home/gaojinghan/FakeNewsJudgement/filelists/test/novel.json'

img_h, img_w = 32, 32
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []
new_names = []
new_labels = []

with open(train_txt_path, 'r') as f:
    j = json.load(f)
    names = j['image_names']
    labels = j['image_labels']
    random.shuffle(names)  # shuffle , 随机挑选图片

    for i in range(len(names)):
        try:
            img_path = names[i]
            img = Image.open(img_path).convert('RGB')
            new_names.append(names[i])
            new_labels.append(labels[i])
            if i % 1000 == 0:
                print(f'{i} pics are done!')
        except:
            print(f'Error: {names[i]}')

with open(train_txt_path, 'w') as f:
    j['image_names'] = new_names
    j['image_labels'] = new_labels
    json.dump(j, f)


imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

means.reverse()  # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
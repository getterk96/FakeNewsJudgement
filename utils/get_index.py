import os, json
from PIL import Image
import sys


def processImage(path, infile):
    try:
        im = Image.open(os.path.join(path, infile))
    except IOError:
        sys.exit(1)
    i = 0
    mypalette = im.getpalette()
    print(infile)
    try:
        while 1:
            im.putpalette(mypalette)
            new_im = Image.new("RGB", im.size)
            new_im.paste(im)
            new_im.save(os.path.join(path, infile[:-4] + '.jpg'))

            i += 1
            im.seek(im.tell() + 1)

    except ValueError:
        return True  # end of sequence
    except EOFError:
        pass
    return False



true_data_dir = ['/data/gaojinghan/FakeNews/true_pic_1', '/data/gaojinghan/FakeNews/truth_pic_2']
rumor_data_dir = ['/data/gaojinghan/FakeNews/rumor_pic']
test_data_dir = ['/data/gaojinghan/FakeNews/stage1_test']


with open('../filelists/real/all.json', 'w') as f:
    j = {'image_names': [], 'image_labels': []}

    for dd in true_data_dir:
        for pf in os.listdir(dd):
            if pf[-4:] == '.gif':
                continue
            j['image_names'].append(os.path.join(dd, pf))
            j['image_labels'].append(1)

    json.dump(j, f)

with open('../filelists/rumor/all.json', 'w') as f:
    j = {'image_names': [], 'image_labels': []}

    for dd in rumor_data_dir:
        for pf in os.listdir(dd):
            if pf[-4:] == '.gif':
                continue
            j['image_names'].append(os.path.join(dd, pf))
            j['image_labels'].append(0)

    json.dump(j, f)

with open('../filelists/test/novel.json', 'w') as f:
    j = {'image_names': [], 'image_labels': []}

    for dd in test_data_dir:
        for pf in os.listdir(dd):
            if pf[-4:] == '.gif':
                Image.open(os.path.join(dd, pf)).convert('RGB').save(os.path.join(dd, pf[:-4] + '.jpg'))
            j['image_names'].append(os.path.join(dd, pf))
            j['image_labels'].append(0)

    json.dump(j, f)
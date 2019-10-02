import os, json

true_data_dir = ['/Users/gaojinghan/Downloads/true_pic_1', '/Users/gaojinghan/Downloads/truth_pic_2']
rumor_data_dir = ['/Users/gaojinghan/Downloads/rumor_pic']
test_data_dir = ['/Users/gaojinghan/Downloads/stage1_test']


with open('all.json', 'w') as f:
    j = {'image_names': [], 'image_labels': []}

    for dd in true_data_dir:
        for pf in os.listdir(dd):
            if pf[-4:] == '.gif':
                continue
            j['image_names'].append(os.path.join(dd, pf))
            j['image_labels'].append(1)

    json.dump(j, f)

with open('all.json', 'w') as f:
    j = {'image_names': [], 'image_labels': []}

    for dd in rumor_data_dir:
        for pf in os.listdir(dd):
            if pf[-4:] == '.gif':
                continue
            j['image_names'].append(os.path.join(dd, pf))
            j['image_labels'].append(0)

    json.dump(j, f)

with open('novel.json', 'w') as f:
    j = {'image_names': [], 'image_labels': []}

    for dd in test_data_dir:
        for pf in os.listdir(dd):
            if pf[-4:] == '.gif':
                continue
            j['image_names'].append(os.path.join(dd, pf))
            j['image_labels'].append(0)

    json.dump(j, f)
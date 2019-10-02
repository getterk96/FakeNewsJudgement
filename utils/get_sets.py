import json

ratio = 5

datadir = '/home/gaojinghan/FakeNewsJudgement/filelists/rumor'
datafile = '/home/gaojinghan/FakeNewsJudgement/filelists/rumor/all.json'

with open(datafile, 'r') as f:
    j = json.load(f)
    image_names = j['image_names']
    image_labels = j['image_labels']

    base_names = image_names[:-len(image_names) // ratio]
    val_names = image_names[-len(image_names) // ratio:]

    base_labels = image_labels[:-len(image_labels) // ratio]
    val_labels = image_labels[-len(image_labels) // ratio:]

    with open(f'{datadir}/base.json', 'w') as f:
        json.dump({'image_names': base_names, 'image_labels': base_labels}, f)

    with open(f'{datadir}/val.json', 'w') as f:
        json.dump({'image_names': val_names, 'image_labels': val_labels}, f)

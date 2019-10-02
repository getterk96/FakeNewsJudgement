save_dir = '/home/gaojinghan/FakeNewsJudgement/saves'
data_dir = {}
data_dir['real'] = './filelists/real/'
data_dir['rumor'] = './filelists/rumor/'
data_dir['test'] = './filelists/test/'
real_normalize_param = dict(mean = [0.5952869, 0.5750592, 0.5580415], std = [0.33048502, 0.33081228, 0.33782595])
rumor_normalize_param = dict(mean = [0.56003433, 0.53176796, 0.5091385], std = [0.32130036, 0.32071647, 0.32737064])
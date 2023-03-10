import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np


class Dataset(data.Dataset):
    def __init__(self, datasets, mode='train', transform=None, return_size=True):
        self.return_size = return_size
        if type(datasets) != list:
            datasets = [datasets]
        self.datas_id = []
        for (i, dataset) in enumerate(datasets):
            if mode == 'train':
                data_dir = './dataset/train/{}'.format(dataset)
                imgset_path = data_dir + '/train.txt'

            else:
                data_dir = './dataset/test/{}'.format(dataset)
                imgset_path = data_dir + '/test.txt'

            imgset_file = open(imgset_path)

            for line in imgset_file:
                data = {}
                img_path = line.strip("\n").split(" ")[0]
                gt_path = line.strip("\n").split(" ")[1]
                data['img_path'] = img_path
                data['gt_path'] = gt_path
                if dataset == 'DUTS-TR':
                    data['split'] = dataset
                else:
                    flow_path = line.strip("\n").split(" ")[2]
                    data['flow_path'] = flow_path
                    data['split'] = data['img_path'].split('/')[-3]
                data['dataset'] = dataset
                self.datas_id.append(data)
        self.transform = transform

    def __getitem__(self, item):

        assert os.path.exists(self.datas_id[item]['img_path']), (
            '{} does not exist'.format(self.datas_id[item]['img_path']))
        assert os.path.exists(self.datas_id[item]['gt_path']), (
            '{} does not exist'.format(self.datas_id[item]['gt_path']))
        if self.datas_id[item]['dataset'] != 'DUTS-TR':
            assert os.path.exists(self.datas_id[item]['flow_path']), (
                '{} does not exist'.format(self.datas_id[item]['flow_path']))

        image = Image.open(self.datas_id[item]['img_path']).convert('RGB')
        label = np.array(Image.open(self.datas_id[item]['gt_path']).convert('L'))

        if self.datas_id[item]['dataset'] == 'DUTS-TR':
            flow = np.zeros((image.size[1], image.size[0], 3))
            flow = Image.fromarray(np.uint8(flow))

        else:
            flow = Image.open(self.datas_id[item]['flow_path']).convert('RGB')

        if label.max() > 0:
            label = label / 255

        w, h = image.size
        size = (h, w)

        sample = {'image': image, 'label': label, 'flow': flow}

        if self.transform:
            sample = self.transform(sample)
        if self.return_size:
            sample['size'] = torch.tensor(size)
        if self.datas_id[item]['dataset'] == 'DUTS-TR':
            sample['flow'] = torch.zeros((3, 512, 512))
        name = self.datas_id[item]['gt_path'].split('/')[-1]
        sample['dataset'] = self.datas_id[item]['dataset']
        sample['split'] = self.datas_id[item]['split']
        sample['name'] = name

        return sample

    def __len__(self):
        return len(self.datas_id)
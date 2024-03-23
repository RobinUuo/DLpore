import h5py
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MyDataset(Dataset):
    def __init__(self, subdir, transform=None, target_transform=None):
        label = []
        data_list = []
        for i in range(len(subdir)):
            lbmDiff = 'D:\\dev\\fiber DL\\lb\\diffusion-final\\lb3d\\lb3d\\out_' + subdir[i] + '\\fluxinorder.txt'
            lbmAdve = 'D:\\dev\\fiber DL\\lb\\permea-final\\lb3d\\lb3d/out_' + subdir[i] + '\\fluxinorder.txt'
            label_dict = {}
            with open(lbmDiff, 'r') as file1:
                lines1 = file1.readlines()
            for line in lines1:
                [id1, value1] = line[:-1].split('\t')
                label_dict[int(id1)] = [float(value1)]
            with open(lbmAdve, 'r') as file2:
                lines2 = file2.readlines()
            for line in lines2:
                [id2, value2] = line[:-1].split('\t')
                label_dict[int(id2)].append(float(value2))
            label.append(label_dict)
            fname = '../pydata/fgeopairs/geoparis_' + subdir[i] + '.hdf5'
            f1 = h5py.File(fname, 'r')
            g_Ratio = f1['geo_ratios'][()]
            g_Index = f1['geo_ids'][()]
            geo = f1['geo_pairs'][()]
            for j in range(len(g_Index)):
                tindex = g_Index[j]
                vdiff = label[i][tindex][0]
                vadve = label[i][tindex][1]
                g_truevalue_diff = vdiff * g_Ratio[j]
                g_truevalue_adve = vadve * g_Ratio[j] ** 3
                raw = geo[j, :, :, :]
                raw_with_channel = np.expand_dims(raw, axis=0)
                g_truevalue_diff = g_truevalue_diff.astype(np.float32)
                g_truevalue_adve = g_truevalue_adve.astype(np.float32)
                data_list.append((raw_with_channel, g_truevalue_diff, g_truevalue_adve))
            f1.close()
        self.data_list = data_list
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        raw = self.data_list[index][0]
        label = self.data_list[index][1:]
        if self.transform is not None:
            raw = torch.from_numpy(raw)
            raw = raw.to(torch.float32)
        return raw, label

    def __len__(self):
        return len(self.data_list)


data = MyDataset(['0', '10', '100'], transform=torchvision.transforms.ToTensor())
train_size = int(len(data) * 0.8)
val_size = int(len(data) * 0.1)
test_size = len(data) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, val_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, drop_last=True)

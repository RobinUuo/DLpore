import h5py
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from resnetModel import resnet50
import time
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os


class MyDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        label = []
        data_list = []

        lbmDiff = 'D:\\dev\\fiber DL\\lb\\diffusion-final\\lb3d\\lb3d\\out_test\\fluxinorder.txt'
        lbmAdve = 'D:\\dev\\fiber DL\\lb\\permea-final\\lb3d\\lb3d/out_test\\fluxinorder.txt'
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
        fname = '../pydata/fgeopairs/geoparis_test.hdf5'
        f1 = h5py.File(fname, 'r')
        g_Ratio = f1['geo_ratios'][()]
        g_Index = f1['geo_ids'][()]
        geo = f1['geo_pairs'][()]
        for j in range(len(g_Index)):
            tindex = g_Index[j]
            vdiff = label[0][tindex][0]
            vadve = label[0][tindex][1]
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


data = MyDataset(transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=data, batch_size=64, shuffle=False, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = resnet50()
net.to(device)

weight = ['adve', 'diff', 'total']
for (i, l) in enumerate(weight):
    weights_path = './' + l + '_resnet50.pth'
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    prev_diff = []
    prev_adve = []
    diff_value = []
    adve_value = []
    net.eval()
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for data in test_loader:
            imgs, true_data = data
            imgs = imgs.to(device)
            diff = true_data[0]
            adve = true_data[1]
            diff = diff.numpy()
            adve = adve.numpy()
            diff_value.append(diff)
            adve_value.append(adve)

            output1, output2 = net(imgs)
            torch.cuda.synchronize()
            end = time.time()
            output1 = output1.cpu().numpy()
            output2 = output2.cpu().numpy()
            prev_diff.append(output1)
            prev_adve.append(output2)
    print('Time-consuming:{:.2f}seconds'.format(end - start))
    r_squared_diff = r2_score(diff_value, prev_diff)
    r_squared_adve = r2_score(adve_value, prev_adve)

    # 绘制散点图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制第一个预测数据对应的散点图
    ax1.scatter(diff_value, prev_diff, color='blue', label=f'R²={r_squared_diff:.2f}')
    ax1.set_xlabel('LBM Based diff')
    ax1.set_ylabel('Deep learning based diff')
    ax1.legend()

    # 绘制第二个预测数据对应的散点图
    ax2.scatter(adve_value, prev_adve, color='red', label=f'R²={r_squared_adve:.2f}')
    ax2.set_xlabel('LBM Based adve')
    ax2.set_ylabel('Deep learning based adve')
    ax2.legend()

    plt.savefig('new' + 'with_' + l + '_weight.png')
    plt.show()

import os

import torch
from dataset import test_loader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import time

from resnetModel import resnet50, resnet101, resnet151

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
structure = '50'
if structure == '50':
    net = resnet50()
elif structure == '101':
    net = resnet101()
else:
    net = resnet151()
net = net.to(device)

weight = ['adve', 'diff', 'total']
for (i, l) in enumerate(weight):
    weights_path = './' + l + '_resnet' + structure + '.pth'
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

    plt.savefig(structure + 'with_' + l + '_weight.png')
    plt.show()

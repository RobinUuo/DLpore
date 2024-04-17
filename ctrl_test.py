import os

import torch
from dataset import test_loader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import time
import torch.nn as nn
from functools import partial
from resnetModel import resnet50
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet50()
# summary(model, input_size=(1, 64, 64, 64))

# 退回单输出层结构
model.out1[3] = nn.Linear(512, 2)
del model.out2


def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.stage2(x)
    x = self.stage3(x)
    x = self.stage4(x)
    x = self.stage5(x)

    x = self.avg(x)
    x = self.out1(x)
    return x


model.forward = partial(forward, model)
model.to(device)
weights_path = './resnet_ctrl.pth'
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
model.load_state_dict(torch.load(weights_path, map_location=device))
prev_diff = []
prev_adve = []
diff_value = []
adve_value = []
model.eval()
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for data in test_loader:
        imgs, values = data
        imgs = imgs.to(device)
        diff = values[0]
        adve = values[1]
        diff = diff.numpy()
        adve = adve.numpy()
        diff_value.append(diff)
        adve_value.append(adve)

        result = model(imgs)
        end = time.time()
        result = result.cpu().numpy()
        # 按列分割成两个一维数组
        column1 = result[:, 0].reshape(-1)
        column2 = result[:, 1].reshape(-1)
        torch.cuda.synchronize()

        prev_diff.append(column1)
        prev_adve.append(column2)
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

plt.savefig('ctrl_weight.png')
plt.show()

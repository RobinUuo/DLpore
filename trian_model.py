import sys

import torch.optim
from dataset import train_loader, val_loader
from resnetModel import resnet50
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = resnet50()
net = net.to(device)

epochs = 100

# print(net)

diff_avg_mse = 0.0
adve_avg_mse = 0.0
diff_avg_vaild_mse = 0.0
adve_avg_vaild_mse = 0.0

loss_fn = nn.MSELoss()
loss_fn = loss_fn.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

for i in range(epochs):
    net.train()

    running_loss_diff = 0.0
    running_loss_adve = 0.0

    with tqdm(total=len(train_loader)) as t:
        for step, data in enumerate(train_loader, start=0):
            imgs, values = data
            imgs = imgs.to(device)
            diff_true_value = values[0]
            diff_true_value = diff_true_value.to(device)
            adve_true_value = values[1]
            adve_true_value = adve_true_value.to(device)

            logit1, logit2 = net(imgs)
            loss_diff = loss_fn(logit1, diff_true_value)
            loss_adve = loss_fn(logit2, adve_true_value)
            running_loss_diff += loss_diff.item()
            running_loss_adve += loss_adve.item()

            t.set_description(desc='train Epoch {}/{}'.format(i + 1, epochs))
            t.set_postfix(loss=(running_loss_adve + running_loss_diff) / 2)
            t.update(1)

            optimizer.zero_grad()
            loss_diff.backward(retain_graph=True)
            loss_adve.backward()
            optimizer.step()

    net.eval()
    vailding_loss_diff = 0.0
    vailding_loss_adve = 0.0

    with torch.no_grad():
        with tqdm(total=len(val_loader)) as tqdm:
            for val_data in val_loader:
                val_images, val_values = val_data
                val_images = val_images.to(device)
                diff_val_true_value = val_values[0]
                diff_val_true_value = diff_val_true_value.to(device)
                adve_val_true_value = val_values[1]
                adve_val_true_value = adve_val_true_value.to(device)

                output1, output2 = net(val_images)
                vloss_diff = loss_fn(output1, diff_val_true_value)
                vloss_adve = loss_fn(output2, adve_val_true_value)
                vailding_loss_diff += vloss_diff.item()
                vailding_loss_adve += vloss_adve.item()

                t.set_description(desc='vaild Epoch {}/{}'.format(i + 1, epochs))
                t.set_postfix(loss=(vailding_loss_diff + vailding_loss_adve) / 2)
                t.update(1)

    diff_avg_mse = running_loss_diff / len(train_loader)
    adve_avg_mse = running_loss_adve / len(train_loader)
    diff_avg_vaild_mse = vailding_loss_diff / len(val_loader)
    adve_avg_vaild_mse = vailding_loss_adve / len(val_loader)

    print('[epoch %d] training_loss in diff :%.3f   training_loss in adve:%.3f' %
          i + 1, diff_avg_mse, adve_avg_mse)

    torch.save(net.state_dict(), './resnet50.pth')

print('Finished Training')

fig1 = plt.figure(figsize=(9, 5))
plt.yscale('log')
plt.plot(epochs, diff_avg_mse,
         linestyle='--', linewidth=3, color='orange',
         alpha=0.7, label='Train mse')
plt.plot(epochs, diff_avg_vaild_mse,
         linestyle='-.', linewidth=2, color='lime',
         alpha=0.8, label='Valid mse')
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
# plt.savefig('train_mse.png', dpi=250)
#

fig2 = plt.figure(figsize=(9, 5))
plt.yscale('log')
plt.plot(epochs, adve_avg_mse,
         linestyle='solid', linewidth=1.5, color='purple',
         alpha=0.7, label='Train mse')
plt.plot(epochs, adve_avg_vaild_mse,
         linestyle='solid', linewidth=1.5, color='goldenrod',
         alpha=0.8, label='Valid mse')

plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)

fig1.savefig('diff_mse.png')
fig2.savefig('adve_mse.png')

plt.show()

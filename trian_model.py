import sys
import os

import torch.optim
from dataset import train_loader, val_loader
from resnetModel import resnet50, resnet101, resnet151
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

structure = '50'
if structure == '50':
    net = resnet50()
elif structure == '101':
    net = resnet101()
else:
    net = resnet151()
net = net.to(device)
epochs = 100

diff_avg_mse = 0.0
adve_avg_mse = 0.0
diff_output_loss = []
val_diff_output_loss = []
adve_output_loss = []
val_adve_output_loss = []
total_output_loss = []
val_total_output_loss = []
best_epoch = [0, 0, 0]

loss_fn = nn.MSELoss()
loss_fn = loss_fn.to(device)
optimizer1 = torch.optim.Adam(net.parameters(), lr=1e-4)
optimizer2 = torch.optim.Adam(net.parameters(), lr=1e-4)

best_in_diff = float('inf')
best_in_adve = float('inf')
best_in_total = float('inf')

for i in range(epochs):
    net.train()

    running_loss_diff = 0.0
    running_loss_adve = 0.0

    with tqdm(total=len(train_loader)) as t:
        torch.autograd.set_detect_anomaly(True)

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

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss_diff.backward(retain_graph=True)
            loss_adve.backward()
            optimizer1.step()
            optimizer2.step()
            running_loss_diff += loss_diff.item()
            running_loss_adve += loss_adve.item()

            t.set_description(desc='train Epoch {}/{}'.format(i + 1, epochs))
            t.set_postfix(loss_diff=running_loss_diff / (step + 1), loss_adve=running_loss_adve / (step + 1))
            t.update(1)

    # calculate three losses for each epoch in training
    diff_avg_mse = running_loss_diff / len(train_loader)
    adve_avg_mse = running_loss_adve / len(train_loader)
    total_mse = diff_avg_mse + adve_avg_mse

    # save the best weight for each
    if diff_avg_mse < best_in_diff:
        best_in_diff = diff_avg_mse
        torch.save(net.state_dict(), './diff_resnet' + structure + '.pth')
        best_epoch[0] = i + 1

    if adve_avg_mse < best_in_adve:
        best_in_adve = adve_avg_mse
        torch.save(net.state_dict(), './adve_resnet' + structure + '.pth')
        best_epoch[1] = i + 1

    if total_mse < best_in_total:
        best_in_total = total_mse
        torch.save(net.state_dict(), './total_resnet' + structure + '.pth')
        best_epoch[2] = i + 1

    net.eval()
    vailding_loss_diff = 0.0
    vailding_loss_adve = 0.0

    with torch.no_grad():
        with tqdm(total=len(val_loader)) as tq:
            for id, val_data in enumerate(val_loader):
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

                tq.set_description(desc='vaild Epoch {}/{}'.format(i + 1, epochs))
                tq.set_postfix(vloss_diff=vailding_loss_diff / (id + 1), vloss_adve=vailding_loss_adve / (id + 1))
                tq.update(1)

    # calculate three losses for each epoch in vailding
    diff_avg_vaild_mse = vailding_loss_diff / len(val_loader)
    adve_avg_vaild_mse = vailding_loss_adve / len(val_loader)
    val_total_mse = diff_avg_vaild_mse + adve_avg_vaild_mse

    # show in the PLT
    total_output_loss.append(total_mse)
    val_total_output_loss.append(val_total_mse)
    diff_output_loss.append(diff_avg_mse)
    adve_output_loss.append(adve_avg_mse)
    val_diff_output_loss.append(diff_avg_vaild_mse)
    val_adve_output_loss.append(adve_avg_vaild_mse)

print('Finished Training')
print('diff:{};adve:{};total:{}'.format(best_epoch[0], best_epoch[1], best_epoch[2]))

loss_dict = {
    'total': total_output_loss,
    'val_total': val_total_output_loss,
    'diff': diff_output_loss,
    'adve': adve_output_loss,
    'val_diff': val_diff_output_loss,
    'val_adve': val_adve_output_loss
}

lossNames = ['total', 'diff', 'adve']
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
for (i, l) in enumerate(lossNames):
    title = 'Loss for {}'.format(l) if l != 'total' else 'Total Loss'
    ax[i].set_title(title)
    ax[i].set_xlabel('Epoch')
    ax[i].set_ylabel('Loss')
    ax[i].plot(np.arange(0, epochs), loss_dict[l], label=l)
    ax[i].plot(np.arange(0, epochs), loss_dict['val_' + l], label='val_' + l)
    ax[i].legend(fontsize=12)

plt.tight_layout()
plt.savefig(structure + '_losses.png')
plt.show()
plt.close()

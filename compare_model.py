from resnetModel import resnet50
import torch
from torchsummary import summary
import torch.nn as nn
from dataset import train_loader, val_loader
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from functools import partial

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
loss_fn = nn.MSELoss()
loss_fn = loss_fn.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 100
best_epoch = 0
train_mse = 0.0
val_mse = 0.0
best_loss = float('inf')
train_loss = []
val_loss = []

for i in range(epochs):
    model.train()

    running_loss = 0.0

    with tqdm(total=len(train_loader)) as t:
        torch.autograd.set_detect_anomaly(True)

        for step, data in enumerate(train_loader, start=0):
            imgs, values = data
            imgs = imgs.to(device)
            diff_true_value = values[0]
            diff_true_value = diff_true_value.to(device)
            adve_true_value = values[1]
            adve_true_value = adve_true_value.to(device)

            result = model(imgs)

            # 将标签合并成一个张量
            targets = torch.stack([diff_true_value, adve_true_value], dim=1)

            optimizer.zero_grad()
            loss = loss_fn(result, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            t.set_description(desc='train Epoch {}/{}'.format(i + 1, epochs))
            t.set_postfix(loss=running_loss / (step + 1))
            t.update(1)

    train_mse = running_loss / len(train_loader)

    if train_mse < best_loss:
        best_loss = train_mse
        torch.save(model.state_dict(), './resnet' + '_ctrl' + '.pth')
        best_epoch = i + 1

    model.eval()
    vailding_loss = 0.0
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as tq:
            for id, val_data in enumerate(val_loader):
                val_images, val_values = val_data
                val_images = val_images.to(device)
                diff_val_true_value = val_values[0]
                diff_val_true_value = diff_val_true_value.to(device)
                adve_val_true_value = val_values[1]
                adve_val_true_value = adve_val_true_value.to(device)

                val_result = model(val_images)
                val_target = torch.stack([diff_val_true_value, adve_val_true_value], dim=1)
                vloss = loss_fn(val_result, val_target)

                vailding_loss += vloss.item()

                tq.set_description(desc='vaild Epoch {}/{}'.format(i + 1, epochs))
                tq.set_postfix(vloss=vailding_loss / (id + 1))
                tq.update(1)

    val_mse = vailding_loss / len(train_loader)

    # show in the PLT
    train_loss.append(train_mse)
    val_loss.append(val_mse)

print('Finished Training')
print('best performace in', best_epoch)

fig = plt.figure(figsize=(9, 5))
plt.yscale('log')
plt.plot(np.arange(0, epochs), train_loss, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train mse')
plt.plot(np.arange(0, epochs), val_loss, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid mse')
# plt.ylim(0.3, 100)
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.savefig('ctrl_mse.png', dpi=250)
plt.show()

import numpy as np
import matplotlib.pyplot as plt


train_loss_file_name = 'train_loss.txt'
val_loss_file_name = 'val_loss.txt'
output_file_name = 'full_train_loss_comp.pdf'


train_loss = []
val_loss = []
with open(train_loss_file_name, 'r') as f:
    for line in f.readlines():
        train_loss.append(np.array(line.split(',')).astype(np.float))

with open(val_loss_file_name, 'r') as f:
    for line in f.readlines():
        val_loss.append(np.array(line.split(',')).astype(np.float))

train_loss = np.array(train_loss)
val_loss = np.array(val_loss)

plt.figure(figsize=(6, 6))
# train_loss_curve = plt.plot(train_loss, label='Train')
# val_loss_curve = plt.plot(val_loss, label='Validation')
train_loss_curve = plt.plot(train_loss[:, 0], train_loss[:, 1],
                            label='Train')
val_loss_curve = plt.plot(val_loss[:, 0], val_loss[:, 1], label='Validation')
plt.title('Loss')
plt.ylabel('MSE Loss')
plt.yscale('log')
plt.xlabel('Number of traing samples')

plt.legend()

plt.tight_layout()

plt.savefig(output_file_name)

# from torch.utils.tensorboard import SummaryWriter

#
# sw = SummaryWriter(log_dir='haha')
my_resnet = 'my_resnet_50.log'
origin_resnet = 'origin_resnet_50.log'

with open(my_resnet, 'r') as f:
    my_lines = f.readlines()
with open(origin_resnet, 'r') as f:
    or_lines = f.readlines()
train_epoch = 0

train_epochs = []
val_epochs = []

accs = []
losses = []
val_accs = []
val_losses = []

for line in or_lines:
    line.strip()
    bbs = line.split()
    acc = float(bbs[-1])
    loss = float(bbs[-3])
    if 'Train' in bbs:
        train_epoch += 1
        train_epochs.append(train_epoch)
        accs.append(acc)
        losses.append(loss)
    elif 'Val' in bbs:
        val_epoch = train_epoch
        val_epochs.append(val_epoch)
        if train_epoch>1:
            val_accs.append((val_accs[-1]+acc)/2)
            val_losses.append((val_losses[-1]+loss)/2)
        val_accs.append(acc)
        val_losses.append(loss)
    else:
        print(line)

print(train_epochs)
print(accs)
print(losses)
print(val_epochs)
print(val_accs)
print(val_losses)
print(len(my_lines),len(train_epochs),len(val_epochs))
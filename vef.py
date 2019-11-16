import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision.models as models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image
# import pandas as pd
import numpy as np
import os
import copy, time
import visdom

import cv2
path =  'C://ProgramData//Anaconda3//Lib//site-packages//matplotlib//mpl-data//fonts//ttf//'
#file_path = '/home/lps/yanzm'
file_path = 'D://pytorch//pokemen//code_vef//'
file_path2 = 'D:\pytorch\pokemen\code_vef'
BATCH_SIZE = 16
EPOCH = 10


# Load data
class dataset(Dataset):

    def __init__(self, root_dir, label_file, transform=None):
        self.root_dir = root_dir
        self.label = np.loadtxt(label_file)
        self.transform = transform
        #print(self.root_dir)
        #print(self.label)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, '%d.jpg' % idx)
        image = Image.open(img_name)
        labels = self.label[idx, :]
        # cv2.imshow('image',image)
        # print(labels)


        #            sample = image

        if self.transform:
            image = self.transform(image)

        return image, labels

    def __len__(self):
        return (self.label.shape[0])

file_path_pwd = os.getcwd()


file_path_test = file_path_pwd + '\code_vef\src_test'
file_path_label_test = file_path_pwd + '\code_vef\label_test.txt'

data_test = dataset(file_path_test,file_path_label_test,transform=transforms.ToTensor())
dataloader_test = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dataset_test_size = len(dataloader_test)

file_path_test = file_path_pwd + '\code_vef\src_val'
file_path_label_test = file_path_pwd + '\code_vef\label_val.txt'

data_val = dataset(file_path_test,file_path_label_test,transform=transforms.ToTensor())
dataloader_val = DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dataset_val_size = len(dataloader_val)

file_path_train = file_path_pwd + '\code_vef\src_train'
file_path_label_train = file_path_pwd + '\code_vef\label_train.txt'

data = dataset(file_path_train,file_path_label_train,transform=transforms.ToTensor())
# x, y = next(iter(data))
# print('sample:', x.shape, y.shape, y)
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dataset_size = len(data)
viz = visdom.Visdom()


# Conv network
class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=2),  # in:(bs,3,60,160)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2),  # out:(bs,32,30,80)

            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2),  # out:(bs,64,15,40)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2)  # out:(bs,64,7,20)
        )

        self.fc1 = nn.Linear(64 * 7 * 30, 500)
        self.fc2 = nn.Linear(500, 40)

    def forward(self, x):
        # print('x_enter',x.shape)
        x = self.conv(x)
        # print('x_output', x.shape)
        x = x.view(x.size(0), -1)  # reshape to (batch_size, 64 * 7 * 30)
        # print('x_reshape', x.shape)
        output = self.fc1(x)
        output = self.fc2(output)
        # print('output.shape',output.shape) #(bs, 40)

        return output


# Train the net
class nCrossEntropyLoss(torch.nn.Module):

    def __init__(self, n=4):
        super(nCrossEntropyLoss, self).__init__()
        self.n = n
        self.total_loss = 0
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, label):#output:(bs, 40) lable:(bs, 4)
        output_t = output[:, 0:10] #(bs,10)
        label = Variable(torch.LongTensor(label.data.cpu().numpy()))
        label_t = label[:, 0] #(bs,1)

        #output_t (bs,10) after cat1 (2*bs,10) after cat2 (3*bs,10) after cat4 (4*bs,10)
        #label_t  (bs,1) after cat1 (2*bs,1) after cat2 (3*bs,1) after cat4 (4*bs,1)
        for i in range(1, self.n):
            output_t = torch.cat((output_t, output[:, 10 * i:10 * i + 10]), 0)  # 损失的思路是将一张图平均剪切为4张小图即4个多分类，然后再用多分类交叉熵方损失
            label_t = torch.cat((label_t, label[:, i]), 0)
            # print('i:',i,'output_t shape:',output_t.shape)
            # print('lable shape:',label_t.shape)
        self.total_loss = self.loss(output_t, label_t)
        print('total loss:',self.total_loss)
        return self.total_loss


def equal(np1, np2):
    n = 0
    for i in range(np1.shape[0]):
        if (np1[i, :] == np2[i, :]).all():
            n += 1

    return n

def evalute(model, loader):
    correct = 0
    total = len(loader.dataset)

    for (x,y) in loader:
        with torch.no_grad():
            pred = torch.LongTensor(BATCH_SIZE, 1).zero_()
            x = Variable(x)  # (bs, 3, 60, 240)
            y = Variable(y)  # (bs, 4)
            logits = model(x)
            for i in range(4):
                pre = F.log_softmax(logits[:, 10 * i:10 * i + 10], dim=1)  # (bs, 10)
                # print(i, 'pred.shape:', pred.shape, 'pred', pred)
                # print('pre.data.max:', pre.data.max(1, keepdim=True)[1])
                pred = torch.cat((pred, pre.data.max(1, keepdim=True)[1].cpu()), dim=1)  #
            correct += equal(pred.numpy()[:, 1:], y.data.cpu().numpy().astype(int))

        # print('correct:',correct,'total:',total)
    return correct / total
def main():
    net = ConvNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nCrossEntropyLoss()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    best_epoch = 0
    global_step = 0
    viz.line([0],[-1],win='loss',opts=dict(title='loss'))
    viz.line([0],[-1],win='acc',opts=dict(title='val_acc'))
    since = time.time()
    for epoch in range(EPOCH):
        running_loss = 0.0
        running_corrects = 0
        testing_corrects = 0

        for step, (inputs, label) in enumerate(dataloader):
            # print('inputs.shape',inputs.shape)
            # print(label)
            pred = torch.LongTensor(BATCH_SIZE, 1).zero_()
            inputs = Variable(inputs) # (bs, 3, 60, 240)
            label = Variable(label) # (bs, 4)

            optimizer.zero_grad()

            output = net(inputs)  # (bs, 40)
            #print('output:',output.shape)
            #print('label:',label.shape)
            loss = loss_func(output, label)

            # for i in range(4):
            #     pre = F.log_softmax(output[:, 10 * i:10 * i + 10], dim=1)  # (bs, 10)
            #     # print(i,'pred.shape:',pred.shape,'pred',pred)
            #     # print('pre.data.max:',pre.data.max(1, keepdim=True)[1])
            #     pred = torch.cat((pred, pre.data.max(1, keepdim=True)[1].cpu()), dim=1)  #

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size()[0]
            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1
            # running_corrects += equal(pred.numpy()[:, 1:], label.data.cpu().numpy().astype(int))

        epoch_loss = running_loss / dataset_size
        # epoch_acc = running_corrects / dataset_size

        if epoch % 1 == 0:
            val_acc = evalute(net, dataloader_test)
            print('epoch:', epoch,'val_acc:',val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                #torch.save(best_model_wts,'best_model_wts.pkl')
                torch.save(net.state_dict(), 'best.mdl')
                viz.line([val_acc], [global_step], win='val_acc', update='append')

        # if epoch == EPOCH - 1:
        #     torch.save(best_model_wts, file_path + '/best_model_wts.pkl')

        # print('EPOCH:',epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Train Loss:{:.4f}'.format(epoch_loss))

    print('best_acc:', best_acc, 'best_epoch:', best_epoch)
    net.load_state_dict(torch.load('best.mdl'))
    #net.load_state_dict(torch.load('best_model_wts.pkl'))
    print('loaded from ckpt!')

    test_acc = evalute(net, dataloader_val)
    print('test_acc:', test_acc)

if __name__ == '__main__':
    main()
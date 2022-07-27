from torchvision import transforms
from torch import optim
import torch
import torch.nn as nn
import os
import numpy as np
class fun:
    def pretreatment(data:torch.tensor,SAVE=False):
        data_reshape = data.permute(2, 1, 0).contiguous()
        # print(data_reshape.shape)
        # print(data.shape)
        if SAVE:
            data_mean = data_reshape.view(8, -1).mean(dim=1)
            data_std = data_reshape.view(8, -1).std(dim=1)
            torch.save(data_mean,'parameters/data_mean.t')
            torch.save(data_std,'parameters/data_std.t')
        else:
            data_mean=torch.load('parameters/data_mean.t')
            data_std=torch.load('parameters/data_std.t')
        print(data_mean)
        print(data_std)
        transform_train = transforms.Compose(
            [transforms.Normalize(mean=data_mean, std=data_std)]
        )
        data_reshape = transform_train(data_reshape)
        # print(data_mean)
        # print(data_std)
        data = data_reshape.permute(2, 0, 1).contiguous()
        # print(data.shape)
        # print(data)
        return data
    def ReadFile(filepath):
        file_path = filepath
        filenames=[name for name in os.listdir(file_path)
        if os.path.splitext(name)[-1]=='.csv']
        data_size=len(filenames)
        data=torch.zeros(data_size,264,8)
        for i,filename in enumerate(filenames):
            data_numpy = np.loadtxt(os.path.join(file_path,filename),
                dtype=np.float32,delimiter=",",skiprows=1,
                max_rows=264,usecols=(1,2,3,4,13,14,15,16))
            data[i]=torch.from_numpy(data_numpy)
        return data,filenames

class Net:
    def __init__(self,train_loader):
        self.model=nn.Sequential(
            nn.Linear(264*8,1024),
            nn.Tanh(),
            nn.Linear(1024,128),
            nn.Tanh(),
            nn.Linear(128,2),

            nn.LogSoftmax(dim=1)
        )
        self.train_loader=train_loader
        self.learning_rate=1e-2
        self.optimizer=optim.SGD(self.model.parameters(),lr=self.learning_rate)
        self.loss_fn=nn.NLLLoss()
        self.n_epochs=300
    def training_loop(self):
        for epochs in range(self.n_epochs):
            for data,labels in self.train_loader:
                batch_size=data.shape[0]
#                for i in range(batch_size):
#                    if labels[i]==0:
#                       data[i]=torch.randn(8,264)
                outputs=self.model(data.view(batch_size,-1))
                loss=self.loss_fn(outputs,labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print("Epoch %d, Loss %f" % (epochs,float(loss)))
        torch.save(self.model,'parameters/model.m')
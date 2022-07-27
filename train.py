import numpy as np
import torch
torch.set_printoptions(edgeitems=2, precision=2, linewidth=75)

from train_utils import fun


train_data,_=fun.ReadFile('data/1_armup')
fake_data,_=fun.ReadFile('data/error')
rand_data=torch.randn(15,264,8)

train_data=fun.pretreatment(data=train_data,SAVE=True)
fake_data=fun.pretreatment(data=fake_data)
#rand_data=fun.pretreatment(data=rand_data)

torch.save(train_data,'parameters/train_data.t')
torch.save(fake_data,'parameters/fake_data.t')

run_data=[(torch.cat([train_data,fake_data])[i],1 if i<train_data.shape[0] else 0)
    for i in range(train_data.shape[0]+fake_data.shape[0])]
train_loader=torch.utils.data.DataLoader(run_data,batch_size=5,shuffle=True)

from train_utils import Net
net=Net(train_loader)
net.training_loop()